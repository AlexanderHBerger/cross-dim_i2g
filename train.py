from functools import partial
import os
import yaml
import json
from argparse import ArgumentParser
from shutil import copyfile
import torch
from data.dataset_mixed import build_mixed_data

from data.dataset_road_network import build_road_network_data
from data.dataset_synth_octa_network import build_octa_network_data
from data.dataset_vessel3d import build_vessel_data
from training.evaluator import build_evaluator
from training.trainer import build_trainer
from models import build_model
from utils.utils import image_graph_collate
from models.matcher import build_matcher
from training.losses import EDGE_SAMPLING_MODE, SetCriterion
import torch.distributed as dist
import ignite.distributed as igdist
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from torch.utils.tensorboard import SummaryWriter
from monai.data import DataLoader

parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config. See /config for examples.')
parser.add_argument('--resume', default=None, help='checkpoint of the last epoch of the model')
parser.add_argument('--restore_state', dest='restore_state', help='whether the state should be restored', action='store_true')
parser.add_argument('--device', default='cuda',
                        help='device to use for training')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0,1],
                        help='list of index where skip conn will be made')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--nproc_per_node", default=None, type=int)
parser.add_argument('--debug', dest='debug', help='do fast debug', action='store_true')  #TODO: remove
parser.add_argument('--exp_name', dest='exp_name', help='name of the experiment', type=str,required=True)
parser.add_argument('--pre2d', dest='pre2d', help='Pretraining the network using 2D-segmentation after backbone', action="store_true")
parser.add_argument('--use_retina', dest='use_retina', help='Use synthetic retina data instead of street data for pretraining', action="store_true")
parser.add_argument('--continuous', dest='continuous', help='Using continuous rotation for gaussian augment', action="store_true")
parser.add_argument("--parallel", dest='parallel', help='Using distributed training', action="store_true")
parser.add_argument("--ignore_backbone",  dest='ignore_backbone', help='', action="store_true")
parser.add_argument('--seg', dest='seg', help='Using segmentation or raw images', action="store_true")
parser.add_argument('--general_transformer', dest='general_transformer', help='Using a general transformer with an adapter', action="store_true")
parser.add_argument('--pretrain_general', dest='pretrain_general', help='Pretraining the general transformer with 2D data', action="store_true")
parser.add_argument('--load_only_decoder', dest='load_only_decoder', help='When resuming, only load state from decoder instead of whole model. Useful when doing multi-modal transfer learning', action="store_true")
parser.add_argument('--no_strict_loading', default=False, action="store_true",
                    help="Whether the model was pretrained with domain adversarial. If true, the checkpoint will be loaded with strict=false")
parser.add_argument('--sspt', default=False, action="store_true",
                    help="Whether the model was pretrained with self supervised pretraining. If true, the checkpoint will be loaded accordingly. Only combine with resume.")




class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def main(rank=0, args=None):
    
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(args.exp_name)
    config = dict2obj(config)
    config.log.exp_name = args.exp_name

    exp_path = os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (args.exp_name, config.DATA.SEED))
    if os.path.exists(exp_path) and args.resume == None:
        print('ERROR: Experiment folder exist, please change exp name in config file')
    else:
        try:
            os.makedirs(exp_path)
            copyfile(args.config, os.path.join(exp_path, "config.yaml"))
        except FileExistsError:
            copyfile(args.config, os.path.join(exp_path, "config2.yaml"))

    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enabled = True
    #torch.multiprocessing.set_sharing_strategy('file_system')
    # device = torch.device("cuda") if args.device=='cuda' else torch.device("cpu")
    args.distributed = False
    args.rank = rank  # args.rank = int(os.environ["RANK"])
    if args.parallel and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.gpu = int(os.environ["LOCAL_RANK"])  # args.gpu = 'cuda:%d' % args.local_rank
        args.world_size = int(os.environ['WORLD_SIZE'])  # igdist.get_world_size()
        print('Running Distributed:',args.distributed, '; GPU:', args.gpu, '; RANK:', args.rank)

    if args.parallel and igdist.get_local_rank() > 0:
        # Ensure that only local rank 0 download the dataset
        # Thus each node will download a copy of the dataset
        igdist.barrier()

    if config.DATA.DATASET == "mixed_synth_3d" or config.DATA.DATASET == "mixed_real_vessels" or config.DATA.DATASET == "mixed_real_vessels_octa" or config.DATA.DATASET == "mixed_synth_3d_octa":
        dataset_func = partial(build_mixed_data, upsample_target_domain=config.TRAIN.UPSAMPLE_TARGET_DOMAIN)
            
        train_ds, val_ds, sampler = dataset_func(
            config,
            mode='split',
            debug=args.debug,
            rotate=True,
            continuous=args.continuous,
            split=0.8
        )
        config.DATA.MIXED = True
    elif config.DATA.DATASET == "road_dataset":
        train_ds, val_ds, sampler = build_road_network_data(
            config, 
            mode='split', 
            split=0.8, 
            debug=args.debug, 
            max_samples=config.DATA.NUM_SOURCE_SAMPLES, 
            domain_classification=-1, 
            gaussian_augment=True, 
            rotate=True, 
            continuous=args.continuous
        ) # type: ignore
        config.DATA.MIXED = False
    elif config.DATA.DATASET == "synth_octa":
        train_ds, val_ds, sampler = build_octa_network_data(
            config, 
            mode='split', 
            split=0.8, 
            debug=args.debug, 
            max_samples=config.DATA.NUM_SOURCE_SAMPLES, 
            domain_classification=-1, 
            gaussian_augment=True, 
            rotate=True, 
            continuous=args.continuous
        ) # type: ignore
        config.DATA.MIXED = False
    else:
        config.DATA.MIXED = False
        train_ds, val_ds, sampler = build_vessel_data(
            config, 
            mode='split', 
            debug=args.debug, 
            max_samples=config.DATA.NUM_SOURCE_SAMPLES, 
            split=0.8
        )

    if args.parallel and igdist.get_local_rank() == 0:
        # Ensure that only local rank 0 download the dataset
        igdist.barrier()

    if args.parallel:
        dataloader_func = igdist.auto_dataloader
    else:
        dataloader_func = DataLoader

    train_loader = dataloader_func(train_ds,
                              batch_size=config.DATA.BATCH_SIZE,
                              shuffle=not sampler,
                              sampler=sampler,
                              num_workers=config.DATA.NUM_WORKERS,
                              collate_fn=lambda x:
                                image_graph_collate(x, args.pre2d, gaussian_augment=config.DATA.MIXED),
                              pin_memory=True)
    
    val_loader = dataloader_func(val_ds,
                            batch_size=config.DATA.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.DATA.NUM_WORKERS,
                            collate_fn=lambda x:
                                image_graph_collate(x, args.pre2d, gaussian_augment=config.DATA.MIXED),
                            pin_memory=True)

    device = torch.device(args.device)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        # dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
        args.rank = igdist.get_rank()
        device = torch.device(f"cuda:{args.rank}")

    if args.general_transformer:
        net = build_model(
            config,
            general_transformer=True,
            pretrain=args.pretrain_general,
            output_dim=2 if args.pretrain_general else 3
        )
    else:
        net = build_model(config, pre2d=args.pre2d)

    net_wo_dist = net.to(device)
    relation_embed = net.relation_embed.to(device)

    if args.distributed:
        net = igdist.auto_model(net)
        relation_embed = igdist.auto_model(relation_embed)
        net_wo_dist = net.module

    matcher = build_matcher(config, dims=2 if args.pretrain_general else 3)
    if config.TRAIN.EDGE_SAMPLING_MODE == "none":
        edge_sampling_mode = EDGE_SAMPLING_MODE.NONE
    elif config.TRAIN.EDGE_SAMPLING_MODE == "up":
        edge_sampling_mode = EDGE_SAMPLING_MODE.UP
    elif config.TRAIN.EDGE_SAMPLING_MODE == "down":
        edge_sampling_mode = EDGE_SAMPLING_MODE.DOWN
    elif config.TRAIN.EDGE_SAMPLING_MODE == "random_up":
        edge_sampling_mode = EDGE_SAMPLING_MODE.RANDOM_UP
    else:
        raise ValueError("Invalid edge sampling mode")

    loss = SetCriterion(
        config,
        matcher,
        relation_embed,
        dims=2 if args.pretrain_general else 3,
        num_edge_samples=config.TRAIN.NUM_EDGE_SAMPLES,
        edge_sampling_mode=edge_sampling_mode,
        domain_class_weight=torch.tensor(config.TRAIN.DOMAIN_WEIGHTING, device=device)
    )
    val_loss = SetCriterion(config, matcher, relation_embed, dims=2 if args.pretrain_general else 3, edge_sampling_mode=EDGE_SAMPLING_MODE.NONE)

    param_dicts = [
        {
            "params": [p for n, p in net.named_parameters() if match_name_keywords(n, ["domain_discriminator"]) and p.requires_grad],
            "lr": float(config.TRAIN.DOMAIN_LR),
            "weight_decay": float(config.TRAIN.WEIGHT_DECAY)
        },
        {
            "params": [p for n, p in net.named_parameters() if not match_name_keywords(n, ["domain_discriminator"]) and p.requires_grad],
            "lr": float(config.TRAIN.BASE_LR),
            "weight_decay": float(config.TRAIN.WEIGHT_DECAY)
        },
    ]


    optimizer = torch.optim.AdamW(
        param_dicts, lr=float(config.TRAIN.BASE_LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY)
    )

    if args.distributed:
        optimizer = igdist.auto_optim(optimizer)

    # LR schedular
    iter_per_epoch = len(train_loader)
    num_warmup_epoch = float(config.TRAIN.WARMUP_EPOCHS)
    warm_lr_init = float(config.TRAIN.WARMUP_LR)
    warm_lr_final = float(config.TRAIN.BASE_LR)
    num_warmup_iter = num_warmup_epoch * iter_per_epoch
    num_after_warmup_iter = config.TRAIN.EPOCHS * iter_per_epoch
    def lr_lambda_polynomial(iter: int):
        if iter < num_warmup_epoch * iter_per_epoch:
            lr_lamda0 = warm_lr_init / warm_lr_final
            return lr_lamda0 + (1 - lr_lamda0) * iter / num_warmup_iter
        else:
            # The total number of epochs is num_warmup_epoch + max_epochs
            return (1 - (iter - num_warmup_iter) / num_after_warmup_iter) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_polynomial)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if args.sspt:
            checkpoint = {k: v for k,v in checkpoint["state_dict"].items() if k.startswith("momentum_encoder")}
            #net.load_state_dict(checkpoint, strict=False)
        else:
            if args.ignore_backbone:
                state_dict = net_wo_dist.decoder.state_dict()
                checkpoint['net'] = {k: v for k, v in checkpoint['net'].items() if k in state_dict}
            if args.load_only_decoder:
                checkpoint['net'] = {k[8:]: v for k, v in checkpoint['net'].items() if k.startswith("decoder")}
                net_wo_dist.decoder.load_state_dict(checkpoint['net'], strict=not args.ignore_backbone and not args.no_strict_loading)
            else:
                net_wo_dist.load_state_dict(checkpoint['net'], strict=not args.ignore_backbone and not args.no_strict_loading)
            if args.restore_state:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])

    writer = SummaryWriter(
        log_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (args.exp_name, config.DATA.SEED)),
    )

    evaluator = build_evaluator(
        val_loader,
        net,
        optimizer,
        scheduler,
        writer,
        config,
        device,
        val_loss,
        pre_2d=args.pre2d,
        pretrain_general=args.pretrain_general,
        gaussian_augment=config.DATA.MIXED,
        seg=args.seg
    )
    trainer = build_trainer(
        train_loader,
        net,
        loss,
        optimizer,
        scheduler,
        writer,
        evaluator,
        config,
        device,
        seg=args.seg
        # fp16=args.fp16,
    )

    if args.resume and args.restore_state:
        last_epoch = int(scheduler.last_epoch/trainer.state.epoch_length)
        evaluator.state.epoch = last_epoch
        trainer.state.epoch = last_epoch
        trainer.state.iteration = trainer.state.epoch_length * last_epoch
    if not args.parallel or dist.get_rank()==0:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform= lambda x: {'loss': x["loss"]["total"]})
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    trainer.run()

def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out



if __name__ == '__main__':
    args = parser.parse_args()

    if args.parallel:
        with igdist.Parallel(backend='nccl', nproc_per_node=args.nproc_per_node) as parallel:
            parallel.run(main, args)
    else:
        main(args=args)
