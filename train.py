from functools import partial
import os
import yaml
import json
import shutil
from argparse import ArgumentParser
from monai.handlers import EarlyStopHandler
import torch
from monai.data import DataLoader
from data.dataset_mixed import build_mixed_data
from data.dataset_road_network import build_road_network_data
from data.dataset_synthetic_eye_vessels import build_synthetic_vessel_network_data
from data.dataset_real_eye_vessels import build_real_vessel_network_data
from training.evaluator import build_evaluator
from training.trainer import build_trainer
from models import build_model
from utils.utils import image_graph_collate_road_network
from torch.utils.tensorboard import SummaryWriter
from models.matcher import build_matcher
from training.losses import EDGE_SAMPLING_MODE, SetCriterion
from ignite.contrib.handlers.tqdm_logger import ProgressBar

parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config. See /config for examples.')
parser.add_argument('--resume', default=None,
                    help='checkpoint of the last epoch of the model')
parser.add_argument('--restore_state', dest='restore_state', help='whether the state should be restored', action='store_true')
parser.add_argument('--seg_net', default=None,
                    help='checkpoint of the segmentation model')
parser.add_argument('--device', default='cuda',
                    help='device to use for training')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=None,
                    help='list of index where skip conn will be made')
parser.add_argument('--recover_optim', default=False, action="store_true",
                    help="Whether to restore optimizer's state. Only necessary when resuming training.")
parser.add_argument('--exp_name', dest='exp_name', help='name of the experiment', type=str,required=True)
parser.add_argument('--pretrain_seg', default=False, action="store_true",
                    help="Whether to pretrain on segs instead of raw images")
parser.add_argument('--no_strict_loading', default=False, action="store_true",
                    help="Whether the model was pretrained with domain adversarial. If true, the checkpoint will be loaded with strict=false")
parser.add_argument('--sspt', default=False, action="store_true",
                    help="Whether the model was pretrained with self supervised pretraining. If true, the checkpoint will be loaded accordingly. Only combine with resume.")


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def main(args):

    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(args.exp_name)
    config = dict2obj(config)
    config.log.exp_name = args.exp_name

    if args.cuda_visible_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
            map(str, args.cuda_visible_device))
        print(os.environ["CUDA_VISIBLE_DEVICES"])

    exp_path = os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (
        config.log.exp_name, config.DATA.SEED))
    if os.path.exists(exp_path):
        print('ERROR: Experiment folder exist, please change exp name in config file')
    else:
        try:
            os.makedirs(exp_path)
            shutil.copyfile(args.config, os.path.join(exp_path, "config.yaml"))
        except:
            pass

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda") if args.device == 'cuda' else torch.device("cpu")

    if config.DATA.DATASET == 'road_dataset':
        build_dataset_function = build_road_network_data
        config.DATA.MIXED = False
    elif config.DATA.DATASET == 'synthetic_eye_vessel_dataset':
        build_dataset_function = build_synthetic_vessel_network_data
        config.DATA.MIXED = False
    elif config.DATA.DATASET == 'real_eye_vessel_dataset':
        build_dataset_function = build_real_vessel_network_data
        config.DATA.MIXED = False
    elif config.DATA.DATASET == 'mixed_road_dataset' or config.DATA.DATASET == 'mixed_synthetic_eye_vessel_dataset' or config.DATA.DATASET == "mixed_real_eye_vessel_dataset":
        build_dataset_function = partial(build_mixed_data, upsample_target_domain=config.TRAIN.UPSAMPLE_TARGET_DOMAIN)
        config.DATA.MIXED = True

    train_ds, val_ds, sampler = build_dataset_function(
        config, mode='split', use_grayscale=args.pretrain_seg, max_samples=config.DATA.NUM_SOURCE_SAMPLES, split=0.8
    )

    train_loader = DataLoader(train_ds,
                              batch_size=config.DATA.BATCH_SIZE,
                              shuffle=not sampler,
                              num_workers=config.DATA.NUM_WORKERS,
                              collate_fn=image_graph_collate_road_network,
                              pin_memory=True,
                              sampler=sampler)

    val_loader = DataLoader(val_ds,
                            batch_size=config.DATA.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.DATA.NUM_WORKERS,
                            collate_fn=image_graph_collate_road_network,
                            pin_memory=True)

    net = build_model(config).to(device)

    seg_net = build_model(config).to(device)

    matcher = build_matcher(config)

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
        net,
        num_edge_samples=config.TRAIN.NUM_EDGE_SAMPLES,
        edge_sampling_mode=edge_sampling_mode,
        domain_class_weight=torch.tensor(config.TRAIN.DOMAIN_WEIGHTING, device=device)
    )
    val_loss = SetCriterion(config, matcher, net, num_edge_samples=9999, edge_sampling_mode=False)

    param_dicts = [
        {
            "params":
                [p for n, p in net.named_parameters()
                 if not match_name_keywords(n, ["encoder.0"]) and not match_name_keywords(n, ['reference_points', 'sampling_offsets']) and not match_name_keywords(n, ["domain_discriminator"]) and p.requires_grad],
            "lr": float(config.TRAIN.LR),
            "weight_decay": float(config.TRAIN.WEIGHT_DECAY)
        },
        {
            "params": [p for n, p in net.named_parameters() if match_name_keywords(n, ["encoder.0"]) and p.requires_grad],
            "lr": float(config.TRAIN.LR_BACKBONE),
            "weight_decay": float(config.TRAIN.WEIGHT_DECAY)
        },
        {
            "params": [p for n, p in net.named_parameters() if match_name_keywords(n, ['reference_points', 'sampling_offsets']) and p.requires_grad],
            "lr": float(config.TRAIN.LR)*0.1,
            "weight_decay": float(config.TRAIN.WEIGHT_DECAY)
        },
        {
            "params": [p for n, p in net.named_parameters() if match_name_keywords(n, ['domain_discriminator']) and p.requires_grad],
            "lr": float(config.TRAIN.LR_DOMAIN),
            "weight_decay": float(config.TRAIN.WEIGHT_DECAY)
        }
    ]

    optimizer = torch.optim.AdamW(
        param_dicts, lr=float(config.TRAIN.LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY)
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, config.TRAIN.LR_DROP)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')

        if args.sspt:
            checkpoint['state_dict'] = {k[17:]: v for k, v in checkpoint['state_dict'].items() if k.startswith("momentum_encoder")}
            net.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            net.load_state_dict(checkpoint['net'], strict=not args.no_strict_loading)
            if args.recover_optim:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if args.restore_state:
                scheduler.load_state_dict(checkpoint['scheduler'])
                last_epoch = scheduler.last_epoch
                scheduler.step_size = config.TRAIN.LR_DROP

    for param_group in optimizer.param_groups:
        print(f'lr: {param_group["lr"]}, number of params: {len(param_group["params"])}')

    if args.seg_net:
        checkpoint = torch.load(args.seg_net, map_location='cpu')
        seg_net.load_state_dict(checkpoint['net'])
        # net.load_state_dict(checkpoint['net'])
    #     # 1. filter out unnecessary keys
    #     pretrained_dict = {k: v for k, v in checkpoint.items() if match_name_keywords(k, ["encoder.0"])}
    #     # 3. load the new state dict
    #     net.load_state_dict(pretrained_dict, strict=False)
    #     # net.load_state_dict(checkpoint['net'], strict=False)
    #     # for param in seg_net.parameters():
        #     param.requires_grad = False

    writer = SummaryWriter(
        log_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (
            config.log.exp_name, config.DATA.SEED)),
    )

    """early_stop_handler = EarlyStopHandler(
            patience=15,
            score_function=lambda x: -x.state.output["loss"]["total"].item()
    )"""

    evaluator = build_evaluator(
        val_loader,
        net,
        val_loss,
        optimizer,
        scheduler,
        writer,
        config,
        device,
        #early_stop_handler
    )
    trainer = build_trainer(
        train_loader,
        net,
        seg_net,
        loss,
        optimizer,
        scheduler,
        writer,
        evaluator,
        config,
        device,
        # fp16=args.fp16,
    )

    #early_stop_handler.set_trainer(trainer)

    if args.resume and args.restore_state:
        evaluator.state.epoch = last_epoch
        trainer.state.epoch = last_epoch
        trainer.state.iteration = trainer.state.epoch_length * last_epoch

    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {
                'loss': x["loss"]["total"].item()})
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

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    main(args)
