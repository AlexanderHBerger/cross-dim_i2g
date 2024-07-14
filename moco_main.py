from argparse import ArgumentParser
import gc
import json
import math
import shutil
import monai
import torch
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import wandb
import yaml
from moco.datasets.dataset_vessel3d import build_moco_vessel_data

from moco.model import MoCo

os.environ['WANDB_API_KEY'] = '94ab84459aa1b42734e2980087f053e645c271e7'

parser = ArgumentParser()
parser.add_argument('--exp_name', dest='exp_name', help='name of the experiment', type=str,required=True)
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. ',
                    required=True)
parser.add_argument('--resume', default=None,
                    help='checkpoint of the last epoch of the model')
parser.add_argument('--device', default='cuda',
                    help='device to use for training')
parser.add_argument('--disable_wandb', default=False, action='store_true',
                    help='disable wandb logging')

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

    config.DATA.MIXED = False

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda") if args.device == 'cuda' else torch.device("cpu")

    # Load model
    model = MoCo(config, device, dim=256, mlp_dim=4096, T=0.2)
    model.to(device)

    # Load optimizer
    optimizer = torch.optim.AdamW(model.parameters(), float(config.TRAIN.BASE_LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY))

    if not args.disable_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="relationformer_moco",
            
            # track hyperparameters and run metadata
            config={
                "exp_name": args.exp_name,
                "learning_rate": float(config.TRAIN.BASE_LR),
                "epochs": config.TRAIN.EPOCHS,
                "batch_size": config.DATA.BATCH_SIZE,
                "accumulated batch size": config.DATA.TARGET_BATCH_SIZE,
                "dataset": config.DATA.DATASET,
            }
        )

    augmentation1 = [
        # Random Gaussian Noise
        monai.transforms.RandGaussianNoise(prob=0.2, mean=0, std=0.1),
        # Random Gaussian Smooth
        monai.transforms.RandGaussianSmooth(prob=0.1),
    ]

    augmentation2 = [
        # Random Gaussian Noise
        monai.transforms.RandGaussianNoise(prob=0.2, mean=0, std=0.1),
        # Random Gaussian Smooth
        monai.transforms.RandGaussianSmooth(prob=0.1),
    ]

    # Create dataset and dataloader
    train_ds = build_moco_vessel_data(config, transforms.Compose(augmentation1), transforms.Compose(augmentation2), max_samples=config.DATA.NUM_SOURCE_SAMPLES)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.DATA.BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    accum_iter = config.DATA.TARGET_BATCH_SIZE // config.DATA.BATCH_SIZE

    # Create tensor to store ks of accum_iter batches
    k1s = torch.ones((config.DATA.TARGET_BATCH_SIZE, 256), dtype=torch.float32, device=device)
    k2s = torch.ones((config.DATA.TARGET_BATCH_SIZE, 256), dtype=torch.float32, device=device)

    # Training Loop
    for epoch in range(config.TRAIN.EPOCHS):
        print("epoch ", epoch)
        train(train_loader, model, optimizer, epoch, args, device, accum_iter, config)

        # Check every 10th epoch
        if (epoch + 1) % 10 == 0:
            # Save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename=f'trained_weights/runs/{args.exp_name}_{config.DATA.SEED}/checkpoint_%04d.pth.tar' % epoch)

        

def train(train_loader, model, optimizer, epoch, args, device, accum_iter, config):
    # switch to train mode
    model.train()

    # Accumulate qs and ks for accum_iter to compute loss
    k1s = []
    k2s = []
    x1s = []
    x2s = []

    for iteration, (x1, x2) in enumerate(tqdm(train_loader)):
        moco_m = 0.99
        x1s.append(x1)
        x2s.append(x2)

        with torch.no_grad():
            # Move data to device
            x1 = x1.to(device)
            x2 = x2.to(device)

            model.update_momentum_encoder(moco_m)  # update the momentum encoder

            # Compute ks because they're needed for loss calculation
            k1, k2 = model.compute_ks(x1, x2, moco_m)
            k1s.append(k1)
            k2s.append(k2)

            x1.cpu()
            x2.cpu()

        torch.cuda.empty_cache()
        gc.collect()

        # When all ks are computed, compute qs and loss, and update model
        if ((iteration + 1) % accum_iter == 0) or (iteration + 1 == len(train_loader)):
            # Concatenate ks 
            k1, k2 = torch.cat(k1s, dim=0).to(device), torch.cat(k2s, dim=0).to(device)
            if k1.shape[0] < config.DATA.TARGET_BATCH_SIZE / 4:
                print("k1 shape: ", k1.shape)
                print("skipping last batches")
                continue

            # Loop over all seen samples and compute qs and loss
            for i, (x1, x2) in enumerate(zip(x1s, x2s)):
                # adjust learning rate and momentum coefficient per iteration
                lr = adjust_learning_rate(optimizer, epoch + (iteration - len(x1s) + i) / len(train_loader), config.TRAIN.EPOCHS, float(config.TRAIN.BASE_LR))
                x1 = x1.to(device)
                x2 = x2.to(device)

                q1, q2 = model(x1, x2)

                # compute loss
                loss = model.contrastive_loss(q1, k2, i) + model.contrastive_loss(q2, k1, i)
                # Scale loss
                loss = loss / len(x1s)

                loss.backward()
            
                if not args.disable_wandb:
                    # print(x1.shape)
                    # conc_views = torch.cat([x1[0, 0].cpu(), x2[0, 0].cpu()], dim=0)
                    # wandb_img_logs = []

                    # for img_slice_no in range(32):
                    #     img = conc_views[:, :, img_slice_no]
                    #     wandb_img_logs.append(wandb.Image(img, caption=f"Slice: {img_slice_no}"))

                    # wandb.log({"Image Views": wandb_img_logs})

                    # Report statistics
                    wandb.log({"loss": loss.item()})
                    wandb.log({"learning_rate": lr})

                del q1, q2, x1, x2, loss
                torch.cuda.empty_cache()
                gc.collect()

            optimizer.step()
            optimizer.zero_grad()

            x1s, x2s, k1s, k2s = [], [], [], []

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, max_epochs, base_lr):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < 10:
        lr = base_lr * epoch / 10 
    else:
        lr = base_lr * 0.5 * (1. + math.cos(math.pi * (epoch - 10) / (max_epochs - 10)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    args = parser.parse_args()

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    main(args)
