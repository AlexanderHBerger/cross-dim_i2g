import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from data.dataset_real_eye_vessels import build_real_vessel_network_data
from data.dataset_road_network import build_road_network_data
from data.dataset_synthetic_eye_vessels import build_synthetic_vessel_network_data
import math

def build_mixed_data(config, mode='split', split=0.95, upsample_target_domain=True, max_samples=-1, use_grayscale=False):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """
    config.DATA.DATA_PATH = config.DATA.SOURCE_DATA_PATH
    source_train_data, source_val_data, _ = build_road_network_data(config, mode, split, config.DATA.NUM_SOURCE_SAMPLES, use_grayscale=config.DATA.TARGET_DATA_PATH != "mixed_road_dataset", domain_classification=0)

    config.DATA.DATA_PATH = config.DATA.TARGET_DATA_PATH
    if config.DATA.DATASET == "mixed_road_dataset":
        target_train_data, target_val_data, _ = build_road_network_data(config, mode, split, config.DATA.NUM_TARGET_SAMPLES, use_grayscale, domain_classification=1)
    elif config.DATA.DATASET == "mixed_synthetic_eye_vessel_dataset":
        target_train_data, target_val_data, _ = build_synthetic_vessel_network_data(config, mode, split, config.DATA.NUM_TARGET_SAMPLES, use_grayscale, domain_classification=1)
    elif config.DATA.DATASET == "mixed_real_eye_vessel_dataset":
        target_train_data, target_val_data, _ = build_real_vessel_network_data(config, mode, split, config.DATA.NUM_TARGET_SAMPLES, use_grayscale, domain_classification=1)

    # Calculate the number of samples in each dataset
    num_samples_A = len(source_train_data)
    num_samples_B = len(target_train_data)

    # Calculate the weights for each sample in each dataset
    weights_A = torch.ones(num_samples_A)
    weights_B = torch.ones(num_samples_B) * (num_samples_A / num_samples_B)

    train_ds = ConcatDataset([source_train_data, target_train_data])
    val_ds = ConcatDataset([source_val_data, target_val_data])

    print(f"samples A: {num_samples_A}")
    print(f"samples B: {num_samples_B}")
    print(f"weight sum a: {torch.sum(weights_A)}")
    print(f"weight sum B: {torch.sum(weights_B)}")

    if upsample_target_domain:
        print("upsampling")
        sampler = WeightedRandomSampler(torch.cat([weights_A, weights_B]), num_samples=math.floor(torch.sum(weights_A) + torch.sum(weights_B)))
    else:
        print("not upsampling")
        sampler = None

    return train_ds, val_ds, sampler

    
