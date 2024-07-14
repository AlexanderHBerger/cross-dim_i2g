import torch
from monai.data import Dataset
from medpy.io import load
import os
from utils.utils import rotate_image, rotate_coordinates

class MoCo_3DSet(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, data, transform1, transform2, patch_size=(64,64,64), normalize_image=1.):
        self.data = data
        self.patch_size = patch_size
        self.normalize_image = normalize_image
        self.transform1 = transform1
        self.transform2 = transform2
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_data, _ = load(self.data[idx])
        image_data = torch.tensor(image_data, dtype=torch.float).unsqueeze(0)
        image_data = image_data / self.normalize_image - 0.5

        image1 = self.transform1(image_data)
        image2 = self.transform2(image_data)

        return image1, image2


def build_moco_vessel_data(config, transform1, transform2, max_samples=0):
    path = config.DATA.DATA_PATH
    nifti_folder = os.path.join(path, 'raw')
    train_files = []

    for file_ in os.listdir(nifti_folder):
        file_ = file_[:-7]
        train_files.append(os.path.join(nifti_folder, file_+'.nii.gz'))
    
    if max_samples > 0:
        train_files = train_files[:max_samples]

    ds = MoCo_3DSet(
        data=train_files,
        transform1=transform1,
        transform2=transform2,
        patch_size=config.DATA.IMG_SIZE,
        normalize_image=255. if config.DATA.DATASET == "synth_3d"  or config.DATA.DATASET == "mixed_synth_3d" else 1.,
    )
    return ds
