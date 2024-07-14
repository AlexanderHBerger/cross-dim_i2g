import torch
from monai.data import Dataset
from medpy.io import load
from monai.transforms import Compose, Zoom, RandGaussianNoise, ScaleIntensity
import pyvista
import numpy as np
import random
import os
from utils.utils import rotate_image, rotate_coordinates

train_transform = Compose(
    [
        RandGaussianNoise(prob=0.2, std=0.015, mean=0),
        ScaleIntensity(minv=-0.5, maxv=0.5),
    ]
)
val_transform = Compose([])

class vessel_loader(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self,
                 data,
                 transform,
                 patch_size=(64,64,64),
                 normalize_image=1.,
                 normalize_nodes = 1.,
                 augment=False,
                 domain_classification=-1):
        """[summary]

        Args:
            data ([type]): [description]
            transform ([type]): [description]
        """
        self.data = data
        self.transform = transform
        self.patch_size = patch_size
        self.normalize_image = normalize_image
        self.normalize_nodes = normalize_nodes
        self.augment = augment
        self.domain_classification = domain_classification
    
    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        data = self.data[idx]
        image_data, _ = load(data['nifti'])
        image_data = torch.tensor(image_data, dtype=torch.float).unsqueeze(0)
        image_data = image_data / self.normalize_image - 0.5
        vtk_data = pyvista.read(data['vtp'])
        seg_data, _ = load(data['seg'])
        seg_data = torch.tensor(seg_data, dtype=torch.int).unsqueeze(0)-0.5

        coordinates = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float) / self.normalize_nodes
        lines = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:,1:]

        if self.augment:
            # Random Rotation
            alpha = random.randint(0, 3) * 90
            beta = random.randint(0, 3) * 90
            gamma = random.randint(0, 3) * 90
            image_data = rotate_image(image_data, alpha, beta, gamma)
            seg_data = rotate_image(seg_data, alpha, beta, gamma)
            coordinates = rotate_coordinates(coordinates, alpha, beta, gamma)

            # Random zoom
            zoom_factor = random.uniform(0.6, 1.)

            zoom = Zoom(zoom_factor, padding_mode="edge")
            image_data = zoom(image_data)
            seg_data = zoom(seg_data)
            coordinates *= zoom_factor
            coordinates += (1. - zoom_factor) / 2.

            # Gaussian Noise
            image_data = train_transform(image_data)

        return [image_data], [seg_data], [coordinates], [lines], [None], [self.domain_classification]


def build_vessel_data(config, mode='train', split=0.95, debug=False, max_samples=0, domain_classification=-1):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """
    path = config.DATA.TEST_DATA_PATH if mode=='test' else config.DATA.DATA_PATH
    nifti_folder = os.path.join(path, 'raw')
    seg_folder = os.path.join(path, 'seg')
    vtk_folder = os.path.join(path, 'vtp')
    nifti_files = []
    vtk_files = []
    seg_files = []

    for i,file_ in enumerate(os.listdir(nifti_folder)):
        file_ = file_[:-7]
        nifti_files.append(os.path.join(nifti_folder, file_+'.nii.gz'))
        vtk_files.append(os.path.join(vtk_folder, file_[:-4]+'graph.vtp'))
        seg_files.append(os.path.join(seg_folder, file_[:-4]+'seg.nii.gz'))
        # if i>45000:
        #     break

    data_dicts = [
        {"nifti": nifti_file, "vtp": vtk_file, "seg": seg_file} for nifti_file, vtk_file, seg_file in zip(nifti_files, vtk_files, seg_files)
        ]
    if mode=='train':
        ds = vessel_loader(
            data=data_dicts,
            transform=train_transform,
            patch_size=config.DATA.IMG_SIZE,
            normalize_image=255. if config.DATA.DATASET == "synth_3d"  or config.DATA.DATASET == "mixed_synth_3d" or config.DATA.DATASET == "mixed_synth_3d_octa" else 1.,
            normalize_nodes=50. if config.DATA.DATASET == "real_vessels" or config.DATA.DATASET == "mixed_real_vessels" or config.DATA.DATASET == "mixed_real_vessels_octa" else 1.,
            augment=False 
        )
        return ds
    elif mode=='test':
        if max_samples > 0:
            data_dicts = data_dicts[:max_samples]
        ds = vessel_loader(
            data=data_dicts,
            transform=None,
            patch_size=config.DATA.IMG_SIZE,
            normalize_image=255. if config.DATA.DATASET == "synth_3d" or config.DATA.DATASET == "mixed_synth_3d" or config.DATA.DATASET == "mixed_synth_3d_octa" else 1.,
            normalize_nodes=50. if config.DATA.DATASET == "real_vessels" or config.DATA.DATASET == "mixed_real_vessels" or config.DATA.DATASET == "mixed_real_vessels_octa"  else 1.,
            augment=False,
        )
        return ds
    elif mode=='split':
        random.seed(config.DATA.SEED)
        random.shuffle(data_dicts)
        train_split = int(split*len(data_dicts))
        train_files, val_files = data_dicts[:train_split], data_dicts[train_split:]
        if debug:
            train_files = train_files[:128]
            val_files = val_files[:32]
        elif max_samples > 0:
            train_files = train_files[:max_samples]
            val_files = val_files[:round(max_samples*(1-split))]
        train_ds = vessel_loader(
            data=train_files,
            transform=train_transform,
            patch_size=config.DATA.IMG_SIZE,
            normalize_image=255. if config.DATA.DATASET == "synth_3d" or config.DATA.DATASET == "mixed_synth_3d" or config.DATA.DATASET == "mixed_synth_3d_octa" else 1.,
            normalize_nodes=50. if config.DATA.DATASET == "real_vessels" or config.DATA.DATASET == "mixed_real_vessels"or config.DATA.DATASET == "mixed_real_vessels_octa"  else 1.,
            augment=False,
            domain_classification=domain_classification
        )
        val_ds = vessel_loader(
            data=val_files,
            transform=val_transform,
            patch_size=config.DATA.IMG_SIZE,
            normalize_image=255. if config.DATA.DATASET == "synth_3d"  or config.DATA.DATASET == "mixed_synth_3d" or config.DATA.DATASET == "mixed_synth_3d_octa" else 1.,
            normalize_nodes=50. if config.DATA.DATASET == "real_vessels" or config.DATA.DATASET == "mixed_real_vessels" or config.DATA.DATASET == "mixed_real_vessels_octa" else 1.,
            augment=False,
            domain_classification=domain_classification
        )
        return train_ds, val_ds, None
