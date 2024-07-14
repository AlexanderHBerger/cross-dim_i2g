import os
import numpy as np
import random
import imageio
import torch
import pyvista
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
from torchvision.transforms import Grayscale
from PIL import Image
from utils.utils import rotate_coordinates
from torchvision.transforms.functional import rotate

class Sat2GraphDataLoader(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, data, augment, use_grayscale=False, domain_classification=-1):
        """[summary]

        Args:
            data ([type]): [description]
            transform ([type]): [description]
        """
        self.data = data
        self.augment = augment

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.domain_classification = domain_classification

        self.use_grayscale = use_grayscale
        self.grayscale = Grayscale(num_output_channels=3)

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
        vtk_data = pyvista.read(data['vtp'])
        raw_seg_data = Image.open(data['seg'])

        seg_data = np.array(raw_seg_data)
        seg_data = np.array(seg_data)/np.max(seg_data)
        seg_data = torch.tensor(seg_data, dtype=torch.int).unsqueeze(0)

        image_data = Image.open(data['img'])

        if self.use_grayscale:
            image_data = np.array(self.grayscale(image_data))
            image_data = torch.tensor(
                image_data, dtype=torch.float).permute(2, 0, 1)
            image_data = image_data / 255.0
            image_data -= 0.5
        else:
            image_data = np.array(image_data)
            image_data = torch.tensor(
                image_data, dtype=torch.float).permute(2, 0, 1)
            image_data = image_data / 255.0
            image_data = tvf.normalize(image_data.clone().detach(), mean=self.mean, std=self.std)

        nodes = torch.tensor(np.float32(
            np.asarray(vtk_data.points)), dtype=torch.float)[:, :2]
        lines = torch.tensor(np.asarray(
            vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)

        if self.augment:
            angle = random.randint(0, 3) * 90
            image_data = rotate(image_data, angle)
            seg_data = rotate(seg_data, angle)
            nodes = rotate_coordinates(nodes, angle)

        return image_data, seg_data-0.5, nodes, lines[:, 1:], self.domain_classification


def build_road_network_data(config, mode='train', split=0.95, max_samples=0, use_grayscale=False, domain_classification=-1):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.
        domain_classification (number, optional): -1 if no domain, 0 for source domain, 1 for target domain.
    Returns:
        [type]: [description]
    """
    if mode == 'train':
        img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
        seg_folder = os.path.join(config.DATA.DATA_PATH, 'seg')
        vtk_folder = os.path.join(config.DATA.DATA_PATH, 'vtp')
        img_files = []
        vtk_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
            seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]
        ds = Sat2GraphDataLoader(
            data=data_dicts,
            augment=True,
            use_grayscale=use_grayscale
        )
        return ds
    elif mode == 'test':
        img_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'raw')
        seg_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'seg')
        vtk_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'vtp')
        img_files = []
        vtk_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
            seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]

        if max_samples > 0:
            data_dicts = data_dicts[:max_samples]
        ds = Sat2GraphDataLoader(
            data=data_dicts,
            use_grayscale=use_grayscale,
            augment=False
        )
        return ds
    elif mode == 'split':
        img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
        seg_folder = os.path.join(config.DATA.DATA_PATH, 'seg')
        vtk_folder = os.path.join(config.DATA.DATA_PATH, 'vtp')
        img_files = []
        vtk_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
            seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]
        random.seed(config.DATA.SEED)
        random.shuffle(data_dicts)
        train_split = int(split*len(data_dicts))
        train_files, val_files = data_dicts[:
                                            train_split], data_dicts[train_split:]

        if max_samples > 0:
            train_files = train_files[:max_samples]
            val_files = val_files[:round(max_samples * (1 - split))]
            
        train_ds = Sat2GraphDataLoader(
            data=train_files,
            use_grayscale=use_grayscale,
            domain_classification=domain_classification,
            augment=True
        )
        val_ds = Sat2GraphDataLoader(
            data=val_files,
            use_grayscale=use_grayscale,
            domain_classification=domain_classification,
            augment=False
        )
        return train_ds, val_ds, None
