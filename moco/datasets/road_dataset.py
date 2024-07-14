import os
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
from PIL import Image
from moco.loader import GaussianBlur, Solarize
import torchvision.transforms as transforms


class MoCo_Road_Dataset(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, data, img_size):
        """[summary]

        Args:
            data ([type]): [description]
            transform ([type]): [description]
        """
        self.data = data
        self.transform1 = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.ToTensor(),
        ])

        self.transform2 = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
            transforms.ToTensor(),
        ])

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.data)

    def __getitem__(self, idx):
        image_data = Image.open(self.data[idx])

        image_data1 = self.transform1(image_data)
        image_data2 = self.transform2(image_data)

        image_data1 = tvf.normalize(image_data1, mean=self.mean, std=self.std)
        image_data2 = tvf.normalize(image_data2, mean=self.mean, std=self.std)

        return image_data1, image_data2


def build_moco_road_dataset(config, max_samples=0):
    img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
    img_files = []

    for file_ in os.listdir(img_folder):
        file_ = file_[:-8]
        img_files.append(os.path.join(img_folder, file_+'data.png'))

    if max_samples > 0:
        img_files = img_files[:max_samples]
        
    train_ds = MoCo_Road_Dataset(
        data=img_files,
        img_size=config.DATA.IMG_SIZE
    )
    return train_ds
