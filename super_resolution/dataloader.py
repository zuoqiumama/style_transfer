import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImgLoader(Dataset):
    def __init__(self, root_path='../super_resolution'):
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
        ])
        self.lr_list = sorted(glob.glob(root_path + '/LR/*.*'))
        self.hr_list = sorted(glob.glob(root_path + '/HR/*.*'))

    def __len__(self):
        return min(len(self.lr_list), len(self.hr_list))

    def __getitem__(self, index):
        lr_img = Image.open(self.lr_list[index])
        hr_img = Image.open(self.hr_list[index])
        return self.transformer(lr_img), self.transformer(hr_img)
