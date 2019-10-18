import pandas as pd
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class SegDataset(Dataset):
    TRAIN_SET = 9
    TEST_SET = 27
    def __init__(self, dataframe, img_dir, mask_dir, dataset_flag=TRAIN_SET):
        self.df = dataframe
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.flag = dataset_flag
        self.img_transforms = transforms.Compose([
            transforms.Resize((64, 400)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize((64, 400)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.flag == SegDataset.TEST_SET:
            img_id = self.df.iloc[idx]['ImageId_ClassId']
            img_id = ''.join(img_id.split('.')[:-1])
            img_path = self.img_dir + img_id + '.jpg'
            img = Image.open(img_path)
            tnsr_img = self.img_transforms(img)
            return tnsr_img, img_id


        img_id = self.df.iloc[idx]['img_id']
        img_id = ''.join(img_id.split('.')[:-1])

        img_path = self.img_dir + img_id + '.jpg'
        mask_path = self.mask_dir + img_id + '.npz'

        img = Image.open(img_path)

        # a because the mask was saved as np.savez_compressed(a=mask)
        mask = np.load(mask_path)['a']

        tnsr_img = self.img_transforms(img)
        tnsr_mask = self.mask_transforms(mask)
        return tnsr_img, tnsr_mask


