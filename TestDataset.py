import data
import numpy as np
import pandas as pd
import data

from SegDataset import SegDataset

from torch.utils.data import DataLoader

if __name__ == '__main__':
    df = pd.read_csv(data.preproc_train_csv)
    ds = SegDataset(df, data.train_dir, data.train_mask_dir, dataset_flag=SegDataset.TRAIN_SET)

    img, mask = ds[9]

    assert img.size() == (3, 256, 1600)
    print('Image size test passed')
    assert mask.size() == (4, 256, 1600)
    print('Mask size test passed')

    dataloader = DataLoader(ds, batch_size=2)

    for img, mask in dataloader:
        print('Data from data loader.')
        print('Image Batch : ', img.size())
        print('Mask Batch : ', mask.size())
        break
