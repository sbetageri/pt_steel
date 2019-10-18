import data
import torch
import pandas as pd
import numpy as np

from models import UNet

from collections import defaultdict
from SegDataset import SegDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(model, train_loader, val_loader, epochs, optim, loss_fn, scheduler, writer):
    train_channel_iou = [0] * 4
    train_channel_dice = [0] * 4
    val_channel_iou = [0] * 4
    val_channel_dice = [0] * 4

    train_loss = 0
    val_loss = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for e in range(epochs):
        running_loss = 0
        running_channel_iou = [0] * 4
        running_channel_dice = [0] * 4

        model.train()
        for idx, (img, mask) in tqdm(enumerate(train_loader)):
            img = img.to(device)
            mask = mask.to(device)

            optim.zero_grad()
            out = model(img)
            loss = loss_fn(out, mask)
            loss.backward()
            optim.step()

            running_loss += loss.item()

        model.eval()
        running_loss = 0
        with torch.no_grad():
            for idx, (img, mask) in tqdm(enumerate(val_loader)):
                img = img.to(device)
                mask = mask.to(device)
                out = model(img)
                loss = loss_fn(out, mask)
                running_loss += loss.item()
        scheduler.step(running_loss)
    return model

def mask2rle(img):
    ## Taken from https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def predict(model, dataframe, dataset):
    df = dataframe.copy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for img, img_id in tqdm(dataset):
        img = img.view(1, *img.size())
        img = img.to(device)
        masks = model(img)
        for c in range(masks.size(0)):
            rle = mask2rle(masks[c].numpy())
            c_img_id = img_id + '.jpg_' + str(c + 1)
            df.loc[df['ImageId_ClassId'] == c_img_id, ['EncodedPixels']] = rle
    return df


def metric_calc(pred, mask):
    pred = torch.round(pred)

    pred = pred.byte()
    mask = mask.byte()

    channel_iou = [0] * 4
    channel_dice = [0] * 4
    for channel in range(pred.size(1)):
        iou_calc = 0
        channel_iou[channel] = 0


def iou_calc(pred, mask):
    ## Assuming Pred is already discrete 0/1. Byte tensors
    intersection = torch.sum(pred & mask)
    union = torch.sum(pred | mask)
    return intersection / union

def dice_calc(pred, mask):
    intersection = torch.sum(pred & mask)
    sum_pred = torch.sum(pred)
    sum_mask = torch.sum(mask)
    return (2 * intersection) / (sum_mask + sum_pred)

if __name__ == '__main__':
    writer = SummaryWriter('runs/steel_trial')

    df = pd.read_csv(data.preproc_train_csv)
    train_df, val_df = train_test_split(df, test_size=0.15)

    train_ds = SegDataset(train_df, data.train_dir, data.train_mask_dir, dataset_flag=SegDataset.TRAIN_SET)
    val_ds = SegDataset(val_df, data.train_dir, data.train_mask_dir, dataset_flag=SegDataset.TRAIN_SET)

    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)

    model = UNet.UNet()

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    model = train(model, train_loader, val_loader, 3, optimizer, loss_fn, scheduler, writer)

    torch.save(model.state_dict(), 'model_weights/small_unet.pt')

    test_df = pd.read_csv(data.test_csv)
    test_ds = SegDataset(test_df, data.test_dir, mask_dir=None, dataset_flag=SegDataset.TEST_SET)

    predictions = predict(model, test_df, test_ds)
    predictions.to_csv('predictions.csv', index=False)
