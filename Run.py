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

def rebatch(tnsr):
    b, nsplit, c, h, w = tnsr.size()
    tnsr = tnsr.view(-1, c, h, w)
    return tnsr

def train(model, train_loader, val_loader, epochs, optim, loss_fn, scheduler, writer, warmup=False):
    train_iou = 0
    train_dice = 0

    val_iou = 0
    val_dice = 0

    train_loss = 0
    val_loss = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for e in range(epochs):
        running_loss = 0
        running_iou = 0
        running_dice = 0

        model.train()
        count = 1
        for img, mask in tqdm(train_loader):
            img = rebatch(img)
            mask = rebatch(mask)
            count += 1
            img = img.to(device)
            mask = mask.to(device)

            optim.zero_grad()
            out = model(img)
            loss = loss_fn(out, mask)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            train_loss += loss.item()

            iou, dice = metric_calc(out.cpu().detach(), mask.cpu().detach())

            running_iou += iou
            running_dice += dice

            train_iou += iou
            train_dice += dice

            if count % 100 == 0:

                div_factor = (100 * train_loader.batch_size)
                plot_metrics(writer, running_iou, running_dice, running_loss, div_factor,
                             iou_tag='r_iou', dice_tag='r_dice', loss_tag='r_loss',
                             train_val_tag='train')

                running_iou = 0
                running_dice = 0

        if warmup:
            continue

        model.eval()
        running_loss = 0

        with torch.no_grad():
            count = 0
            for img, mask in tqdm(val_loader):
                img = rebatch(img)
                mask = rebatch(img)
                count += 1
                img = img.to(device)
                mask = mask.to(device)
                out = model(img)
                loss = loss_fn(out, mask)
                running_loss += loss.item()
                val_loss += loss.item()

                iou, dice = metric_calc(out.detach().cpu(), mask.detach().cpu())

                running_dice += dice
                running_iou += iou

                val_iou += iou
                val_dice += iou

                if count % 100 == 0:
                    div_factor = 100 * val_loader.batch_size

                    plot_metrics(writer, running_iou, running_dice, running_loss, div_factor,
                                 iou_tag='r_iou', dice_tag='r_dice', loss_tag='r_loss',
                                 train_val_tag='val')

                    running_iou = torch.zeros((1, 4))
                    running_dice = torch.zeros((1, 4))

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
    print('Pixel shape : ', pixels.shape)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def predict(model, dataframe, dataset):
    df = dataframe.copy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    with torch.no_grad():
        for img, img_id in tqdm(dataset):
            img = img.view(1, *img.size())
            img = rebatch(img)
            img = img.to(device)
            masks = model(img)

            masks = torch.chunk(masks, 4, dim=0)
            masks = [mask.view(mask.size()[1:]) for mask in masks]
            masks = torch.cat(masks, dim=2)
            masks = masks
            masks = torch.round(masks)
            masks = masks.byte()
            # masks = masks.view((4, 1600, 256))
            # mrle = mask2rle(masks)
            # masks = masks.view(masks.size()[1:])
            # print(masks.size())
            for c in range(masks.size(0)):
                print('Channel size : ', masks[c].size())
                rle = mask2rle(masks[c].numpy())
                c_img_id = img_id + '.jpg_' + str(c + 1)
                df.loc[df['ImageId_ClassId'] == c_img_id, ['EncodedPixels']] = rle
    return df

def plot_metrics(writer, iou, dice, loss, div, iou_tag, dice_tag, loss_tag, train_val_tag):
    iou = torch.div(iou, div)
    dice = torch.div(dice, div)
    loss = torch.div(loss, div)

    writer.add_scalar(iou_tag + '/' + train_val_tag, torch.sum(iou) / 4)
    writer.add_scalar(dice_tag + '/' + train_val_tag, torch.sum(dice) / 4)
    writer.add_scalar(loss_tag + '/' + train_val_tag, loss)

def metric_calc(pred, mask):
    pred = torch.round(pred)

    pred = pred.byte()
    mask = mask.byte()

    pred = pred.view(1, -1)
    mask = mask.view(1, -1)

    intersection = torch.sum(pred & mask)
    union = torch.sum(pred | mask)
    denom = torch.sum(pred) + torch.sum(mask)

    if union == 0:
        iou = 0
    else:
        iou = intersection / union

    if denom == 0:
        dice = 0
    else:
        dice = (2 * intersection) / denom

    return iou, dice

def dice_calc(pred, mask):
    intersection = torch.sum(pred & mask)
    sum_pred = torch.sum(pred)
    sum_mask = torch.sum(mask)
    if (sum_mask + sum_pred) == 0:
        return 0
    return (2 * intersection) / (sum_mask + sum_pred)

if __name__ == '__main__':
    writer = SummaryWriter('runs/steel_trial')

    df = pd.read_csv(data.preproc_train_csv)
    train_df, val_df = train_test_split(df, test_size=0.15)

    warmup_ds = SegDataset(train_df, data.train_dir, data.train_mask_dir, dataset_flag=SegDataset.TRAIN_SET)
    warmup_loader = DataLoader(warmup_ds, batch_size=2)

    train_ds = SegDataset(train_df, data.train_dir, data.train_mask_dir, dataset_flag=SegDataset.TRAIN_SET)
    val_ds = SegDataset(val_df, data.train_dir, data.train_mask_dir, dataset_flag=SegDataset.TRAIN_SET)

    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)

    model = UNet.UNet()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model = train(model, warmup_loader,
                  val_loader=None, epochs=3,
                  optim=optimizer, loss_fn=loss_fn,
                  scheduler=None, writer=writer,
                  warmup=True)

    torch.save(model.state_dict(), 'model_weights/small_unet.pt')

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    model = train(model, train_loader,
                  val_loader=val_loader, epochs=20,
                  optim=optimizer, loss_fn=loss_fn,
                  scheduler=scheduler, writer=writer)

    model = train(model, train_loader, val_loader, 3, optimizer, loss_fn, scheduler, writer)

    torch.save(model.state_dict(), 'model_weights/unet.pt')

    test_df = pd.read_csv(data.test_csv)
    test_ds = SegDataset(test_df, data.test_dir, mask_dir=None, dataset_flag=SegDataset.TEST_SET)

    predictions = predict(model, test_df, test_ds)
    predictions.to_csv('predictions.csv', index=False)
