import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Encoder import Encoder
from models.Decoder import Decoder

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = Encoder(3, 128)
        self.enc2 = Encoder(128, 256)
        self.enc3 = Encoder(256, 512)
        self.enc4 = Encoder(512, 1024)

        self.conv1 = nn.Conv2d(1024, 2048, 3, padding=1)
        self.conv2 = nn.Conv2d(2048, 2048, 3, padding=1)
        self.t_conv = nn.ConvTranspose2d(2048, 1024, 2, stride=2)

        self.dec4 = Decoder(2048, 1024)
        self.dec3 = Decoder(1024, 512)
        self.dec2 = Decoder(512, 256)
        self.dec1 = Decoder(256, 128, is_final=True)

    def forward(self, x):
        # print('X : ', x.size())

        down1, feat1 = self.enc1(x)
        down2, feat2 = self.enc2(down1)
        down3, feat3 = self.enc3(down2)
        down4, feat4 = self.enc4(down3)
        # down5, feat5 = self.enc5(down4)

        # print('Down 5 : ', down5.size())
        # print('Feat 5 : ', feat5.size())

        x = self.conv1(down4)
        x = self.conv2(x)
        up1 = self.t_conv(x)

        # print('Up 1 : ', up1.size())
        up2 = self.dec4(up1, feat4)
        up3 = self.dec3(up2, feat3)
        up4 = self.dec2(up3, feat2)
        out = self.dec1(up4, feat1)
        return out
