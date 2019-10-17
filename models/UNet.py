import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Encoder import Encoder
from models.Decoder import Decoder

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = Encoder(3, 64)
        self.enc2 = Encoder(64, 128)
        self.enc3 = Encoder(128, 256)
        self.enc4 = Encoder(256, 512)

        self.conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.t_conv = nn.ConvTranspose2d(1024, 512, 2)

        self.dec4 = Decoder(1024, 512)
        self.dec3 = Decoder(512, 256)
        self.dec2 = Decoder(256, 128)
        self.dec1 = Decoder(128, 64, is_final=True)

    def forward(self, x):
        down1, feat1 = self.enc1(x)
        print(down1.size())
        print(feat1.size())
        assert False

        down2, feat2 = self.enc2(down1)
        down3, feat3 = self.enc3(down2)
        down4, feat4 = self.enc4(down3)

        x = self.conv1(down4)
        x = self.conv2(x)
        up1 = self.t_conv(x)

        up2 = self.dec4(up1, feat4)
        up3 = self.dec3(up2, feat3)
        up4 = self.dec2(up3, feat2)
        out = self.dec1(up4, feat1)
        return out
