import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, final_dim=4, is_final=False):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3)

        self.t_conv = nn.ConvTranspose2d(out_dim, out_dim/2, 2)
        self.is_final = is_final
        if self.is_final:
            self.final_conv = nn.Conv2d(out_dim, final_dim, 1)

    def forward(self, up_sampled, copied):
        x = torch.cat((up_sampled, copied))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        if self.is_final:
            x = self.final_conv(x)
            return x
        else:
            return self.t_conv(x)