import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        down = self.max_pool(x)
        return down, x




