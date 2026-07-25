import torch
from torch import nn


class MFM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels, self.out_channels * 2, kernel_size, stride, padding
        )

    def forward(self, x):
        x = self.conv(x)

        first_half = x[:, : self.out_channels]
        second_half = x[:, self.out_channels :]

        return torch.max(first_half, second_half)
