"""UNet used by the seismic segmentation experiments."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels, norm=None, groups=8):
    kind = (norm or os.environ.get("UNET_NORM", "batch")).lower()
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        groups = int(os.environ.get("UNET_NORM_GROUPS", groups)) if norm is None else groups
        while groups > 1 and channels % groups != 0:
            groups //= 2
        return nn.GroupNorm(groups, channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unknown normalization {kind!r} (expected batch|group|instance)")


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, norm=None, groups=8):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _norm(mid_channels, norm, groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels, norm, groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None, groups=8):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, norm=norm, groups=groups)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, norm=None, groups=8):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm, groups)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm, groups=groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, bilinear=False, norm=None, groups=8):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(in_channels, 64, norm=norm, groups=groups)
        self.down1 = Down(64, 128, norm=norm, groups=groups)
        self.down2 = Down(128, 256, norm=norm, groups=groups)
        self.down3 = Down(256, 512, norm=norm, groups=groups)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, norm=norm, groups=groups)
        self.up1 = Up(1024, 512 // factor, bilinear, norm, groups)
        self.up2 = Up(512, 256 // factor, bilinear, norm, groups)
        self.up3 = Up(256, 128 // factor, bilinear, norm, groups)
        self.up4 = Up(128, 64, bilinear, norm, groups)
        self.outc = OutConv(64, num_classes)
        self.reconstruct = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x), self.reconstruct(x)

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
