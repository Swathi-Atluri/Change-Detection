import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """A helper module that performs two convolutional layers with ReLU and BatchNorm"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class remoteSensor(nn.Module):
    """
    U-Net style model for change detection.
    Input: two images (A and B) each with 3 channels (RGB), concatenated along channel dim.
    Output: binary change mask.
    """
    def __init__(self):
        super(remoteSensor, self).__init__()

        # Encoder
        self.conv1 = DoubleConv(6, 64)  # 3 channels from A + 3 from B
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, imgA, imgB):
        # Concatenate A and B along channel axis => shape: (B, 6, H, W)
        x = torch.cat([imgA, imgB], dim=1)

        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        bn = self.bottleneck(p4)

        # Decoder + Skip Connections
        up1 = self.up1(bn)
        merge1 = torch.cat([up1, c4], dim=1)
        d1 = self.dec1(merge1)

        up2 = self.up2(d1)
        merge2 = torch.cat([up2, c3], dim=1)
        d2 = self.dec2(merge2)

        up3 = self.up3(d2)
        merge3 = torch.cat([up3, c2], dim=1)
        d3 = self.dec3(merge3)

        up4 = self.up4(d3)
        merge4 = torch.cat([up4, c1], dim=1)
        d4 = self.dec4(merge4)

        out = self.final_conv(d4)
        return out  # shape: [B, 1, H, W]
