import torch
import torch.nn as nn

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv(x)
        out = out + self.shortcut(x)
        return nn.ReLU()(out)

class ResNet3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.layer1 = ResidualBlock3D(1, 32)
        self.layer2 = ResidualBlock3D(32, 64)
        self.layer3 = ResidualBlock3D(64, 128)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x
