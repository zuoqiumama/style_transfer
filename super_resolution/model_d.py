import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_c, output_c, kernel_size, stride, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(output_c),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )
        self.blocks = nn.Sequential(
            ConvBlock(64, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=1),
            ConvBlock(128, 128, kernel_size=3, stride=2),
            ConvBlock(128, 256, kernel_size=3, stride=1),
            ConvBlock(256, 256, kernel_size=3, stride=2),
            ConvBlock(256, 512, kernel_size=3, stride=1),
            ConvBlock(512, 512, kernel_size=3, stride=2)
        )
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dense(self.blocks(self.conv1(x)))


if __name__ == '__main__':
    g = Discriminator()
    a = torch.rand([2, 3, 512, 512])
    print(g(a).shape)
