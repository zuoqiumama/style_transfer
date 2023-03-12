import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, input_c=64, output_c=64, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, bias=False, padding=padding),
            nn.BatchNorm2d(output_c),
            nn.PReLU(),
            nn.Conv2d(output_c, output_c, kernel_size=kernel_size, stride=stride, bias=False, padding=padding),
            nn.BatchNorm2d(output_c)
        )

    def forward(self, x0):
        x1 = self.net(x0)
        return x1 + x0


class Generator(nn.Module):
    def __init__(self, scale=2, n_residuals=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        self.residuals = nn.Sequential()
        for ii in range(n_residuals):
            self.residuals.append(ResidualBlock())
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.residuals(x1)
        x = self.conv2(x)
        x = self.conv3(x + x1)
        x = self.conv4(x)
        return x


if __name__ == '__main__':
    g = Generator()
    a = torch.rand([1, 3, 64, 64])
    print(g(a).shape)


