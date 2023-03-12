import torch
import torch.nn as nn
import torchvision.models as models


class VGG(nn.Module):
    def __init__(self, device):
        super(VGG, self).__init__()
        vgg = models.vgg19(True)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.features[:16]
        self.vgg = self.vgg.to(device)

    def forward(self, x):
        return self.vgg(x)


class ContentLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = VGG(device)
        self.criterion = nn.MSELoss()

    def forward(self, fake, real):
        fake_f = self.vgg(fake)
        real_f = self.vgg(real)
        return self.criterion(fake_f, real_f)


def adversarial_loss(x):
    return torch.sum(-torch.log(x))


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg_l = ContentLoss(device)

    def forward(self, fake, real, disc):
        vgg = self.vgg_l(fake, real)
        adv = adversarial_loss(disc)
        return vgg + 1e-3*adv

