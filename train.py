import os
import random
import torch
import itertools
import time
from datasets import ImageDataset
from model import Generator
from model import Discriminator
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


n_epochs = 200
batch_size = 1
lr = 0.0002
decay_epoch = 100
size = 256
input_nc = 3
output_nc = 3


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


transforms_ = [transforms.Resize(int(size * 1.12)),
               transforms.RandomCrop(size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
datasets_dicts = ['monet2photo', 'ukiyoe2photo', 'cezanne2photo', 'vangogh2photo']

Tensor = torch.cuda.FloatTensor
input_A = Tensor(batch_size, input_nc, size, size)
input_B = Tensor(batch_size, output_nc, size, size)
target_real = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


for b in datasets_dicts:
    epoch = 0
    dataloader = DataLoader(
        ImageDataset(root='./datasets', datas=b, transforms_=transforms_, unaligned=True),
        batch_size=batch_size, shuffle=True)
    generator_A2B = Generator(input_nc, output_nc).cuda().apply(weights_init_normal)
    generator_B2A = Generator(output_nc, input_nc).cuda().apply(weights_init_normal)
    discriminator_A = Discriminator(input_nc).cuda().apply(weights_init_normal)
    discriminator_B = Discriminator(output_nc).cuda().apply(weights_init_normal)

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(itertools.chain(generator_A2B.parameters(), generator_B2A.parameters()),
                                   lr=lr,
                                   betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(discriminator_A.parameters(),
                                     lr=lr,
                                     betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(discriminator_B.parameters(),
                                     lr=lr,
                                     betas=(0.5, 0.999))
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                         lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                         lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    for epoch in range(epoch, n_epochs):
        time_start = time.time()
        if os.path.exists('output/' + b + '/netG_A2B.pth'):
            generator_A2B = torch.load('output/' + b + '/netG_A2B.pth')
            generator_B2A = torch.load('output/' + b + '/netG_B2A.pth')
            discriminator_A = torch.load('output/' + b + '/netD_A.pth')
            discriminator_B = torch.load('output/' + b + '/netD_B.pth')
        for i, batch in enumerate(dataloader):
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            # generator loss
            # identity loss
            optimizer_G.zero_grad()
            samp_A = generator_B2A(real_A)
            loss_identity_A = criterion_identity(samp_A, real_A) * 5.0
            samp_B = generator_A2B(real_B)
            loss_identity_B = criterion_identity(samp_B, real_B) * 5.0

            # GAN loss
            fake_B = generator_A2B(real_A)
            pred_fake = discriminator_B(fake_B)
            loss_g_B = criterion_GAN(pred_fake, target_real)

            fake_A = generator_B2A(real_B)
            pred_fake = discriminator_A(fake_A)
            loss_g_A = criterion_GAN(pred_fake, target_real)

            # cycle loss
            re_A = generator_B2A(fake_B)
            loss_c_A = criterion_cycle(re_A, real_A) * 10.0

            re_B = generator_A2B(fake_A)
            loss_c_B = criterion_cycle(re_B, real_B) * 10.0

            # total loss
            loss_G = loss_identity_A + loss_identity_B + loss_g_B + loss_g_A + loss_c_A + loss_c_B
            loss_G.backward()
            optimizer_G.step()

            # discriminator loss
            # A
            optimizer_D_A.zero_grad()
            pred_real = discriminator_A(real_A)
            loss_d_pra = criterion_GAN(pred_real, target_real)

            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = discriminator_A(fake_A)
            loss_d_pfa = criterion_GAN(pred_fake, target_fake)

            total_l_a = (loss_d_pfa + loss_d_pra) * 0.5
            total_l_a.backward()

            optimizer_D_A.step()

            # B
            optimizer_D_B.zero_grad()
            pred_real = discriminator_B(real_B)
            loss_d_prb = criterion_GAN(pred_real, target_real)

            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = discriminator_B(fake_B)
            loss_d_pfb = criterion_GAN(pred_fake, target_fake)

            total_l_b = (loss_d_pfb + loss_d_prb) * 0.5
            total_l_b.backward()

            optimizer_D_B.step()
            print(f"[epoch]:{epoch}, progress [{(i + 1) * 100 /len(dataloader)}%], generator_total_loss: {loss_G}, "
                  f"discriminator_A_total_loss: {total_l_a}, discriminator_B_total_loss: {total_l_b}")
        time_end = time.time()  # 记录结束时间
        time_sum = time_end - time_start
        print(f"one epoch time = {time_sum}")
        print(f"last time = {(n_epochs - epoch - 1) * time_sum}")
        print("================================================")
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        torch.save(generator_A2B, 'output/' + b + '/netG_A2B.pth')
        torch.save(generator_B2A, 'output/' + b + '/netG_B2A.pth')
        torch.save(discriminator_A, 'output/' + b + '/netD_A.pth')
        torch.save(discriminator_B, 'output/' + b + '/netD_B.pth')
