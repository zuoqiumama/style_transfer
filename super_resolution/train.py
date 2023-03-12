import time
import os
import torch
from torch.utils.data import DataLoader
import criterion
import dataloader
import model_d
import model_g

net_param_save_path = 'output/dicts.pt'
batch_size = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = model_g.Generator()
discriminator = model_d.Discriminator()
generator.train()
discriminator.train()
generator.to(device)
discriminator.to(device)
img_loader = DataLoader(dataloader.ImgLoader(),
                        batch_size=batch_size, shuffle=True, drop_last=True)
criterion_g = criterion.PerceptualLoss(device)
criterion_d = torch.nn.BCELoss()
epoch = 0
n_epochs = 100
lr = 1e-3
if not os.path.exists(net_param_save_path):
    print("No params, start training...")
else:
    param_dict = torch.load(net_param_save_path)
    epoch = param_dict["epoch"]
    lr = param_dict["lr"]
    discriminator.load_state_dict(param_dict["discriminator"])
    generator.load_state_dict(param_dict["generator"])
    print("Loaded params from {}\n[Epoch]: {}   [lr]: {} ".format(net_param_save_path, epoch, lr))
optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr * 0.1)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
real_label = torch.ones([batch_size, 1, 1, 1]).to(device)
fake_label = torch.zeros([batch_size, 1, 1, 1]).to(device)

train_loss_g = 0.
train_loss_d = 0.
train_loss_all_d = 0.
train_loss_all_g = 0.
for e in range(epoch, n_epochs):
    start = time.time()
    for i, (lr_img, hr_img) in enumerate(img_loader):
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        # generator train
        optimizer_g.zero_grad()
        fake_img = generator(lr_img)
        loss_g = criterion_g(fake_img, hr_img, discriminator(fake_img))
        loss_g.backward()
        optimizer_g.step()

        # g loss calculate
        train_loss_g += loss_g.item()
        train_loss_all_g += loss_g.item()

        # discriminator train
        if i % 2 == 0:
            optimizer_d.zero_grad()
            real_res = discriminator(hr_img)
            fake_res = discriminator(fake_img.detach())
            loss_d = criterion_d(real_res, real_label) + criterion_d(fake_res, fake_label)
            loss_d.backward()
            optimizer_d.step()

            # d loss calculate
            train_loss_d += loss_d.item()
            train_loss_all_d += loss_d.item()

        if (i + 1) % 20 == 0:
            end = time.time()
            print("[Epoch]: {}[Progress: {:.1f}%]time:{:.2f} dnet_loss:{:.5f} gnet_loss:{:.5f} ".format(
                e, (i + 1) * 100 / len(img_loader), end - start,
                train_loss_d / 200, train_loss_g / 200
            ))
            train_loss_g = 0
            train_loss_d = 0
    if os.path.exists(net_param_save_path):
        os.remove(net_param_save_path)
    param_dict = {
        'epoch': e,
        'lr': lr,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict()
    }
    torch.save(param_dict, net_param_save_path)

