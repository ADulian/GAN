import os
import cv2
import sys
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets,transforms

# --------------------------------------------------------------------------------
class Data_Manager:
    def __init__(self, batch_size=64, img_size=32):
        self.batch_size = batch_size
        self.img_size = img_size
        self.transform = transforms.Compose([transforms.Resize(self.img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5),(0.5))])

        self.train_set = datasets.MNIST('./data', train=True, download=True, transform=self.transform)
        self.test_set = datasets.MNIST('./data', train=False, download=True, transform=self.transform)

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size)

# --------------------------------------------------------------------------------
class GAN(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self, data_manager, device, img_shape, epochs=1, lr=1e-3, gen_z_dim=100):
        super(GAN, self).__init__()
        self.results_root = datetime.datetime.now().strftime("(%H:%M:%S)_(%d_%m_%Y)")

        self.data_manager = data_manager
        self.device = device

        self.img_shape = img_shape

        self.epochs = epochs
        self.lr = lr
        self.gen_z_dim = gen_z_dim

        # Fixed Data
        self.x_fixed = torch.randn(5, gen_z_dim).to(self.device)

        # Generator and Discriminator
        self.gen = Generator(gen_z_dim, img_shape)
        self.disc = Discriminator(img_shape)

        # Optims and Criterion
        self.gen_optim = optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999))
        self.disc_optim = optim.Adam(self.disc.parameters(), lr=lr, betas=(0.5, 0.999))

        self.criterion = nn.BCELoss()
        # self.criterion = nn.MSELoss()

        # Send to GPU/CPU
        self.to(self.device)


    # --------------------------------------------------------------------------------
    def forward(self, x):
        pass

    # --------------------------------------------------------------------------------
    def fit(self):
        print("--- Training...")
        best_test_loss = float('inf')

        for i in range(self.epochs):
            disc_train_loss, gen_train_loss = self.epoch_train()
            # val_loss = self.epoch_val()

            print("\nEpoch {}/{}".format(i+1, self.epochs))
            print("Train - Disc: {:.5f} Gen: {:.5f}".format(disc_train_loss, gen_train_loss))
            print("Generating and Saving Fixed Data")
            self.gen_fixed(i)

    # --------------------------------------------------------------------------------
    def epoch_train(self):
        self.train()

        disc_train_loss = []
        gen_train_loss = []

        for i, data in enumerate(self.data_manager.train_loader):
            x, _ = data
            batch_size = x.shape[0]

            x = x.view(batch_size, -1)
            x = x.to(self.device)

            true_labels = torch.ones(batch_size,1).to(self.device)
            fake_labels = torch.zeros(batch_size,1).to(self.device)

            # Generate Fake Data
            gen_z = torch.randn(batch_size, self.gen_z_dim).to(self.device)
            gen_out = self.gen(gen_z)

            # --------------------
            # Discriminator
            # --------------------
            self.disc_optim.zero_grad()

            disc_true_out = self.disc(x)
            disc_fake_out = self.disc(gen_out.detach())

            disc_true_loss = self.criterion(disc_true_out, true_labels)
            disc_fake_loss = self.criterion(disc_fake_out, fake_labels)

            disc_loss = (disc_true_loss + disc_fake_loss) / 2
            disc_loss.backward()

            self.disc_optim.step()

            disc_train_loss.append(disc_loss.item())

            # --------------------
            # Generator
            # --------------------
            self.gen_optim.zero_grad()

            disc_gen_fake_out = self.disc(gen_out)

            gen_loss = self.criterion(disc_gen_fake_out, true_labels)
            gen_loss.backward()

            self.gen_optim.step()

            gen_train_loss.append(gen_loss.item())

        return np.mean(disc_train_loss), np.mean(gen_train_loss)

    # --------------------------------------------------------------------------------
    @torch.no_grad()
    def epoch_val(self):
        self.eval()

    # --------------------------------------------------------------------------------
    @torch.no_grad()
    def gen_fixed(self, epoch):
        if not os.path.exists(self.results_root):
            os.mkdir(self.results_root)

        gen_out = self.gen(self.x_fixed)
        gen_out = gen_out.cpu().numpy()
        gen_out = np.transpose(gen_out, (0,2,3,1))

        for i, sample in enumerate(gen_out):
            sample *= 255.0
            cv2.imwrite("{}/S_{}_E_{}.png".format(self.results_root,i, epoch), sample)

# --------------------------------------------------------------------------------
class Generator(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.gen = nn.Sequential(*self._block(latent_dim, 128, normalize=False),
                                 *self._block(128, 256),
                                 *self._block(256, 512),
                                 *self._block(512, 1024),
                                 nn.Linear(1024, int(np.prod(img_shape))),
                                 nn.Tanh())

    # --------------------------------------------------------------------------------
    def forward(self, z):
        gen_out = self.gen(z)
        gen_out = gen_out.view(gen_out.shape[0], *self.img_shape)

        return gen_out

    # --------------------------------------------------------------------------------
    def _block(self, in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]

        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return layers

# --------------------------------------------------------------------------------
class Discriminator(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.dec = nn.Sequential(nn.Linear(int(np.prod(img_shape)), 512),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(512, 512),
                                 nn.Dropout(0.4),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(512, 512),
                                 nn.Dropout(0.4),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(512, 1),
                                 nn.Sigmoid())

    # --------------------------------------------------------------------------------
    def forward(self, x):

        dec_out = self.dec(x.view(x.shape[0], -1))

        return dec_out
