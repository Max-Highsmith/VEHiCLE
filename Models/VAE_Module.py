#The VAE component of VEHiCLE is built off of a vae tutorial page
#with adjustments to architecture
#the stencil code comes from
#https://github.com/AntixK/PyTorch-VAE/blob/master/models/betatc_vae.py

import torch
import pdb
import pytorch_lightning as pl
import math
import torch
from torch import nn
from torch.nn import functional as F
from Data.GM12878_DataModule import GM12878Module
from pytorch_lightning import Trainer

class VAE_Model(pl.LightningModule):

    def __init__(self,
            batch_size=-9,
            kld_weight=0.0,
            kld_weight_inc=0.000,
            lr=0.0001,
            gamma=0.99,
            latent_dim=100,
            pre_latent=4608,
            condensed_latent=3,
            ):
        super(VAE_Model, self).__init__()
        torch.manual_seed(0)
        self.batch_size       = batch_size
        self.epoch_num        = 0
        self.kld_weight_inc   = kld_weight_inc
        self.kld_weight       = kld_weight
        self.lr               = lr
        self.gamma            = gamma
        self.latent_dim       = latent_dim
        self.PRE_LATENT       = pre_latent
        self.CONDENSED_LATENT = condensed_latent
        hidden_dims      = [32, 64, 128, 256, 256, 512, 512]
        modules               = []

        self.save_hyperparameters()

        in_channels = 1 
        #Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                                kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        self.fc_mu    = nn.Linear(self.PRE_LATENT, self.latent_dim)
        self.fc_var   = nn.Linear(self.PRE_LATENT, self.latent_dim)

        #Build Decoder
        modules = []
        self.decoder_input = nn.Linear(self.latent_dim, self.PRE_LATENT)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i +1],
                                       kernel_size=3,
                                       stride =2,
                                       padding=1),
                                       #output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU())
                )
        
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[-1],
                                        hidden_dims[-1],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1),
                                        #output_padding=1),
                        nn.BatchNorm2d(hidden_dims[-1]),
                        nn.LeakyReLU(),
                        nn.Conv2d(hidden_dims[-1], out_channels=1,
                                kernel_size=3, padding=1),
                        #nn.Tanh())
                        nn.Sigmoid())

    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        rez  = self.decoder_input(z)
        rez  = rez.view(-1,
                512,
                self.CONDENSED_LATENT,
                self.CONDENSED_LATENT)
        result  = list(self.decoder.children())[0](rez)
        result  = list(self.decoder.children())[1](result)
        result  = list(self.decoder.children())[2](result)
        result  = list(self.decoder.children())[3](result)
        result  = list(self.decoder.children())[4](result)
        result  = list(self.decoder.children())[5](result)
        #result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_z(self, x):
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return  [recon, x, mu, log_var]

    def loss_function(self, *args):
        recons  = args[0]
        x       = args[1]
        mu      = args[2]
        log_var = args[3]
        
        kld_weight = self.kld_weight 

        recon_loss = F.mse_loss(recons, x)
        kld_loss   = torch.mean(-0.5 * torch.sum(1 + log_var - mu **2 - log_var.exp(), dim = 1), dim = 0)
        loss = recon_loss + (kld_weight*kld_loss)
        self.log('kld_weight', kld_weight)
        self.log('train_loss', loss)
        self.log('recon_loss', recon_loss)
        self.log('kld_loss', kld_loss)
        return loss, recon_loss, kld_loss

    def training_step(self, batch, batch_idx):
        data, target, _               = batch
        results                    = self.forward(target)
        loss, recon_loss, kld_loss = self.loss_function(*results) 
        return loss

    def training_epoch_end(self, training_step_outputs):
        print(self.epoch_num)
        self.epoch_num = self.epoch_num+1
        if self.epoch_num > 0:
           self.kld_weight = self.kld_weight+self.kld_weight_inc

    def validation_step(self, batch, batch_idx):
        data, target, _            = batch
        results                    = self.forward(target)
        loss, recon_loss, kld_loss = self.loss_function(*results) 
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                    lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                gamma=self.gamma)
        return [optimizer], [scheduler]

