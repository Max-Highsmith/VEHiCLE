import torch.nn.functional as F
import torch.nn as nn
import Models.VehicleGAN as vgan 
import torch
import pdb
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from Utils.loss import vae_loss as vl
from Utils.loss import insulation as ins

class GAN_Model(pl.LightningModule):
    def __init__(self):
        super(GAN_Model, self).__init__()
        self.mse_lambda     = 1
        self.tad_lambda     = 1
        self.vae_lambda     = 1e-3 
        self.gan_lambda     = 2.5e-3
        self.G_lr           = 1e-5
        self.D_lr           = 1e-5
        self.beta_1         = 0.9  
        self.beta_2         = 0.99
        self.num_res_blocks = 15
        self.generator      = vgan.Generator(num_res_blocks=self.num_res_blocks)
        self.discriminator  = vgan.Discriminator() 
        self.generator.init_params()
        self.discriminator.init_params()
        self.bce            = nn.BCEWithLogitsLoss()
        self.mse  = nn.L1Loss()
        self.vae_yaml        = "Trained_Models/vehicle_vae_hparams.yaml"
        self.vae_weight      = "Trained_Models/vehicle_vae.ckpt"
        self.vae             = vl.VaeLoss(self.vae_yaml, self.vae_weight)
        self.tad             = ins.InsulationLoss()
    
    def forward(self, x):
        fake = self.generator(x)
        return fake

    def tad_loss(self, target, output):
        return self.tad(target, output)

    def vae_loss(self, target, output):
        return self.vae(target, output)

    def adversarial_loss(self, target, output):
        return self.bce(target, output)

    def meanSquaredError_loss(self, target, output):
        return self.mse(target, output)

    def training_step(self, batch, batch_idx, optimizer_idx):
        data, full_target, info = batch
        target = full_target[:,:,6:-6,6:-6]

        #Generator
        if optimizer_idx == 0:
            self.generator.zero_grad()
            output      = self.generator(data)
            MSE_loss    = self.meanSquaredError_loss(output, target)
            VAE_loss    = self.vae_loss(output, target)
            TAD_loss    = self.tad_loss(output, target)
            pred_fake   = self.discriminator(output)
            labels_real = torch.ones_like(pred_fake, requires_grad=False)
            GAN_loss    = self.adversarial_loss(pred_fake, labels_real)
            
            total_loss_G = (self.mse_lambda*TAD_loss)+(self.vae_lambda*VAE_loss)+(self.mse_lambda * MSE_loss)+(self.gan_lambda *GAN_loss)
            self.log("total_loss_G", total_loss_G)
            return total_loss_G
        
        #Discriminator
        if optimizer_idx == 1:
            self.discriminator.zero_grad()
            #train on real data
            pred_real       = self.discriminator(target)
            labels_real     = torch.ones_like(pred_real, requires_grad=False)
            pred_labels_real = (pred_real>0.5).float().detach()
            acc_real        = (pred_labels_real == labels_real).float().sum()/labels_real.shape[0]
            loss_real       = self.adversarial_loss(pred_real, labels_real)
            
            #train on fake data
            output           = self.generator(data)
            pred_fake        = self.discriminator(output.detach())
            labels_fake      = torch.zeros_like(pred_fake, requires_grad=False)
            pred_labels_fake = (pred_fake > 0.5).float()
            acc_fake         = (pred_labels_fake == labels_fake).float().sum()/labels_fake.shape[0]
            loss_fake        = self.adversarial_loss(pred_fake, labels_fake)

            total_loss_D = loss_real + loss_fake
            self.log("total_loss_D",total_loss_D)
            return total_loss_D



    def validation_step(self, batch, batch_idx):
        data, full_target, info  = batch
        output       = self.generator(data)
        target       = full_target[:,:,6:-6,6:-6]
        MSE_loss     = self.meanSquaredError_loss(output, target)
        return MSE_loss

       
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.G_lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.D_lr)
        return [opt_g, opt_d]
