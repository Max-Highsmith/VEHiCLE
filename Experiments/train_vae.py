import pdb
import sys
sys.path.append(".")
from pytorch_lightning import Trainer
from Data.GM12878_DataModule import GM12878Module
from Models.VAE_Module import VAE_Model
dm      = GM12878Module()
dm.prepare_data()
dm.setup(stage='fit')
pargs = {'batch_size': 512,
        'condensed_latent': 3,
        'gamma': 1.0, 
        'kld_weight': .000001,
        'kld_weight_inc': 0.000,
        'latent_dim': 200,
        'lr': 0.00001,
        'pre_latent': 4608}
model    = VAE_Model(batch_size=pargs['batch_size'],
                    condensed_latent=pargs['condensed_latent'],
                    gamma=pargs['gamma'],
                    kld_weight=pargs['kld_weight'],
                    kld_weight_inc=pargs['kld_weight_inc'],
                    latent_dim=pargs['latent_dim'],
                    lr=pargs['lr'],
                    pre_latent=pargs['pre_latent'])
trainer = Trainer(gpus=1)
trainer.fit(model, dm)

'''
pargs = {'batch_size': 512,
        'condensed_latent': 3,
        'gamma': 1.0, 
        'kld_weight': .0001,
        'kld_weight_inc': 0.000,
        'latent_dim': 110,
        'lr': 0.00001,
        'pre_latent': 4608}
'''
