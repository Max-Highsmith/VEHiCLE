import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from matplotlib.widgets import Slider
import pdb
import torch
import sys
sys.path.append("Data")
sys.path.append(".")
sys.path.append("../")
import argparse
import numpy as np
from sklearn.decomposition import PCA
import yaml
import glob
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from Data.GM12878_DataModule import GM12878Module
from Models.VAE_Module import VAE_Model

#PARAMS
NUM_SLIDERS = 15

PATH    = "Trained_Models/vehicle_vae.ckpt"
hparams = yaml.load(open("Trained_Models/vehicle_vae_hparams.yaml"))
print(hparams)

dm_train = GM12878Module()
dm_train.prepare_data()
dm_train.setup(stage='fit')
ds        = dm_train.val_dataloader().dataset.data

#Get Model
model   = VAE_Model(
        condensed_latent=hparams['condensed_latent'],
        gamma=['gamma'],
        kld_weight=['kld_weight'],
        latent_dim=hparams['latent_dim'],
        lr=hparams['lr'],
        pre_latent=hparams['pre_latent'])

pretrained_model = model.load_from_checkpoint(PATH)
pretrained_model.freeze()

#fit PCA
real_z = pretrained_model.get_z(torch.tensor(ds))
pca    = PCA(n_components=NUM_SLIDERS)
cond_z = pca.fit_transform(real_z[0])

def update(val):
    cond_vec =[] 
    for i in range(0, NUM_SLIDERS):
        cond_vec.append(sliders[i].val)
    cond_vec = np.array([cond_vec])
    full_loc = pca.inverse_transform(cond_vec)
    out_im   = pretrained_model.decode(torch.from_numpy(full_loc).reshape(1,1,1, hparams['latent_dim']).type(torch.float32))
    print("out_im", out_im.shape)
    ax[1].imshow(out_im[0][0], cmap="Reds")
    print(cond_vec)


fig, ax = plt.subplots(1,2)
ax[0].set_visible(False)
ax[1].set_xticks([])
ax[1].set_yticks([])
axiss   = []
sliders = []
y = 0
for i in range(0, NUM_SLIDERS):
    y = 0.1+(int(i)*0.05)
    tempa  = plt.axes([0.2, y, 0.25, 0.02], facecolor="red")
    temps  = Slider(tempa,
            'PC'+str(i)+": "+"({:.2f}%)".format(100*pca.explained_variance_ratio_[i]),
            -10,
            10,
            valinit=0)
    axiss.append(tempa)
    sliders.append(temps)
    sliders[i].on_changed(update)
plt.show()
