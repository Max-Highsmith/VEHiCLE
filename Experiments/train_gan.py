import pdb
import sys
sys.path.append(".")
from pytorch_lightning import Trainer
from Data.GM12878_DataModule import GM12878Module
from Models.VEHiCLE_Module import GAN_Model

dm  = GM12878Module(batch_size=1, piece_size=269)
dm.prepare_data()
dm.setup(stage='fit')

model = GAN_Model()

trainer = Trainer(gpus=1)
trainer.fit(model, dm)
