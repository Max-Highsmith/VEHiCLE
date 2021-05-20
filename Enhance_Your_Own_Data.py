import matplotlib.pyplot as plt
import torch
import Models.VEHiCLE_Module as vehicle
import matplotlib.pyplot as plt
import os
import sys
from Utils import utils as ut
import pdb
import subprocess
import glob
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import random_split, DataLoader, Dataset

class CustomModule(pl.LightningDataModule):

    def extract_constraint_mats(self):
        for i in range(1,23):
            juice_command = "java -jar "\
                    ""+str(self.juicer_tool)+" dump observed KR "\
                    ""+str(self.low_res_fn)+" "+str(i)+" "+str(i)+""\
                    " BP "+str(self.res)+" "+self.line_name+"/Constraints/low_chr"+str(i)+"_res_"+str(self.res)+".txt"
            subprocess.run(juice_command, shell=True)

    def __init__(self):
        super().__init__()
        self.line_name   = "Your_Line"  #add your cell line name
        self.low_res_fn  = "custom.hic" #add your juicer file here
        self.juicer_tool = "other_tools/juicer_tools_1.22.01.jar" #make sure you have juicer tools from aidenlab
        self.batch_size  = 1
        self.res         = 10000
        self.step        = 50
        self.piece_size  = 269
        if not os.path.exists("self.line_name"):
            subprocess.run("mkdir "+self.line_name, shell=True)

        #extract constraints
        if not os.path.exists(self.line_name+"/Constraints"):
            subprocess.run("mkdir "+self.line_name+"/Constraints", shell=True)
            self.extract_constraint_mats()
        #create numpye()
        if not os.path.exists(self.line_name+"/Full_Mats"):
            subprocess.run("mkdir "+self.line_name+"/Full_Mats", shell=True)
            self.create_numpy()

        #split numpy()
        if not os.path.exists(self.line_name+"/Splits"):
            subprocess.run("mkdir "+self.line_name+"/Splits", shell=True)
            self.split_numpy()
    
    def create_numpy(self):
        for i in range(1,23):
            low_txt = self.line_name+"/Constraints/low_chr"+str(i)+"_res_"+str(self.res)+".txt"
            target, data = ut.loadBothConstraints(low_txt,
                                                low_txt,
                                                self.res)
            np.save(self.line_name+"/Full_Mats/chr"+str(i)+"_res_"+str(self.res), target)


    def split_numpy(self):
        for i in range(1,23):
            data = ut.splitPieces(self.line_name+"/Full_Mats/chr"+str(i)+"_res_"+str(self.res)+".npy",
                    self.piece_size,
                    self.step)
            np.save(self.line_name+"/Splits/chr_"+str(i)+"_res_"+str(self.res)+"_piece_"+str(self.piece_size),
                    data)
        print("0")

    class lowResDataset(Dataset):
        def __init__(self,
                line_name,
                res,
                piece_size):
            self.line_name  = line_name
            self.res        = res
            self.piece_size = piece_size
            self.chros = list(range(1,4))
            self.data  = np.load(self.line_name+"/Splits/chr_"+str(self.chros[0])+"_res_"+str(self.res)+"_piece_"+str(self.piece_size)+".npy")
            for c, chro in enumerate(self.chros[1:]):
                temp   = np.load(self.line_name+"/Splits/chr_"+str(chro)+"_res_"+str(self.res)+"_piece_"+str(self.piece_size)+".npy")
                self.data = np.concatenate((self.data, temp))

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            return self.data[idx]


    def setup(self):
        self.set = self.lowResDataset(line_name=self.line_name,
                res=self.res,
                piece_size=self.piece_size)

    def test_dataloader(self):
        return DataLoader(self.set)

if __name__ == "__main__":
    print("main")
    #dataset
    dm = CustomModule()
    dm.setup()

    #Choose index of region interested in viewing
    region = 10
    ds = torch.from_numpy(dm.test_dataloader().dataset.data[region:region+1])

    #model
    vehicleModel   = vehicle.GAN_Model()
    model_vehicle  = vehicleModel.load_from_checkpoint("Trained_Models/vehicle_gan.ckpt")
    vehicle_out  = model_vehicle(ds).detach()[0][0]
    plt.imshow(vehicle_out)
    plt.show()
    print(vehicle_out.shape)
