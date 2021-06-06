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

#Config parameters
YOUR_CELL_LINE = "GM18534"
LOW_RES_HIC    = "GM18534_30.hic"

#default
CHRO = 17
STEP = 50 
RES  = 10000
class CustomModule(pl.LightningDataModule):

    def extract_constraint_mats(self):
        print("extracting_constraints")
        for i in range(CHRO,CHRO+1):
            juice_command = "java -jar "\
                    ""+str(self.juicer_tool)+" dump observed KR "\
                    ""+str(self.low_res_fn)+" "+str(i)+" "+str(i)+""\
                    " BP "+str(self.res)+" "+self.line_name+"/Constraints/low_chr"+str(i)+"_res_"+str(self.res)+".txt"
            subprocess.run(juice_command, shell=True)

    def __init__(self):
        super().__init__()
        self.line_name   = YOUR_CELL_LINE #"Your_Line"  #add your cell line name
        self.low_res_fn  = LOW_RES_HIC   #"custom.hic" #add your juicer file here
        self.juicer_tool = "other_tools/juicer_tools_1.22.01.jar" #make sure you have juicer tools from aidenlab
        self.batch_size  = 1
        self.res         = RES
        self.step        = STEP
        self.piece_size  = 269
        if not os.path.exists(self.line_name):
            subprocess.run("mkdir "+self.line_name, shell=True)

        #extract constraints
        if not os.path.exists(self.line_name+"/Constraints"):
            subprocess.run("mkdir "+self.line_name+"/Constraints", shell=True)
        self.extract_constraint_mats()
        #create numpye()
        if not os.path.exists(self.line_name+"/Full_Mats_Coords"):
            subprocess.run("mkdir "+self.line_name+"/Full_Mats_Coords", shell=True)

        if not os.path.exists(self.line_name+"/Full_Mats"):
            subprocess.run("mkdir "+self.line_name+"/Full_Mats", shell=True)
        self.create_numpy()

        #split numpy()
        if not os.path.exists(self.line_name+"/Splits"):
            subprocess.run("mkdir "+self.line_name+"/Splits", shell=True)
        self.split_numpy()
    
    def create_numpy(self):
        for i in range(CHRO,CHRO+1):
            low_txt = self.line_name+"/Constraints/low_chr"+str(i)+"_res_"+str(self.res)+".txt"
            target, coordinates = ut.loadSingleConstraints(low_txt,self.res)
            np.save(self.line_name+"/Full_Mats/chr"+str(i)+"_res_"+str(self.res), target)
            np.save(self.line_name+"/Full_Mats_Coords/coords_chr"+str(i)+"_res_"+str(self.res), coordinates)


    def split_numpy(self):
        for i in range(CHRO, CHRO+1):
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
            self.chros = list(range(CHRO,CHRO+1))
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

def enhance(dm, model):
    condense = 6
    low_hic = torch.from_numpy(dm.test_dataloader().dataset.data)
    #enh_hic = torch.zeros_like(low_hic)
    enh_hic  = torch.zeros(low_hic.shape[0],
            low_hic.shape[1], 
            low_hic.shape[2]-(2*condense),
            low_hic.shape[3]-(2*condense))
    for i in range(0, low_hic.shape[0]):
        print(str(i)+"/"+str(low_hic.shape[0]))
        #enh_hic[i:i+1,:,6:-6,6:-6] = model_vehicle(low_hic[i:i+1,:,:,:]).detach()[0][0]
        enh_hic[i:i+1,:,:,:] = model_vehicle(low_hic[i:i+1,:,:,:]).detach()[0][0]
    return enh_hic

def split2full(splits,
        coords,
        step):
    condense = 6
    mat   = np.zeros((coords.shape[0], coords.shape[0]))
    for i in range(0, splits.shape[0]):
        print(str(i)+"/"+str(splits.shape[0]))
        #mat[i*step:(i*step)+splits.shape[2], i*step:(i*step)+splits.shape[2]] = splits[i,0]
        staPos = i*step+condense
        endPos = (i*step)+splits.shape[2]+condense
        mat[staPos:endPos, staPos:endPos] = splits[i,0]
    return mat

if __name__ == "__main__":
    #dataset
    dm = CustomModule()
    dm.setup()

    #model
    vehicleModel   = vehicle.GAN_Model()
    model_vehicle  = vehicleModel.load_from_checkpoint("Trained_Models/vehicle_gan.ckpt")
    
    low_hic  = torch.from_numpy(dm.test_dataloader().dataset.data)
    coords   = np.load(YOUR_CELL_LINE+"/Full_Mats_Coords/coords_chr"+str(CHRO)+"_res_"+str(RES)+".npy")
    enh_hic  = enhance(dm, model_vehicle)
    full_enh = split2full(enh_hic, coords, STEP)
    if not os.path.isdir(YOUR_CELL_LINE+"/Full_Enhanced"):
        os.mkdir(YOUR_CELL_LINE+"/Full_Enhanced")
    np.save(YOUR_CELL_LINE+"/Full_Enhanced/chr"+str(CHRO)+"_res_"+str(RES)+".npy", full_enh)

