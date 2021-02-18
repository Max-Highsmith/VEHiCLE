import sys
sys.path.append(".")
from Utils.loss import insulation as ins
import matplotlib.pyplot as plt
import glob
import yaml
import subprocess
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Data.GM12878_DataModule import GM12878Module
from Data.K562_DataModule    import K562Module
from Data.IMR90_DataModule   import IMR90Module
from Data.HMEC_DataModule    import HMECModule

#Load Models
import Models.VEHiCLE_Module as vehicle
import Models.deephic        as deephic
import Models.hicplus        as hicplus
import Models.hicsr          as hicsr

methods = ['downs', 'hicplus', 'deephic', 'hicsr', 'vehicle'] 
colors  = ['black', 'silver', 'blue', 'darkviolet', 'coral']

getIns   = ins.computeInsulation()

#vehicle
WEIGHT_PATH="Trained_Models/vehicle_gan.ckpt"
vehicleModel = vehicle.GAN_Model()
model_vehicle = vehicleModel.load_from_checkpoint(WEIGHT_PATH)

##HiCPlus
model_hicplus = hicplus.Net(40,28)
model_hicplus.load_state_dict(torch.load("Trained_Models/hicplus_weights"))

##HiCSR
model_hicsr   = hicsr.Generator(num_res_blocks=15)
HICSR_WEIGHTS = "Trained_Models/hicsr_weights.pth"
model_hicsr.load_state_dict(torch.load(HICSR_WEIGHTS))
model_hicsr.eval()

#DeepHiC
model_deephic = deephic.Generator(scale_factor=1, in_channel=1, resblock_num=5)
model_deephic.load_state_dict(torch.load("Trained_Models/deephic_weights.pytorch"))


for CHRO in [4, 14, 16, 20]:
    RES        = 10000
    PIECE_SIZE = 269
    CELL_LINE = "GM12878"

    if CELL_LINE == "HMEC":
        dm_test = HMECModule(batch_size=1, res=RES, piece_size=PIECE_SIZE)

    if CELL_LINE == "IMR90":
        dm_test = IMR90Module(batch_size=1, res=RES, piece_size=PIECE_SIZE)

    if CELL_LINE == "GM12878":
        dm_test = GM12878Module(batch_size=1, res=RES, piece_size=PIECE_SIZE)

    if CELL_LINE == "K562":
        dm_test = K562Module(batch_size=1, res=RES, piece_size=PIECE_SIZE)

    dm_test.prepare_data()
    dm_test.setup(stage=CHRO)


    full_insulation_dist = {
            'hicsr':[],
            'down':[],
            'vehicle':[],
            'deephic':[],
            'hicplus':[]
            }

    directionality_comp = {
            'hicsr':[],
            'down':[],
            'vehicle':[],
            'deephic':[],
            'hicplus':[],
            'target':[]
            }


    def getTadBorderDists(x,y):
        nearest_distances = []
        for border1 in x:
            if border1 >50 and border1 <101:
                nearest = 9999
                for border2 in y:
                    dist = abs(border1-border2)
                    if dist < nearest:
                        nearest = dist
                nearest_distances.append(nearest)

        return nearest_distances



    STEP_SIZE = 50
    BUFF_SIZE = 36
    pdb.set_trace()
    NUM_ITEMS = dm_test.test_dataloader().dataset.data.shape[0]
    for s, sample in enumerate(dm_test.test_dataloader()):
        print(str(s)+"/"+str(NUM_ITEMS))
        data, target, _ = sample
        downs   = data[0][0]
        target  = target[0][0]
        
        #Pass through Models
        #Pass through HicPlus
        hicplus_out = torch.zeros((PIECE_SIZE, PIECE_SIZE))
        for i in range(0, PIECE_SIZE-40, 28):
            for j in range(0, PIECE_SIZE-40, 28):
                temp                            = data[:,:,i:i+40, j:j+40]
                hicplus_out[i+6:i+34, j+6:j+34] =  model_hicplus(temp)
        hicplus_out = hicplus_out.detach()[6:-6, 6:-6]

        #Pass through Deephic
        deephic_out = torch.zeros((PIECE_SIZE, PIECE_SIZE))
        for i in range(0, PIECE_SIZE-40, 28):
            for j in range(0, PIECE_SIZE -40, 28):
                temp                            = data[:,:,i:i+40, j:j+40]
                deephic_out[i+6:i+34, j+6:j+34] = model_deephic(temp)[:,:,6:34, 6:34]
        deephic_out = deephic_out.detach()[6:-6,6:-6]

        #Pass through HiCSR
        hicsr_out = torch.zeros((PIECE_SIZE, PIECE_SIZE))
        for i in range(0, PIECE_SIZE-40, 28):
            for j in range(0, PIECE_SIZE-40, 28):
                temp                          = data[:,:,i:i+40, j:j+40]
                hicsr_out[i+6:i+34, j+6:j+34] = model_hicsr(temp)
        hicsr_out = hicsr_out.detach()[6:-6, 6:-6]
        hicsr_out = torch.clamp(hicsr_out,0, 100000000)

        #PASS through VeHICLE
        vehicle_out = model_vehicle(data).detach()[0][0]

        downs   = downs[6:-6,6:-6]
        target  = target[6:-6,6:-6]

        directionality_comp['down'].extend(getIns.forward(downs.reshape(1,1,257,257))[1][0][0][0:50].tolist())
        directionality_comp['hicplus'].extend(getIns.forward(hicplus_out.reshape(1,1,257,257))[1][0][0][0:50].tolist())
        directionality_comp['deephic'].extend(getIns.forward(deephic_out.reshape(1,1,257,257))[1][0][0][0:50].tolist())
        directionality_comp['vehicle'].extend(getIns.forward(vehicle_out.reshape(1,1,257,257))[1][0][0][0:50].tolist())
        directionality_comp['hicsr'].extend(getIns.forward(hicsr_out.reshape(1,1,257,257))[1][0][0][0:50].tolist())
        directionality_comp['target'].extend(getIns.forward(target.reshape(1,1,257,257))[1][0][0][0:50].tolist())


        
    down_direction    = np.linalg.norm(np.array(directionality_comp['down'])-np.array(directionality_comp['target']))
    hicplus_direction    = np.linalg.norm(np.array(directionality_comp['hicplus'])-np.array(directionality_comp['target']))
    deephic_direction    = np.linalg.norm(np.array(directionality_comp['deephic'])-np.array(directionality_comp['target']))
    hicsr_direction    = np.linalg.norm(np.array(directionality_comp['hicsr'])-np.array(directionality_comp['target']))
    vehicle_direction    = np.linalg.norm(np.array(directionality_comp['vehicle'])-np.array(directionality_comp['target']))
    print("------"+str(CELL_LINE)+"--Chro:"+str(CHRO)+"-------")
    print("down direction: "     +str(down_direction)+"\n"\
            "hicplus direction: "+str(hicplus_direction)+"\n"\
            "deephic_direction: "+str(deephic_direction)+"\n"\
            "hicsr_direction: "  +str(hicsr_direction)+"\n"\
            "vehicle_direction: "+str(vehicle_direction))
