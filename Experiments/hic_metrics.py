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

#Load Models
import Model.VEHiCLE_Module as vehicle
import Model.deephic as deephic
import Model.hicplus as hicplus
import Model.hicsr   as hicsr

##VeHICLE
PATH          = "Trained_Models/vehicle_gan.ckpt"
vehicleModel  = vehicle.GAN_Model()
model_vehicle = vehicleModel.load_from_checkpoint(PATH)

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


CHRO       = 4
RES        = 10000
PIECE_SIZE = 269
vehicle_hic  = open("hicqc_inputs/vehicle_"+str(CHRO), 'w')
hicsr_hic    = open("hicqc_inputs/hicsr_"+str(CHRO), 'w')
deephic_hic  = open("hicqc_inputs/deephic_"+str(CHRO), 'w')
hicplus_hic  = open("hicqc_inputs/hicplus_"+str(CHRO), 'w')
original_hic = open("hicqc_inputs/original_"+str(CHRO), 'w')
down_hic     = open("hicqc_inputs/down_"+str(CHRO), 'w')
bins_file    = open("hicqc_inputs/bins_"+str(CHRO)+".bed",'w')


dm_test = GM12878Module(batch_size=1, res=RES, piece_size=PIECE_SIZE)
dm_test.prepare_data()
dm_test.setup(stage=CHRO)

pdb.set_trace()

for s, sample in enumerate(dm_test.test_dataloader()):
    if s >170:
        break

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

    #PASS through VeHICLE TODO TODO
    vehicle_out = model_deepchromap(data).detach()[0][0]


    downs   = downs[6:-6,6:-6]
    target  = target[6:-6,6:-6]
    for i in range(0, 50):      #downs.shape[0]):
        bina = (50*s*RES)+((i+6)*RES)
        bina_end = bina+RES
        bins_file.write(str(CHRO)+"\t"+str(bina)+"\t"+str(bina_end)+"\t"+str(bina)+"\n")
        for j in range(i, 50):  # downs.shape[1]):
            bina = (50*s*RES)+((i+6)*RES)
            binb = (50*s*RES)+((j+6)*RES)
            down_hic.write(    str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(downs[i,j]*100))+"\n")     
            original_hic.write(str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(target[i,j]*100))+"\n") 
            hicplus_hic.write( str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(hicplus_out[i,j]*100))+"\n") 
            deephic_hic.write( str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(deephic_out[i,j]*100))+"\n") 
            hicsr_hic.write(   str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(hicsr_out[i,j]*100))+"\n") 
            vehicle_hic.write( str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(vehicle_out[i,j]*100))+"\n") 


down_hic.close()
bins_file.close()
original_hic.close()
hicplus_hic.close()
deephic_hic.close()
hicsr_hic.close()
vehicle_hic.close()

subprocess.run("gzip hicqc_inputs/vehicle_"+str(CHRO),  shell=True)
subprocess.run("gzip hicqc_inputs/hicsr_"+str(CHRO),    shell=True)
subprocess.run("gzip hicqc_inputs/deephic_"+str(CHRO),  shell=True)
subprocess.run("gzip hicqc_inputs/hicplus_"+str(CHRO),  shell=True)
subprocess.run("gzip hicqc_inputs/original_"+str(CHRO), shell=True)
subprocess.run("gzip hicqc_inputs/down_"+str(CHRO),     shell=True)
subprocess.run("gzip hicqc_inputs/bins_"+str(CHRO)+".bed",     shell=True)

#hic_metric_samples = open("hicqc_inputs/hic_metric.samples", 'w')
#hic_metric_pairs   = open("hicqc_inputs/hic_metric.pairs", 'w')
#SAMPLE_STRING="Down     /home/heracles/Documents/Professional/Research/lsdcm/other_tools/3DChromatin_ReplicateQC/examples/down_"+str(CHRO)+".gz\n"\
#"Original /home/heracles/Documents/Professional/Research/lsdcm/other_tools/3DChromatin_ReplicateQC/examples/original_"+str(CHRO)+".gz\n"
#"HiCPlus  /home/heracles/Documents/Professional/Research/lsdcm/other_tools/3DChromatin_ReplicateQC/examples/hicplus_"+str(CHRO)+".gz\n"
#"DeepHiC  /home/heracles/Documents/Professional/Research/lsdcm/other_tools/3DChromatin_ReplicateQC/examples/deephic_"+str(CHRO)+".gz\n"
#"VEHiCLE  /home/heracles/Documents/Professional/Research/lsdcm/other_tools/3DChromatin_ReplicateQC/examples/vehicle_"+str(CHRO)+".gz"

#PAIR_STRING="Original\tDown\tHiCPlus\tDeepHiC\tVEHiCLE"
#hic_metric_samples.write(SAMPLE_STRING) 
#hic_metric_pairs.write(PAIR_STRING)   

#experiment_command = "3DChromatin_ReplicateQC run_all --metadata_samples hicqc_inputs/hic_metric.samples --metadata_pairs hicqc_inputs/hic_metric.pairs --bins hicqc_inputs/bins_20.bed.gz --outdir qc_results"

#subprocess.run(experiment_command)

#"a3DChromatin_ReplicateQC run_all --metadata_samples other_tools/3DChromatin_ReplicateQC/examples/vehicle_down.samples --metadata_pairs other_tools/3DChromatin_ReplicateQC/examples/vehicle_down.pairs --bins other_tools/3DChromatin_ReplicateQC/examples/bins_20.bed.gz --outdir qc_results


