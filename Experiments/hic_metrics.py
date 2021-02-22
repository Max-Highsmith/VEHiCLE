import os
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
import glob
import yaml
import subprocess
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Data.GM12878_DataModule import GM12878Module

CHRO       = 4
RES        = 10000
PIECE_SIZE = 269

#Load Models
import Models.VEHiCLE_Module as vehicle
import Models.deephic as deephic
import Models.hicplus as hicplus
import Models.hicsr   as hicsr

##VeHICLE
PATH          = "Trained_Models/vehicle_gan.ckpt"
#PATH           = "lightning_logs/version_0/checkpoints/epoch=49.ckpt"
vehicleModel  = vehicle.GAN_Model()
model_vehicle = vehicleModel.load_from_checkpoint(PATH)

##HiCPlus
model_hicplus_small = hicplus.Net(40,28)
model_hicplus_big = hicplus.Net(269, 257)

model_hicplus_small.load_state_dict(torch.load("Trained_Models/hicplus_weights"))
model_hicplus_big.load_state_dict(torch.load("Trained_Models/Big_Models/big_269_hicplus.pytorch"))

##HiCSR
model_hicsr_small   = hicsr.Generator(num_res_blocks=15)
HICSR_WEIGHTS = "Trained_Models/hicsr_weights.pth"
model_hicsr_small.load_state_dict(torch.load(HICSR_WEIGHTS))
model_hicsr_small.eval()

model_hicsr_big = hicsr.Generator(num_res_blocks=15)
model_hicsr_big.load_state_dict(torch.load("Trained_Models/Big_Models/big_269_hicsr.pth"))



#DeepHiC
model_deephic_small = deephic.Generator(scale_factor=1, in_channel=1, resblock_num=5)
model_deephic_small.load_state_dict(torch.load("Trained_Models/deephic_weights.pytorch"))
model_deephic_big = deephic.Generator(scale_factor=1, in_channel=1, resblock_num=5)
model_deephic_big.load_state_dict(torch.load("Trained_Models/Big_Models/big_269_deephic.pytorch"))


if not os.path.isdir("hicqc_inputs"):
   os.mkdir("hicqc_inputs")

vehicle_hic       = open("hicqc_inputs/vehicle_"+str(CHRO), 'w')
hicsr_hic_big     = open("hicqc_inputs/hicsr_big_"+str(CHRO), 'w')
hicsr_hic_small   = open("hicqc_inputs/hicsr_small_"+str(CHRO), 'w')
deephic_hic_big   = open("hicqc_inputs/deephic_big_"+str(CHRO), 'w')
deephic_hic_small = open("hicqc_inputs/deephic_small_"+str(CHRO), 'w')
hicplus_hic_big   = open("hicqc_inputs/hicplus_big_"+str(CHRO), 'w')
hicplus_hic_small = open("hicqc_inputs/hicplus_small_"+str(CHRO), 'w')
original_hic      = open("hicqc_inputs/original_"+str(CHRO), 'w')
down_hic          = open("hicqc_inputs/down_"+str(CHRO), 'w')
bins_file         = open("hicqc_inputs/bins_"+str(CHRO)+".bed",'w')


dm_test = GM12878Module(batch_size=1, res=RES, piece_size=PIECE_SIZE)
dm_test.prepare_data()
dm_test.setup(stage=CHRO)



for s, sample in enumerate(dm_test.test_dataloader()):
    print(str(s)+"/"+str(dm_test.test_dataloader().dataset.data.shape[0]))
    if s >100:
        break

    data, target, _ = sample
    downs   = data[0][0]
    target  = target[0][0]
    
    #Pass through Models
    #Pass through HicPlus
    hicplus_out_small = torch.zeros((PIECE_SIZE, PIECE_SIZE))
    for i in range(0, PIECE_SIZE-40, 28):
        for j in range(0, PIECE_SIZE-40, 28):
            temp                            = data[:,:,i:i+40, j:j+40]
            hicplus_out_small[i+6:i+34, j+6:j+34] =  model_hicplus_small(temp)
    hicplus_out_small = hicplus_out_small.detach()[6:-6, 6:-6]

    hicplus_out_big = model_hicplus_big(data).detach()[0][0]

    #Pass through Deephic
    deephic_out_small = torch.zeros((PIECE_SIZE, PIECE_SIZE))
    for i in range(0, PIECE_SIZE-40, 28):
        for j in range(0, PIECE_SIZE -40, 28):
            temp                            = data[:,:,i:i+40, j:j+40]
            deephic_out_small[i+6:i+34, j+6:j+34] = model_deephic_small(temp)[:,:,6:34, 6:34]
    deephic_out_small = deephic_out_small.detach()[6:-6,6:-6]
    deephic_out_big   = model_deephic_big(data).detach()[0][0][6:-6,6:-6]

    #Pass through HiCSR
    hicsr_out_small = torch.zeros((PIECE_SIZE, PIECE_SIZE))
    for i in range(0, PIECE_SIZE-40, 28):
        for j in range(0, PIECE_SIZE-40, 28):
            temp                          = data[:,:,i:i+40, j:j+40]
            hicsr_out_small[i+6:i+34, j+6:j+34] = model_hicsr_small(temp)
    hicsr_out_small = hicsr_out_small.detach()[6:-6, 6:-6]
    hicsr_out_small = torch.clamp(hicsr_out_small,0, 100000000)

    hicsr_out_big   = model_hicsr_big(data).detach()[0][0]

    #PASS through VeHICLE TODO TODO
    vehicle_out = model_vehicle(data).detach()[0][0]


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
            hicplus_hic_big.write( str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(hicplus_out_big[i,j]*100))+"\n") 
            hicplus_hic_small.write( str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(hicplus_out_small[i,j]*100))+"\n") 
            deephic_hic_big.write( str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(deephic_out_big[i,j]*100))+"\n") 
            deephic_hic_small.write( str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(deephic_out_small[i,j]*100))+"\n") 
            hicsr_hic_big.write(   str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(hicsr_out_big[i,j]*100))+"\n") 
            hicsr_hic_small.write(   str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(hicsr_out_small[i,j]*100))+"\n") 
            vehicle_hic.write( str(CHRO)+"\t"+str(bina)+"\t"+str(CHRO)+"\t"+str(binb)+"\t"+str(int(vehicle_out[i,j]*100))+"\n") 


down_hic.close()
bins_file.close()
original_hic.close()
hicplus_hic_big.close()
hicplus_hic_small.close()
deephic_hic_big.close()
deephic_hic_small.close()
hicsr_hic_big.close()
hicsr_hic_small.close()
vehicle_hic.close()

subprocess.run("gzip hicqc_inputs/vehicle_"+str(CHRO),  shell=True)
subprocess.run("gzip hicqc_inputs/hicsr_small_"+str(CHRO),    shell=True)
subprocess.run("gzip hicqc_inputs/deephic_small_"+str(CHRO),  shell=True)
subprocess.run("gzip hicqc_inputs/hicplus_small_"+str(CHRO),  shell=True)
subprocess.run("gzip hicqc_inputs/hicsr_big_"+str(CHRO),    shell=True)
subprocess.run("gzip hicqc_inputs/deephic_big_"+str(CHRO),  shell=True)
subprocess.run("gzip hicqc_inputs/hicplus_big_"+str(CHRO),  shell=True)
subprocess.run("gzip hicqc_inputs/original_"+str(CHRO), shell=True)
subprocess.run("gzip hicqc_inputs/down_"+str(CHRO),     shell=True)
subprocess.run("gzip hicqc_inputs/bins_"+str(CHRO)+".bed",     shell=True)


tool_names   = ['hicplus_small', 'deephic_small', 'hicsr_small', 'hicplus_big', 'deephic_big', 'hicsr_big', 'vehicle', 'down']
BASE_STR = '/home/heracles/Documents/Professional/Research/VEHiCLE/hicqc_inputs/'
sample_files = [
            'hicqc_inputs/metric_hicplus_small_'+str(CHRO)+".samples",
            'hicqc_inputs/metric_deephic_small_'+str(CHRO)+".samples",
            'hicqc_inputs/metric_hicsr_small_'+str(CHRO)+".samples",
            'hicqc_inputs/metric_hicplus_big_'+str(CHRO)+".samples",
            'hicqc_inputs/metric_deephic_big_'+str(CHRO)+".samples",
            'hicqc_inputs/metric_hicsr_big_'+str(CHRO)+".samples",
            'hicqc_inputs/metric_vehicle_'+str(CHRO)+".samples",
            'hicqc_inputs/metric_down_'+str(CHRO)+".samples"
            ]

pair_files  = [
            'hicqc_inputs/metric_hicplus_small_'+str(CHRO)+".pairs",
            'hicqc_inputs/metric_deephic_small_'+str(CHRO)+".pairs",
            'hicqc_inputs/metric_hicsr_small_'+str(CHRO)+".pairs",
            'hicqc_inputs/metric_hicplus_big_'+str(CHRO)+".pairs",
            'hicqc_inputs/metric_deephic_big_'+str(CHRO)+".pairs",
            'hicqc_inputs/metric_hicsr_big_'+str(CHRO)+".pairs",
            'hicqc_inputs/metric_vehicle_'+str(CHRO)+".pairs",
            'hicqc_inputs/metric_down_'+str(CHRO)+".pairs"
            ]

for tool_name, sample_fn, pair_fn in zip(tool_names, sample_files, pair_files):
    hic_metric_sample = open(sample_fn, 'w')
    hic_metric_pair   = open(pair_fn, 'w')
    SAMPLE_STRING="original     "+BASE_STR+"original_"+str(CHRO)+".gz\n"+str(tool_name)+"    "+BASE_STR+str(tool_name)+"_"+str(CHRO)+".gz"
    PAIR_STRING  = "original\t"+str(tool_name)
    hic_metric_sample.write(SAMPLE_STRING)
    hic_metric_pair.write(PAIR_STRING)
'''

hic_metric_samples = open("hicqc_inputs/hic_metric.samples", 'w')
hic_metric_pairs   = open("hicqc_inputs/hic_metric.pairs", 'w')
SAMPLE_STRING="Down     /home/heracles/Documents/Professional/Research/lsdcm/other_tools/3DChromatin_ReplicateQC/examples/down_"+str(CHRO)+".gz\n"\
"Original /home/heracles/Documents/Professional/Research/lsdcm/other_tools/3DChromatin_ReplicateQC/examples/original_"+str(CHRO)+".gz\n"
"HiCPlus  /home/heracles/Documents/Professional/Research/lsdcm/other_tools/3DChromatin_ReplicateQC/examples/hicplus_"+str(CHRO)+".gz\n"
"DeepHiC  /home/heracles/Documents/Professional/Research/lsdcm/other_tools/3DChromatin_ReplicateQC/examples/deephic_"+str(CHRO)+".gz\n"
"VEHiCLE  /home/heracles/Documents/Professional/Research/lsdcm/other_tools/3DChromatin_ReplicateQC/examples/vehicle_"+str(CHRO)+".gz"

PAIR_STRING="Original\tDown\tHiCPlus\tDeepHiC\tVEHiCLE"
hic_metric_samples.write(SAMPLE_STRING) 
hic_metric_pairs.write(PAIR_STRING)   

#if not os.path.isdir("other_tools/3DChromatin_ReplicateQC"):
#    subprocess.run("git clone https://github.com/kundajelab/3DChromatin_ReplicateQC other_tools/3DChromatin_ReplicateQC", shell=True)
#    subprocess.run("", shell=True)
#experiment_command = "3DChromatin_ReplicateQC run_all --metadata_samples hicqc_inputs/hic_metric.samples --metadata_pairs hicqc_inputs/hic_metric.pairs --bins hicqc_inputs/bins_20.bed.gz --outdir qc_results"

#subprocess.run(experiment_command)

#"3DChromatin_ReplicateQC run_all --metadata_samples other_tools/3DChromatin_ReplicateQC/examples/vehicle_down.samples --metadata_pairs other_tools/3DChromatin_ReplicateQC/examples/vehicle_down.pairs --bins other_tools/3DChromatin_ReplicateQC/examples/bins_20.bed.gz --outdir qc_results
'''


