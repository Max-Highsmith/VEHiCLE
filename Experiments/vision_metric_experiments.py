import sys
sys.path.append(".")
import numpy as np
import glob
import yaml
import matplotlib.pyplot as plt
import pdb
import torch
import numpy

import Models.VEHiCLE_Module as vehicle
#other models
import Models.hicsr   as hicsr
import Models.deephic as deephic
import Models.hicplus as hicplus

from Utils import vision_metrics as vm
from Data.GM12878_DataModule import GM12878Module

#load data
dm_test = GM12878Module(batch_size=1, res=10000, piece_size=269)
dm_test.prepare_data()
dm_test.setup(stage=14)

ds     = torch.from_numpy(dm_test.test_dataloader().dataset.data[65:66])
target = torch.from_numpy(dm_test.test_dataloader().dataset.target[65:66])

vehicleModel  = vehicle.GAN_Model()
model_vehicle = vehicleModel.load_from_checkpoint("Trained_Models/vehicle_gan.ckpt")

model_hicplus = hicplus.Net(40,28)
model_hicplus.load_state_dict(torch.load("Trained_Models/hicplus_weights"))

model_hicsr   = hicsr.Generator(num_res_blocks=15)
HICSR_WEIGHTS  = "Trained_Models/hicsr_weights.pth"
model_hicsr.load_state_dict(torch.load(HICSR_WEIGHTS))
model_hicsr.eval()

model_deephic = deephic.Generator(scale_factor=1, in_channel=1, resblock_num=5)
model_deephic.load_state_dict(torch.load("Trained_Models/deephic_weights.pytorch"))

#pass through models
vehicle_out = model_vehicle(ds).detach()[0][0]

FULL_RES    = 269
hicplus_out = torch.zeros((269,269))
for i in range(0, FULL_RES-40, 28):
    for j in range(0, FULL_RES-40, 28):
        temp                  = ds[:,:,i:i+40, j:j+40]
        hicplus_out[i+6:i+34, j+6:j+34] =  model_hicplus(temp)
hicplus_out = hicplus_out.detach()[6:-6, 6:-6]

deephic_out = torch.zeros((FULL_RES, FULL_RES))
for i in range(0, FULL_RES-40, 28):
    for j in range(0, FULL_RES -40, 28):
        temp                        = ds[:,:,i:i+40, j:j+40]
        deephic_out[i+6:i+34, j+6:j+34] = model_deephic(temp)[:,:,6:34, 6:34]
deephic_out = deephic_out.detach()[6:-6,6:-6]


hicsr_out = torch.zeros(FULL_RES, FULL_RES)
for i in range(0, FULL_RES-40, 28):
    for j in range(0, FULL_RES-40, 28):
        temp                          = ds[:,:,i:i+40, j:j+40]
        hicsr_out[i+6:i+34, j+6:j+34] = model_hicsr(temp).detach()
hicsr_out = hicsr_out[6:-6, 6:-6]

lowres_out  = ds[0][0][6:-6,6:-6]
target_out  = target[0][0][6:-6,6:-6]


#show comparison plots
fig, ax = plt.subplots(2,6)
for i in range(0, 2):
    for j in range(0,6):
       ax[i,j].set_xticks([])
       ax[i,j].set_yticks([])


#contact maps
ax[0,0].imshow(lowres_out,  cmap="Reds")
ax[0,1].imshow(hicplus_out, cmap="Reds")
ax[0,2].imshow(deephic_out, cmap="Reds")
ax[0,3].imshow(hicsr_out,   cmap="Reds")
ax[0,4].imshow(vehicle_out, cmap="Reds")
ax[0,5].imshow(target_out,  cmap="Reds")

ax[1,0].imshow(lowres_out[40:140,40:140], cmap="Reds")
ax[1,1].imshow(hicplus_out[40:140,40:140], cmap="Reds")
ax[1,2].imshow(deephic_out[40:140,40:140], cmap="Reds")
ax[1,3].imshow(hicsr_out[40:140,40:140], cmap="Reds")
ax[1,4].imshow(vehicle_out[40:140,40:140], cmap="Reds")
ax[1,5].imshow(target_out[40:140,40:140], cmap="Reds")
'''
ax[0,0].imshow(lowres_out -target_out, cmap="RdBu")
ax[0,1].imshow(hicplus_out -target_out, cmap="RdBu")
ax[0,2].imshow(deephic_out -target_out, cmap="RdBu")
ax[0,3].imshow(hicsr_out -target_out, cmap="RdBu")
ax[0,4].imshow(vehicle_out - target_out, cmap="RdBu")
ax[0,5].imshow(target_out -target_out, cmap="RdBu")

ax[1,0].imshow(lowres_out[40:140,40:140] - target_out[40:140, 40:140], cmap="RdBu")
ax[1,1].imshow(hicplus_out[40:140,40:140]- target_out[40:140, 40:140], cmap="RdBu")
ax[1,2].imshow(deephic_out[40:140,40:140] - target_out[40:140, 40:140], cmap="RdBu")
ax[1,3].imshow(hicsr_out[40:140,40:140]- target_out[40:140, 40:140], cmap="RdBu")
ax[1,4].imshow(vehicle_out[40:140,40:140] - target_out[40:140, 40:140], cmap="RdBu")
ax[1,5].imshow(target_out[40:140,40:140]- target_out[40:140, 40:140], cmap="RdBu")
'''

ax[0,0].set_title("DownSampled")
ax[0,1].set_title("HiCPlus")
ax[0,2].set_title("DeepHiC")
ax[0,3].set_title("HiCSR")
ax[0,4].set_title("VEHiCLE(ours)")
ax[0,5].set_title("Target")
plt.show()


v_m ={}
chro = 20
#compute vision metrics
print("vehicle")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro)
v_m[chro, 'vehicle']=visionMetrics.getMetrics(model=model_vehicle, spliter="vehicle")
 
print("HiCSR")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro)
v_m[chro, 'hicsr']=visionMetrics.getMetrics(model=model_hicsr, spliter="hicsr")

print("deephic")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro)
v_m[chro, 'deephic']=visionMetrics.getMetrics(model=model_deephic, spliter="deephic")

print("hicplus")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro)
v_m[chro, 'hicplus']=visionMetrics.getMetrics(model=model_hicplus, spliter="hicplus")

model_names  = ['downsampled', 'vehicle','hicsr', 'deephic','hicplus']
metric_names = ['pcc','spc','ssim', 'mse', 'snr']

cell_text = []
for mod_nm in model_names:
    met_list = []
    for met_nm in metric_names:
        if mod_nm=="downsampled":
            met_list.append("{:.4f}".format(np.mean(v_m[chro, "vehicle"]['pre_'+str(met_nm)])))
        else:
            met_list.append("{:.4f}".format(np.mean(v_m[chro, mod_nm]['pas_'+str(met_nm)])))
    cell_text.append(met_list)

plt.subplots_adjust(left=0.2, top=0.8)
plt.table(cellText=cell_text, rowLabels=model_names, colLabels=metric_names, loc='top')
plt.title(chro)
plt.show()



