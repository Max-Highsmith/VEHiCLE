import matplotlib.pyplot as plt
import sys
sys.path.append(".")
import torch
import numpy as np
import pdb

from Utils.loss import insulation as ins
import Models.deephic as deephic
import Models.hicplus as hicplus
import Models.hicsr as hicsr
import Models.VEHiCLE_Module as vehicle
from Data.GM12878_DataModule import GM12878Module
from Data.HMEC_DataModule import HMECModule
from Data.IMR90_DataModule import IMR90Module
from Data.K562_DataModule import K562Module

from Utils import vision_metrics as vm


getIns = ins.computeInsulation()
chro = 14
dm_test = GM12878Module(batch_size=1,
                       res=10000,
                       piece_size=269)
dm_test.prepare_data()
dm_test.setup(stage=chro)

#create models
model_vehicle = vehicle.GAN_Model()
model_hicplus = hicplus.Net(269, 257)
model_hicsr   = hicsr.Generator(num_res_blocks=15)
model_deephic = deephic.Generator(scale_factor=1,
                                in_channel=1,
                                resblock_num=5)

#load models
#model_vehicle = model_vehicle.load_from_checkpoint("Trained_Models/vehicle_gan.ckpt")
model_vehicle = model_vehicle.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=49.ckpt")
model_hicplus.load_state_dict(torch.load("Trained_Models/Big_Models/big_269_hicplus.pytorch"))
model_hicsr.load_state_dict(torch.load("Trained_Models/Big_Models/big_269_hicsr.pth"))
model_deephic.load_state_dict(torch.load("Trained_Models/Big_Models/big_269_deephic.pytorch"))


'''
pdb.set_trace()
for s, sample in enumerate(dm_test.test_dataloader()):
    data, target, _ = sample
    downs           = data[0][0]
    target          = target[0][0]
    vehicle_out     = model_vehicle(data).detach()[0][0]
    hicplus_out     = model_hicplus(data).detach()[0][0]
    hicsr_out       = model_hicsr(data).detach()[0][0]
    deephic_out     = model_deephic(data).detach()[0][0]

    fig, ax = plt.subplots(2,6, figsize=(20,8))
    ax[0,0].imshow(downs, cmap="Reds")
    ax[0,1].imshow(target, cmap="Reds")
    ax[0,2].imshow(deephic_out, cmap="Reds")
    ax[0,3].imshow(hicplus_out, cmap="Reds")
    ax[0,4].imshow(hicsr_out, cmap="Reds")
    ax[0,5].imshow(vehicle_out, cmap="Reds")
    ax[1,0].imshow(downs[100:200,100:200], cmap="Reds")
    ax[1,1].imshow(target[100:200,100:200], cmap="Reds")
    ax[1,2].imshow(hicplus_out[100:200,100:200], cmap="Reds")
    ax[1,3].imshow(deephic_out[100:200,100:200], cmap="Reds")
    ax[1,4].imshow(hicsr_out[100:200,100:200], cmap="Reds")
    ax[1,5].imshow(vehicle_out[100:200,100:200], cmap="Reds")
    ax[1,0].set_xlabel("down")
    ax[1,1].set_xlabel("target")
    ax[1,2].set_xlabel("hicplus")
    ax[1,3].set_xlabel("deephic")
    ax[1,4].set_xlabel("hicsr")
    ax[1,5].set_xlabel("vehicle")
    plt.show()
'''


for CHRO in [20]: #,14,4]:
    RES        = 10000
    PIECE_SIZE = 269
    #CELL_LINE = "GM12878"
    CELL_LINE  = "K562"

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

    for s, sample in enumerate(dm_test.test_dataloader()):
        print(str(s)+"/"+str(dm_test.test_dataloader().dataset.data.shape[0]))
        data, target, _ = sample
        hicsr_out    = model_hicsr(data)[0][0]
        deephic_out  = model_deephic(data)[0][0][6:-6,6:-6]
        hicplus_out  = model_hicplus(data)[0][0]
        vehicle_out  = model_vehicle(data)[0][0]
        downs        = data[0][0][6:-6,6:-6]
        target       = target[0][0][6:-6,6:-6]


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

pdb.set_trace()
v_m  = {}
chro = 16
print("HiCSR")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro)
v_m[chro, 'hicsr']=visionMetrics.getMetrics(model=model_hicsr, spliter="large")

print("deephic")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro)
v_m[chro, 'deephic']=visionMetrics.getMetrics(model=model_deephic, spliter="large_deephic")

print("hicplus")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro)
v_m[chro, 'hicplus']=visionMetrics.getMetrics(model=model_hicplus, spliter="large")

print("vehicle")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro)
v_m[chro, 'vehicle']=visionMetrics.getMetrics(model=model_vehicle,
                                                spliter='vehicle')



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


