import warnings
import sys
sys.path.append(".")
import Models.VEHiCLE_Module as vehicle
import pdb
import os
import tmscoring
import glob
import subprocess
import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch

from Data.GM12878_DataModule import GM12878Module

PIECE_SIZE = 269

CHRO_LENGTHS={
        1:248956422,
        2:242193529,
        3:198295559,
        4:190214555,
        5:181538259,
        6:170805979,
        7:159345973,
        8:145138636,
        9:138394717,
        10:133797422,
        11:135086622,
        12:133275309,
        13:114364328,
        14:107043718,
        15:101991189,
        16:90338345,
        17:83257441,
        18:80373285,
        19:58617616,
        20:64444167,
        21:46709983,
        22:50818468
        }

def buildFolders():
    if not os.path.exists('3D_Mod'):
        os.makedirs('3D_Mod')
    if not os.path.exists('3D_Mod/Constraints'):
        os.makedirs('3D_Mod/Constraints')
    if not os.path.exists('3D_Mod/output'):
        os.makedirs('3D_Mod/output')
    if not os.path.exists('3D_Mod/Parameters'):
        os.makedirs('3D_Mod/Parameters')

def convertChroToConstraints(chro,
                            cell_line="GM12878",
                            res=10000):
    bin_num = int(CHRO_LENGTHS[chro]/res)
    print(bin_num)

    if cell_line=="GM12878":
        dm_test = GM12878Module(batch_size=1, res=res, piece_size=PIECE_SIZE)
    dm_test.prepare_data()
    dm_test.setup(stage=chro)

    target_chro    = np.zeros((bin_num, bin_num))
    down_chro      = np.zeros((bin_num, bin_num))
    vehicle_chro   = np.zeros((bin_num, bin_num))

    WEIGHT_PATH    = "Trained_Models/vehicle_gan.ckpt"
    vehicleModel   = vehicle.GAN_Model()
    model_vehicle  = vehicleModel.load_from_checkpoint(WEIGHT_PATH)

    NUM_ENTRIES = dm_test.test_dataloader().dataset.data.shape[0]
    for s, sample in enumerate(dm_test.test_dataloader()):
        if s % 5 ==0:
            print(str(s)+"/"+str(NUM_ENTRIES))
            data, target, _ = sample
            vehicle_output =  model_vehicle(data).detach()[0][0]
            vehicle_output[vehicle_output<(torch.median(vehicle_output)/4)] =0
            data[data<(torch.median(data)/4)]=0
            data   = data[0][0][6:-6, 6:-6]
            target = target[0][0][6:-6, 6:-6]
            #fig, ax = plt.subplots(1,2)
            #ax[0].imshow(target)
            #ax[1].imshow(vehicle_output)
            #plt.show()
            #pdb.set_trace()
            target_const_name   = "3D_Mod/Constraints/chro_"+str(chro)+"_target_"+str(s)+"_"
            data_const_name     = "3D_Mod/Constraints/chro_"+str(chro)+"_data_"+str(s)+"_"
            vehicle_const_name  = "3D_Mod/Constraints/chro_"+str(chro)+"_vehicle_"+str(s)+"_"
            target_constraints  = open(target_const_name, 'w')
            data_constraints    = open(data_const_name, 'w')
            vehicle_constraints = open(vehicle_const_name, 'w')
            for i in range(0, data.shape[0]):
                for j in range(i, data.shape[1]):
                    data_constraints.write(str(i)+"\t"+str(j)+"\t"+str(data[i,j].item())+"\n")
                    target_constraints.write(str(i)+"\t"+str(j)+"\t"+str(target[i,j].item())+"\n")
                    vehicle_constraints.write(str(i)+"\t"+str(j)+"\t"+str(vehicle_output[i,j].item())+"\n")
            target_constraints.close()
            data_constraints.close()
            vehicle_constraints.close()

def buildParameters(chro,
                cell_line="GM12878",
                res=10000):
    constraints  = glob.glob("3D_Mod/Constraints/chro_"+str(chro)+"_*")
    for constraint in  constraints:
        suffix = constraint.split("/")[-1]
        stri = """NUM = 3\r
OUTPUT_FOLDER = 3D_Mod/output/\r
INPUT_FILE = """+constraint+"""\r
CONVERT_FACTOR = 0.6\r
VERBOSE = true\r
LEARNING_RATE = 1\r
MAX_ITERATION = 10000\r"""
        param_f = open("3D_Mod/Parameters/"+suffix, 'w')
        param_f.write(stri)
    
JAR_LOCATION = "other_tools/3DMax/examples/3DMax.jar"
if not os.path.exists(JAR_LOCATION):
    subprocess.run("git clone https://github.com/BDM-Lab/3DMax.git other_tools")

def runSegmentParams(chro, position_index):
    for struc in ['data', 'target', 'vehicle']:
        subprocess.run("java -Xmx5000m -jar "+JAR_LOCATION+" 3D_Mod/Parameters/chro_"+str(chro)+"_"+struc+"_"+str(position_index)+"_", shell=True)

def runParams(chro):
    params = glob.glob("3D_Mod/Parameters/chro_"+str(chro)+"_*")
    for par in params:
            subprocess.run("java -Xmx5000m -jar "+JAR_LOCATION+" "+par, shell=True)

def getSegmentTMScores(chro, position_index):
    data_strucs     = glob.glob("3D_Mod/output/chro_"+str(chro)+"_data_"+str(position_index)+"_*.pdb")
    target_strucs   = glob.glob("3D_Mod/output/chro_"+str(chro)+"_target_"+str(position_index)+"_*.pdb")
    vehicle_strucs  = glob.glob("3D_Mod/output/chro_"+str(chro)+"_vehicle_"+str(position_index)+"_*.pdb")
    struc_types      = [data_strucs, target_strucs, vehicle_strucs]
    struc_type_names = ['data_strucs', 'target_strucs', 'vehicle_strucs'] 
    
    internal_scores = {'data_strucs':[],
                    'target_strucs':[],
                    'vehicle_strucs':[]}


    for struc_type, struc_type_name in zip(struc_types, struc_type_names):
        for i, data_a in enumerate(struc_type):
            for j, data_b in enumerate(struc_type):
                if not struc_type_name in internal_scores.keys():
                    internal_scores[struc_type_name] = []
                if i>=j:
                    continue
                else:
                    alignment = tmscoring.TMscoring(data_a, data_b)
                    alignment.optimise()
                    indiv_tm = alignment.tmscore(**alignment.get_current_values())
                    internal_scores[struc_type_name].append(indiv_tm)

    relative_scores = {'data_strucs':[],
                        'vehicle_strucs':[]}
    for struc_type, struc_type_name in zip(struc_types, struc_type_names):
       if struc_type_name == 'target_strucs':
           continue
       for i, data_a in enumerate(struc_type):
        for j, data_b in enumerate(target_strucs):   
            alignment = tmscoring.TMscoring(data_a, data_b)
            alignment.optimise()
            indiv_tm  = alignment.tmscore(**alignment.get_current_values())
            relative_scores[struc_type_name].append(indiv_tm)
    return relative_scores, internal_scores

def getTMScores(chro):
    internal_scores = {'data_strucs':[],
                    'target_strucs':[],
                    'vehicle_strucs':[]}
    relative_scores = {'data_strucs':[],
                        'vehicle_strucs':[]}

    getSampleNum = lambda a: a.split("_")[-2]
    for position_index in list(map(getSampleNum, glob.glob("3D_Mod/Parameters/chro_"+str(chro)+"_*"))):
        temp_relative_scores, temp_internal_scores = getSegmentTMScores(chro, position_index)
        for key in temp_relative_scores.keys():
            relative_scores[key].extend(temp_relative_scores[key])
        for key in temp_internal_scores.keys():
            internal_scores[key].extend(temp_internal_scores[key])
    print("INTERNAL SCORES")
    for key in internal_scores.keys():
        print(key+":\t"+str(np.mean(internal_scores[key])))
    print("RELATIVE SCORES")
    for key in relative_scores.keys():
        print(key+":\t"+str(np.mean(relative_scores[key])))
    return relative_scores, internal_scores

def viewModels():
    struc_index=0
    chro=20
    models = glob.glob("3D_Mod/output/chro_"+str(chro)+"_*_"+str(struc_index)+"_*.pdb")
    subprocess.run("pymol "+' '.join(models),  shell=True)

def parallelScatter():
    #relative, internal = getTMScores(4)
    chros = [4,14,16,20]
    relative_data = []
    internal_data = []
    for chro in chros:
        relative, internal = getTMScores(chro)
        for key in relative.keys():
            relative_data.append(relative[key])
        for key in internal.keys():
            internal_data.append(internal[key])
    pdb.set_trace()
    #relative
    fig, ax = plt.subplots()
    bp = ax.boxplot(relative_data, 
            positions=[1,2,4,5, 7,8, 10,11],
            patch_artist=True)
    for b, box in enumerate(bp['boxes']):
        if b % 2 ==0:
            box.set(facecolor = 'crimson')
        else:
            box.set(facecolor= 'forestgreen')

    ax.set_xticks([1.5, 4.5, 7.5, 10.5])
    ax.set_xticklabels(['Chro4', 'Chro14', 'Chro16', 'Chro20'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Relative")
    plt.show()

    fig, ax = plt.subplots()
    bp = ax.boxplot(internal_data, 
            positions=[1,2,3, 5,6,7, 9,10,11, 13,14,15],
            patch_artist=True)
    for b, box in enumerate(bp['boxes']):
        if b % 3 ==0:
            box.set(facecolor = 'crimson')
        elif b%3 ==1:
            box.set(facecolor = 'bisque')
        else:
            box.set(facecolor= 'forestgreen')

    ax.set_xticks([2,6,10,14])
    ax.set_xticklabels(['Chro4', 'Chro14', 'Chro16', 'Chro20'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Internal")

    pdb.set_trace()

if __name__ == "__main__":
    #buildFolders()
    #for chro in [4,16,14,20]:
    #    convertChroToConstraints(chro)
    #    buildParameters(chro)
    #    runParams(chro)
    #getTMScores(4)
    parallelScatter()
    #viewModels()
