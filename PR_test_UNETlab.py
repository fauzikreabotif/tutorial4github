
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:46:05 2022

@author: rn
"""
import os
import numpy as np
from os.path import join
from skimage import io
import scipy
from scipy import ndimage
import torch

from mayavi import mlab

root_function = r'D:\RIZKI\CHIBA\KAMPUS\Bleeding'
os.chdir(root_function)
from function_LUNA_LIDC import *
root_function = r'D:\RIZKI\CHIBA\KAMPUS\Bleeding\ct_bleeding_detection'
os.chdir(root_function)
from networks.unet3d import Unet3d
from utils.create_voi_for_bleeding_unet import do_normalization

def adj_frame(shape, patch_length, stride_length):
    awall = []; plusss = []; kalis = [];
    for i,n,t in zip(shape, patch_length, stride_length):
        awal = 0; pluss = 0
        s = (i-n)%(n-t)
        kali = ((i-n)//(n-t))+1
        if s>0.3*(n-t): # here, we state set a treshold for adding pads 
            pluss = abs(s-(n-t))
            pluss += pluss%2 # genapkan
            pluss /=2
            kali+1
        else:
            awal = s//2
        awall.append(int(awal)); plusss.append(int(pluss)); kalis.append(int(kali))
    return awall, plusss, kalis

def ext_patches(iso_dicom, patch_length, stride_length):
    
    awall, plusss, kalis = adj_frame(np.shape(iso_dicom), patch_length, stride_length)
    
    j,k,l = plusss
    mod_dicom = np.pad(iso_dicom, ((j,j),(k,k),(l,l)) , 'constant', constant_values=np.min(iso_dicom))
    shape1 = np.shape(mod_dicom)
    
    
    rang = [list(range(a, i-j, j-k)) for i,a,j,k in zip(shape1, awall, patch_length, stride_length)]
    patches = []
    for i in rang[0]:
        for j in rang[1]:
            for k in rang[2]:
               patches.append( mod_dicom[i:i+patch_length[0], j:j+patch_length[1], k:k+patch_length[2]])
    notes = {'rank': rang, 'shape_plusPad': shape1, 'awal': awall, 'pad_pluss': plusss}
    
    return patches, notes

def recons_patches_v2(patches, notes): # this version, let the function to reconstruct the binary patches
    # patches-> extracted patches are resulted by  ext_patches
    # notes -> the position of patches and 3Dplus pad size. 
    mod_dicom = np.zeros(notes['shape_plusPad'])
    norm_dicom = np.zeros(notes['shape_plusPad'])
    
    pz, px, py = patches[0].shape
    nrm_patch = np.ones((pz,px,py))
    
    z,x,y = notes['shape_plusPad']
    rang = notes['rank']
    pad_plus = notes['pad_pluss']
    datakosong = mod_dicom.copy()*0
    st = 0
    for i in rang[0]:
        for j in rang[1]:
            for k in rang[2]:
                datakosong[i:i+patch_length[0], j:j+patch_length[1], k:k+patch_length[2]] += patches[st] #--------->> bedanya disini, nnti di bagi dgn normalize nya
                norm_dicom[i:i+patch_length[0], j:j+patch_length[1], k:k+patch_length[2]] += nrm_patch
                st+=1
                
    norm_dicom[norm_dicom==0]=1             
    datakosong/=norm_dicom
    data_awal =  datakosong[pad_plus[0]:z-pad_plus[0], pad_plus[1]:x-pad_plus[1], pad_plus[2]:y-pad_plus[2]]     
         
    return data_awal


#=========  LOAD MODEL   ===================
# Define network (U-Net)
# sesuai yg ada di .yml 
depth = 3
first_filter_num = 32
use_residual_block = False
model = Unet3d(1, 1, depth, first_filter_num, use_residual_block)
device = f'cuda:{0}'
model_file_name = r'\\192.168.10.21\home\bleeding_data_alanysis\init_trial_230601\test1\model_best_test1.pth'
model.load_state_dict(torch.load(model_file_name, map_location=device))
model = model.to(device)
model.eval()

# ===========================================================================
data_add = r'\\192.168.10.21\WorkShare\TraumaCT_bleeding_230516'
grup = 6
with open(join(data_add, 'groups_230516.csv'),'r') as data:
    data_test = [join(data_add, line[0]) for line in  csv.reader(data) if int(line[1]) == grup]

patch_length = 64, 64, 64
stride_length = 10,10,10
for ii in data_test:
    org_volume_n = io.imread(join(ii, 'iso_original.mhd'), plugin='simpleitk')
    ref = np.unique(org_volume_n[:5,:5,:5])[0]
    org_volume_n = do_normalization(org_volume_n, [50, 350])
    org_mask = io.imread(join(data_add, ii, 'iso_bleeding_label.mhd'), plugin='simpleitk').astype(np.int8)
    # org_label = io.imread(join(data_add, ii, 'iso_bleeding_label.mhd'), plugin='simpleitk').astype(np.int8)
    
    patches, notes = ext_patches(org_volume_n, patch_length, stride_length)
    pred_pad = []
    for pi in patches:
        ps = np.expand_dims(np.expand_dims(pi, axis=0), axis=0)
        ps = torch.from_numpy(ps).type(torch.FloatTensor)
        ps = ps.to(device)
        outputs = model(ps)
        outputs = outputs.cpu()[0, 0].detach().numpy().copy()
        pred_pad.append(outputs)
    
    
    iso_dicom2 = recons_patches_v2(pred_pad, notes)
    iso_dicom2 = iso_dicom2>0.9
    structure = np.ones((5, 5, 5))
    iso_dicom2 = scipy.ndimage.binary_closing(iso_dicom2,structure)
    iso_dicom2  = morphology.remove_small_objects(scipy.ndimage.binary_fill_holes(iso_dicom2),  min_size=5)
    
    
    enh = org_volume_n.copy()
    enh[enh<-300] = -300
    enh[enh>1000] = 1000
    enh = streching(enh, [-300, 1000], [0,255], 1)
    
    
    plot_3d([iso_dicom2*1, org_mask], enh , alpas = [1,0.1], colors = [(0.5,0.5,0.9), (1,0,0)], xyz=True, bright=True)
    
    #nyari center of mass prediksi
    # iso_dicom2 = iso_dicom2*org_label
    structure = np.ones((3, 3, 3))
    labeled_volume, candidate_num = ndimage.label(iso_dicom2, structure)
    label_idxs = np.arange(1, candidate_num + 1)
    gravities = ndimage.measurements.center_of_mass(iso_dicom2,
                                                    labeled_volume,
                                                    label_idxs)
    #nyari center of mass GT
    structure = np.ones((3, 3, 3))
    gt_volume, gt_num = ndimage.label(org_mask, structure)
    gt_idxs = np.arange(1, gt_num + 1)
    gravities_gt = ndimage.measurements.center_of_mass(org_mask,
                                                    gt_volume,
                                                    gt_idxs)
    # prediksi
    r = 10
    gravities = [np.append(i, r) for i in gravities]
    alpas = list(np.ones(candidate_num)*0.6)
    colors = [(0.2,0.9,0.2) for c in range(candidate_num) ]
    modes = ['cube' for c in range(candidate_num) ]
    # GT
    all_grafity = gravities  + [np.append(i, r) for i in gravities_gt]
    all_alpas = alpas + list(np.ones(gt_num)*0.6)
    all_colors = colors + [(0.9,0.1,0.1) for c in range(candidate_num) ]
    all_modes = modes + ['sphere' for c in range(candidate_num) ]
    
    
    plot_3d_point(all_grafity, enh,  alpas = all_alpas, colors = all_colors, modes= all_modes, xyz=True, bright=True)
    