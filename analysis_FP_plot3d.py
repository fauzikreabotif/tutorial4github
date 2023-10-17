# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:19:06 2023

@author: rizki
"""

import numpy as np
import os
from scipy import ndimage
import skimage.io
import json
from skimage import io
root_function = r'D:\RIZKI\CHIBA\KAMPUS\LUNG'
os.chdir(root_function)
from function_LUNA_LIDC import *
from mayavi import mlab

#-==================== for analysis  =========================================
addrs_out = r'D:\RIZKI\CHIBA\KAMPUS\Bleeding\froc2'    
json_list = [os.path.join(r, ij) for r,f, files in os.walk(addrs_out) for ij in files if 'json' in ij] 
resize_mode = True

for i in range(len(json_list)):
    nm = json_list[i]
    with open(nm, 'r') as f:
        cc=json.load(f) 
    addrs = cc['other']['adress']   
    size_ori = cc['other']['size']
    
    org_volume_n = io.imread(os.path.join(addrs, 'iso_original.mhd'), plugin='simpleitk')
    org = org_volume_n.copy()
    
    if resize_mode:
        org, new_spacing = resample(org_volume_n, old_spacing=[1,1,1], new_spacing=[2,2,2])
        size_ori = np.shape(org)
    
    
    gt = cc['features']['data_gt']
    evals = cc['main_code']['eval']
    nc = np.array(evals['detection_results'])
    nr = np.array(evals['distance_results'])
    nl = np.array(evals['candidate_data'])
    
    area_GT =  [ix[3] for ix in cc['features']['data_gt']]
    area_Pred =  [ix[3] for ix in cc['features']['data_pred']]
    
    
    idxs = []
    FPs = []; TPs = []; FNs = [] ; sgt = 0
    for ij, predi in enumerate(nl):
        if sum(nc[:,0]==ij)==0:
            FPs.append(predi[1:5])
            idxs.append([0,predi[4]])
        if sum(nc[:,0]==ij)!=0:
            TPs.append(predi[1:5]) 
            idxs.append([1,predi[4]])
    
    GT = np.zeros(size_ori).astype(np.uint8)
    Pred_TP = GT.copy(); Pred_FP = GT.copy()
    for ig, gi in enumerate(area_GT):
        if resize_mode:
            gi = np.round(np.array(gi)/2).astype(np.int8)
        else: 
            gi = np.array(gi)
            
        GT[gi[:,0], gi[:,1], gi[:,2]] = 1 
        
    
    #eliminate by confident thresholding
    idxs = np.array(idxs)
    elim = np.where(idxs[:,1] >= 0.9)[0]
    idx = idxs[elim]
    areapred = np.array(area_Pred)[elim]
    
    TP=0; FP=0
    for ip, pi in enumerate(areapred):
        if resize_mode:
            pi = np.round(np.array(pi)/2).astype(np.int8)
        else:
            pi = np.array(pi)
            
        if idx[ip][0]==1:
            Pred_TP[pi[:,0], pi[:,1], pi[:,2]] = 1
            TP+=1
        else:
            Pred_FP[pi[:,0], pi[:,1], pi[:,2]] = 1
            FP+=1
    
    
    

    enh = org.copy()
    enh[enh<-100] = -100
    enh[enh>500] = 500
    enh = streching(enh, [-30, 500], [0,255], 1)

    
    plot_3d([GT, Pred_TP, Pred_FP],enh, alpas = [0.8, 0.8, 0.5], colors = [(0.3,0.5,0.9), (0.3,0.9,0.3), (0.9,0.3,0.3)], xyz=True, bright=True)

    plot_3d_point(all_grafity, enh,  alpas = all_alpas, colors = all_colors, modes= all_modes, xyz=True, bright=True)