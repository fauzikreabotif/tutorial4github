# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:09:11 2023

@author: rizki
"""


import numpy as np
import os
from scipy import ndimage
import skimage.io
import json
from skimage import io
root_function = r'D:\CodeinGit\bleeding'
os.chdir(root_function)
from function_LUNA_LIDC import *
from mayavi import mlab
from skimage.measure import regionprops, regionprops_table
from matplotlib import pyplot as plt
import json

addrs = r'\\192.168.10.21\home\bleeding_data_alanysis\case_list_NEW_dataset'
addrs_data = r'\\192.168.10.21\WorkShare\TraumaCT_bleeding_230831'

add_expriment = r'\\192.168.10.21\home\bleeding_data_alanysis\RGB_UNet_\feature'
# group = 2
GT = []
for group in range(1,7):
    addrs_prediction = os.path.join(add_expriment, 'test'+ str(group), 'test')
    
    # txt_list = [fls for fls in os.listdir(addrs) if fls.endswith(".txt")]
    # load patient's names in the group
    with open(os.path.join(addrs, 'group_'+str(group)+'.txt')) as f:
         patients_list = [line.rstrip('\n') for line in f]
    patients_list = [os.path.split(detailed_adress)[-1] for detailed_adress in patients_list]
    
    
    # for i in len(patients_list):
    # i=1 
    GT_info =[]
    for i in range(len(patients_list)):
        print('group->'+str(group)+'< '+str(i)+'/'+str(len(patients_list)-1))
        # load GT mask 
        GT_mask = io.imread(os.path.join(addrs_data, patients_list[i], 'iso_bleeding_label.mhd'), plugin='simpleitk')
                  
        # load iso_CT
        org_volume = io.imread(os.path.join(addrs_data, patients_list[i], 'iso_original.mhd'), plugin='simpleitk')
        
        #load predicted mask 
        predicted_mask = io.imread(os.path.join(addrs_prediction, str(i)+'.mhd'), plugin='simpleitk')
        
        
        GT_features = regionprops_table(GT_mask, org_volume, properties=(
                                                                    'centroid',
                                                                    'area', 
                                                                    # 'axis_major_length',
                                                                    # 'axis_minor_length',
                                                                    'intensity_max',
                                                                    'intensity_min',
                                                                    'intensity_mean',
                                                                    # 'solidity',
                                                                    'coords',
                                                                    'bbox',
                                                                    ))
        
        # find the FN which cannot be detected
        gt_detect =[]; gt_feature = {'area': [], 'intensity_mean': [], 'centroid': [], 'bbox': [], 'type':[]}
        for c in range(np.max(GT_mask)):
            pos = GT_features['coords'][c]
            cb=np.unique(np.array([predicted_mask[cr[0], cr[1], cr[2]] for cr in pos]))
            cb = np.setdiff1d(cb,0) # remove label 0
            gt_detect.append(cb)
            
            gt_feature['area'].append(int(GT_features['area'][c]))
            gt_feature['intensity_mean'].append(int(GT_features['intensity_mean'][c]))
            gt_feature['centroid'].append([int(GT_features['centroid-0'][c]), int(GT_features['centroid-1'][c]), int(GT_features['centroid-2'][c])])
            gt_feature['bbox'].append([int(GT_features['bbox-0'][c]), int(GT_features['bbox-1'][c]), int(GT_features['bbox-2'][c]), 
                                    int(GT_features['bbox-3'][c]), int(GT_features['bbox-4'][c]), int(GT_features['bbox-5'][c])])
    
        for ti, tt in enumerate(gt_detect):
            if len(tt)==0:
                gt_feature['type'].append(0)
            elif len(tt)>0:
                tmp = gt_detect.copy(); del tmp[ti]
                prob_num = np.unique(np.array([item for sublist in tmp for item in sublist]))
                ints = np.intersect1d(tt, prob_num)
                if len(ints)==0 and len(tt)==1:
                   gt_feature['type'].append(1) 
                elif len(ints)==0 and len(tt)>1:
                   gt_feature['type'].append(2)
                elif len(ints)>0 and len(tt)==1:
                   gt_feature['type'].append(3)
                elif len(ints)>0 and len(tt)>1:
                   gt_feature['type'].append(4)
    
        
        GT_info.append(gt_feature)

    GT.append(GT_info)  
 
# with open(os.path.join(add_expriment,"GT_features"), "w") as fp:
#     json.dump(GT, fp)



# gtt = [i.astype(np.int) for i in GT_features['coords']]
# with open(os.path.join(add_expriment,"GT_features_all"), "w") as fp:
#     json.dump(gtt, fp)

np.save(os.path.join(add_expriment,"annalysis.npy"), GT)

npz = np.load(os.path.join(add_expriment,"annalysis.npy"), allow_pickle=True)



#load the GT features
with open(os.path.join(add_expriment,"GT_features"), "r") as fp:
    GTs = json.load(fp)    
    
    
    
typ = np.array([ci for cc in GTs[2] for ci in cc['type']])
sum(typ==0)
    

    

    # print('the number of cleared FN:'+str(n_fn))
colors = ['r', 'g', 'c', 'm', 'k']    
for pi in range(len(typ)):
    plt.scatter(pi,1,color = colors[typ[pi]])
    
    plt.scatter(i,1,'o'+colors[type_gts[pi]])



