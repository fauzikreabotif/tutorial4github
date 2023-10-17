# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:49:09 2023

@author: rizki
"""


import argparse
from datetime import datetime
import numpy as np
import random as rn
import os
from scipy import ndimage
import SimpleITK as sitk
import skimage.io
import torch
from skimage.measure import regionprops
import math
import json

addrs_out = r'C:\Users\rizki\Downloads\froc'
addrs_code = r'\\192.168.10.21\home\ct_bleeding_detection'
os.chdir(addrs_code)
from utils import mhd_io

addrs_code = r'\\192.168.10.21\home\ct_bleeding_detection'

addrs_out = r'C:\Users\rizki\Downloads\froc'

all_result={}
for ij in range(1, 7)  :  
    addrs_list = r'\\192.168.10.21\home\bleeding_data_alanysis\case_list_230516\group_'+str(ij)+'.txt'
    in_path_list_file_name = addrs_list; 
    # load case list
    in_path_list = []
    with open(in_path_list_file_name, 'r') as fp:
        for line in fp:
            line = line.rstrip('\r\n')[16:].replace('/','\\')
            line = os.path.join(r'\\192.168.10.21\\WorkShare', line)
            in_path_list.append(line)
    
    total_tp_num = 0
    for idx, in_path in enumerate(in_path_list):
        print(str(ij)+'-'+str(idx))
        # Get voxel spacing from mhd file
        in_volume_file_name = os.path.join(in_path, "iso_original.mhd")
        in_label_file_name = os.path.join(in_path, "iso_bleeding_label.mhd")
        
        org_label = skimage.io.imread(in_label_file_name, plugin='simpleitk')
        voxel_spacing = mhd_io.get_voxel_spacing_from_mhd(in_volume_file_name)
        
        ans_num = np.max(org_label)
        histogram = ndimage.histogram(org_label, 1, ans_num, ans_num)
        lesion_radius = (np.array(histogram) * voxel_spacing[0] * 3 / (4 * math.pi)) ** (1/3) * 1.5
        
        
        
        
        
        nm = os.path.join(addrs_out,str(ij)+'_'+str(idx)+'.json')
        with open(nm, 'r') as f:
            cc=json.load(f) 
        cc['other']['voxel_spacing'] =  voxel_spacing
        for i in range(ans_num):
            cc['features']['data_gt'][i].append(lesion_radius[i])
        with open(nm, 'w') as f:
            json.dump(cc, f)
    
    