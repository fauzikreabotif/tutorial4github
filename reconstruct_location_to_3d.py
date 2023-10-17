# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:10:05 2023

@author: rizki
"""


import argparse
import csv
import numpy as np
from scipy import ndimage
import skimage.io

import os
from os.path import join
from skimage import io
import scipy
import torch

from mayavi import mlab


root_function = r'D:\RIZKI\CHIBA\KAMPUS\LUNG'
os.chdir(root_function)
from function_LUNA_LIDC import *

def _get_gravities_from_label_data(label_volume):

    binary_volume = (label_volume > 0).astype(np.uint8)
    label_num = np.max(label_volume)
    label_idxs = np.arange(1, label_num + 1)
    gravities = np.array(ndimage.measurements.center_of_mass(binary_volume,
                                                             label_volume,
                                                             label_idxs))
    gravities = np.round(gravities).astype(np.uint16)

    return gravities


def _get_voi_array_from_padded_volume(padded_volume, pos, voi_size):

    ret = np.zeros([pos.shape[0], voi_size, voi_size, voi_size])

    for n in range(pos.shape[0]):
        ret[n] = np.copy(padded_volume[pos[n][0]:pos[n][0] + voi_size,
                                       pos[n][1]:pos[n][1] + voi_size,
                                       pos[n][2]:pos[n][2] + voi_size])

    return ret


def _create_voi_data_for_each_label(org_volume, org_label, voi_size=64):

    half_width = voi_size // 2
    margin = half_width

    # get garivies of label
    gravities = _get_gravities_from_label_data(org_label)

    # padding
    padding_size = half_width
    pad_width = ((padding_size, padding_size),
                 (padding_size, padding_size),
                 (padding_size, padding_size))
    normalized_vol = np.pad(org_volume,
                            pad_width=pad_width,
                            mode='edge')
    label_volume = np.pad(org_label,
                          pad_width=pad_width,
                          mode='constant',
                          constant_values=0)
    label_volume[label_volume > 0] = 1

    org_voi = _get_voi_array_from_padded_volume(normalized_vol, gravities, voi_size)
    label_voi = _get_voi_array_from_padded_volume(label_volume, gravities, voi_size)

    return org_voi, label_voi


def do_normalization(in_volume, in_range, out_range=[0.0, 1.0]):
    dst_volume = np.copy(in_volume)

    slope = (float(out_range[1]) - float(out_range[0]))\
            / (float(in_range[1]) - float(in_range[0]))

    dst_volume = (dst_volume.astype(np.float64) - float(in_range[0])) * slope \
                 + float(out_range[0])

    dst_volume[dst_volume < out_range[0]] = out_range[0]
    dst_volume[dst_volume > out_range[1]] = out_range[1]

    return dst_volume


def create_voi_data(in_file_list, out_file_name,
                    voi_size=64, in_value_range=[0, 1000]):

    org_voi = np.empty([0, voi_size, voi_size, voi_size])
    label_voi = np.empty([0, voi_size, voi_size, voi_size])

    with open(in_file_list) as fp:
        reader = csv.reader(fp)

        for items in reader:

           org_volume = skimage.io.imread(items[0], plugin='simpleitk')
           org_label = skimage.io.imread(items[1], plugin='simpleitk')

           # normalization (in_value_range => [0,1])
           normalized_volume = do_normalization(org_volume, in_value_range)

           tmp_voi, tmp_label = _create_voi_data_for_each_label(normalized_volume,
                                                                org_label,
                                                                voi_size)
           org_voi = np.concatenate((org_voi, tmp_voi), axis=0)
           label_voi = np.concatenate((label_voi, tmp_label), axis=0)

           del org_volume, org_label, tmp_voi, tmp_label

    np.savez_compressed(out_file_name, data=org_voi, label=label_voi)


def resample(image, old_spacing=[1,1,1], new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, old_spacing)
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = ndimage.zoom(image, real_resize_factor, mode='constant')# order default = 3
    
    return image, new_spacing


def plot_slices(v, direction = 0, range_num = False):
    v = np.moveaxis(v, direction, 0)
    if range_num:
        v = v[range_num]
    z, x, y = np.shape(v)
    w = int(np.ceil(np.sqrt(z))) #widh
    h = int(np.ceil(z/w))  #height
    fig,ax = plt.subplots(h,w,figsize=[w*2,h*2], gridspec_kw={'wspace':0, 'hspace':0.1})
    n = 0; wi = 0 
    while n<z:
        print(int(np.floor(n/w)),wi)
        ax[int(np.floor(n/w)),wi].imshow(v[n])
        ax[int(np.floor(n/w)),wi].set_title(n,fontsize=8)
        ax[int(np.floor(n/w)),wi].axis('off')
        n+=1
        wi = wi+1 if wi<w-1 else 0




voi_size=64
alm_data = r'\\192.168.10.21\WorkShare\TraumaCT_bleeding_230516'
alm_out = r'\\192.168.10.21\home\bleeding_data_results\case_list_230516'
dataall = {}
for i in range(1,7):
    dataall[i] = []
with open(join(alm_data, 'groups_230516.csv'),'r') as data:
   for line in csv.reader(data):
       dataall[int(line[1])].append(line[0])

for i, ni in dataall.items():#group loop
    org_voi = np.empty([0, voi_size, voi_size, voi_size])
    label_voi = np.empty([0, voi_size, voi_size, voi_size])
    for nii in ni: #item loop
        
        org_volume = skimage.io.imread(join(alm_data,nii, 'iso_original.mhd'), plugin='simpleitk')
        org_label = skimage.io.imread(join(alm_data,nii, 'iso_bleeding_label.mhd'), plugin='simpleitk')
        normalized_volume = do_normalization(org_volume, [-50, 300])# [-10, 500] => [0,1] (for bleeding detection)
        tmp_voi, tmp_label = _create_voi_data_for_each_label(normalized_volume,
                                                             org_label,
                                                             voi_size)
        org_voi = np.concatenate((org_voi, tmp_voi), axis=0)
        label_voi = np.concatenate((label_voi, tmp_label), axis=0)
        del org_volume, org_label, tmp_voi, tmp_label
        # plot_slices(tmp_label[2], direction = 0)
    np.savez_compressed(join(alm_out, 'group_'+str(i)+'.npz'), data=org_voi, label=label_voi)   


