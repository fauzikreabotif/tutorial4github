# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:51:33 2022

@author: sysai
"""
import os, xmltodict
import numpy as np
from os.path import join
import pydicom as dicom
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import scipy
import csv
from skimage import  morphology
from skimage.measure import label, regionprops
# import scipy.ndimage
from scipy import ndimage
import random
import colorsys
import SimpleITK as sitk
from mayavi import mlab

def streching(img, batas_lama, batas_baru, apha):
    img=(img<batas_lama[0]).choose(img,batas_lama[0])
    img=(img>batas_lama[1]).choose(img,batas_lama[1])
    y=((((img-batas_lama[0])/(batas_lama[1]-batas_lama[0]))**apha)*(batas_baru[1]-batas_baru[0]))+batas_baru[0]
    m=(batas_baru[1]-batas_baru[0])/(y.max()-y.min())
    y1=(m*y)-((m*y.min())+batas_baru[0])
    y1=(y1<batas_baru[0]).choose(y1,batas_baru[0])
    y1=(y1>batas_baru[1]).choose(y1,batas_baru[1])
    return y1

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

def sphere(cubic, radius, position):
    #skrip ini X dan Y nya kebalik 
    
    """Generate an n-dimensional spherical mask."""
    # assume shape and position have the same length and contain ints
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    shape = cubic.shape
    assert len(position) == len(shape)
    # n = len(shape)
    semisizes = (radius,) * len(shape)

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    # arr = np.zeros(shape, dtype=float)
    
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        cubic += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below or equal to 1
    return cubic

def getCTscale(dicomCT):
    image = np.stack([s.pixel_array for s in dicomCT])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    
    intercept = dicomCT[0].RescaleIntercept
    slope = dicomCT[0].RescaleSlope
    if slope == 1:#-> jika iamge di CT scale
        image -= np.int16(intercept)
        image = np.round(image.astype(np.float64)/slope).astype(np.int16) 
    
    ref = np.unique(image[:5,:5,:5])[0]
    image-=ref
    # image[image<0]=0
    return image.astype(np.int16), ref

def stuck3D(dicomCT):
    image = np.stack([s.pixel_array for s in dicomCT]).astype(np.int16)
    ref = np.unique(image[:5,:5,:5])[0]
    return image, ref
    
# =====================   LIDC LOAD DATASET and GT  ===============================
def LIDC_getHU(dicomCT):
    image = np.stack([s.pixel_array for s in dicomCT])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    # image[image == -2000] = 0 # ---> baru tau , ini rule di smua scan LIDC?
    # Convert to Hounsfield units (HU)
    intercept = dicomCT[0].RescaleIntercept
    slope = dicomCT[0].RescaleSlope
    
    if slope != 1:#-> jika iamge di CT scale
        image = np.round(slope * image.astype(np.float64))
        image = image.astype(np.int16)   
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def LIDC_load_CTs (root_ct):
    dcm_files = [dicom.dcmread(join(root_ct,f)) for f in os.listdir(root_ct) if f.endswith('.dcm')]
    insnum = np.array([int(ic.InstanceNumber) for ic in dcm_files])
    argsort = np.argsort(insnum)
    dcm_files = [dcm_files[iu] for iu in argsort] #--> sortten based on instanceNumber
    return (dcm_files)

def LIDC_cekSID(xml_path):
    with open(xml_path, 'r', encoding='utf-8') as file:
        my_xml = file.read()
    gt_ann = {}    
    try:
        my_dict = xmltodict.parse(my_xml)['LidcReadMessage']
        gt_ann['StudyInstanceUID'] = my_dict['ResponseHeader']['StudyInstanceUID'] # --yt dipakai di LIDC
        gt_ann['SeriesInstanceUid'] = my_dict['ResponseHeader']['SeriesInstanceUid'] #--> yg di pakai di LUNA
    except:
        print("Sorry, suspected XML for roentgen. ")
    return gt_ann

def LIDC_xml2dict(xml_path):
    with open(xml_path, 'r', encoding='utf-8') as file:
        my_xml = file.read()
        
    try:
        my_dict = xmltodict.parse(my_xml)['LidcReadMessage']
    except:
        raise Exception("Sorry, suspected XML for roentgen. ")
    
    annos = my_dict['readingSession']
    
    gt_ann = {}
    gt_ann['StudyInstanceUID'] = my_dict['ResponseHeader']['StudyInstanceUID']
    
    i_ann = 1
    for i in range(len(annos)): #iter anotator
        anns = 'annot-'+str(i_ann)
        try:  
            ann_i = annos[i]['unblindedReadNodule']
        except: # means not existing nodules
            continue
        
        if type(ann_i)!=list: #means exsisting only one nodule
            ann_i=[ann_i]
        
        gtnodules ={} ; nod = 0;
        for ii in range(len(ann_i)): #iter noudule
            s_nod = 'nodule-'+str(nod)
            ann_ii= ann_i[ii]['roi']
            
            if type(ann_ii)!=list: #--> eliminate satu slice dan satu titik
                if type(ann_ii['edgeMap'])!=list:
                    continue

            if 'characteristics' in list(ann_i[ii].keys()):
                kelas = (ann_i[ii]['characteristics'])
            else:
                kelas = 'lupa'; print('lupa:')

            if type(ann_ii)!=list: #means the nodule presented only in one slice
                ann_ii=[ann_ii]
            
            location = {'class': kelas, 'posisi':{}}
            for iii in range(len(ann_ii)):#iter slice of nodule
                uid = ann_ii[iii]['imageSOP_UID']
                posZ=[]
                poss = ann_ii[iii]['edgeMap']
                if type(poss)!=list: # rarely, there are miss annotating, indicate a nodule, but the anotator inits by only one position 
                    poss = [poss]
                for pos in poss:
                    posZ.append([int(pos['xCoord']),int(pos['yCoord'])])
                location['posisi'][uid] = posZ
                
                
            gtnodules[s_nod] = location
            nod+=1
            
        if len(gtnodules)>0:  
            gt_ann[anns] = gtnodules
        i_ann+=1
    return gt_ann

def LIDC_extract_mask(gt,dicomCT):
    #gt =>> output of LIDC_xml2dict, dicomCT is the output of LIDC_load_CTs
    #extract 
    slice_ids= [str(i.SOPInstanceUID) for i in dicomCT]
    xy = dicomCT[0].Columns
    
    ens = []
    anns = list(gt.keys())
    anns.remove('StudyInstanceUID')
    masking = {}
    for ann in anns:
        for nod in gt[ann]:
            poss = gt[ann][nod]['posisi']
            for ip in poss:
                n_slic = slice_ids.index(ip); ens.append(n_slic) # --> index
                n_slic = str(dicomCT[n_slic].InstanceNumber) #--> string
                poly=[]
                for pl in poss[ip]:
                    poly+=pl
                img = Image.new('L', (xy, xy), 0)
                try: #coba krn ada hanya cuma 1 titik, jd tidak bisa di polygonkan 
                    ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
                    mask = np.array(img)
                except:
                    # print(ann+'*'+n_slic)
                    continue
                
                if not n_slic in masking: # check n-slice is exis
                    masking[n_slic] = {} 

                masking[n_slic][ann] = mask
                
    #hasil maskingm, berupa dict yg keys nya InstanceNumber mulai dari 1. sehingga kurangi 1 untuk mendapat slice ke. 
    return masking

def LIDC_mask2im(extractedMask):
    #extractedMask=> output of LIDC_extract_mask
    masking = extractedMask
    masks = {}
    for j in masking:
        mask = []
        for i in masking[j]:
            ms = masking[j][i]
            if len(mask) == 0:
                mask = np.zeros(np.shape(ms), dtype='uint8')
            mask +=  ms.astype(np.uint8)
        if masks.get(str(j)) == None:  
            masks[str(j)] = mask
    return masks



# =====================  LUNA LOAD DATASET and GT  =================================

def LUNA_stuftITK(al_luna):
    itkimage = sitk.ReadImage(al_luna)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    # numpyOrigin = np.array(list(itkimage.GetOrigin())) # X,Y,Z
    # numpySpacing = np.array(list(itkimage.GetSpacing())) # X,Y,Z
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    # numpyOrigin = np.roll(numpyOrigin,1) # Z,X,Y
    # numpySpacing = np.roll(numpySpacing,1)# Z,X,Y
    return numpyImage, numpyOrigin, numpySpacing


def LUNA_readCSV(filename):
    lines = []
    with open(filename, "rt", encoding="ascii") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def LUNAlistCT(root_luna):
    luna = []
    for root, dirnames, filenames in os.walk(root_luna):
        if 'subset' in root:
            if len(filenames)>10:
                luna.extend([join(root, f) for f in filenames if f.endswith('.mhd')])
    return luna

def LUNA_extractGT(filename):
    gts = LUNA_readCSV(filename)[1:] 
    gt_all = {}
    for c in gts:
        # _--> tulisan di .xls nya ids, cox, coy, coz, tp jika di proyeksikan ke image -->  structure: [IDS, coY, coX, coZ, diameter]
        # jd perlu di rubah jadi ->> [coZ, coX, coY]
        im_proyeksi = [c[3],c[2],c[1], c[4]]
        if not c[0] in gt_all.keys():
            gt_all[c[0]] = [im_proyeksi]
        else:
            gt_all[c[0]].append(im_proyeksi)
    return gt_all

def LUNA_worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

# ====================  LUNG SEGMENTATION   ===================================
def findTrachea (CTHUslice):
    th = [-400]
    v=CTHUslice>th[0]; 
    filll = scipy.ndimage.binary_fill_holes(v).astype(int)-v
    filll =  morphology.remove_small_objects(filll.astype(np.bool8), min_size=100)
    filll, n = label(filll, return_num=True)
    pro = regionprops(filll)
    sele = []
    for ni, ip in enumerate(pro):
        if ip['centroid'][1]>200 and ip['centroid'][1]<300 : #sbx
            if ip['centroid'][0]>200 and ip['centroid'][0]<350 : #sby
                if ip['eccentricity']>0.5:
                    # print(ip['eccentricity'])
                    sele.append(ni+1)
    return filll, sele, pro

def findApex(CT3D):
    ii = 0; b = 20; ke = []
    while ii < b:
        seg, ns, pro = findTrachea(CT3D[ii,:,:])
        if len(ns)==0:
            ii+=1
            continue
        ape = [j['area'] for j in pro]
        ke_trac = ns[0] 
        if len(ns)>0 and len(pro)>1 and max(ape)>5*ape[ke_trac-1]: #0-> ambil kandidate pertama. -1 -> krn intensitas,rubah keurut. 
            ke = ii; ii = b+1
        else:
            ii+=1
    return ke, seg, pro, ke_trac 

#===========================  Patches extraction and reconstruction  ======================================

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


def recons_patches_v2(patches, patch_length, notes): # this version, let the function to reconstruct the binary patches
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

#================= Ploting ===================
def plothistoHU(CTimage, minmaxCT=0):
    if minmaxCT!=0:
        CTimage[CTimage>CTimage]
    plt.hist(CTimage.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()
    # Show some slice in the middle
    # plt.imshow(CTimage, cmap=plt.cm.gray)
    # plt.show()

def plot_slices(v, direction = 0, range_num = False):
    #v = volumetric; direction = the axis that yuo want to be flated, range_num = the slice numbers that you prefered. 
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
        ax[int(np.floor(n/w)),wi].imshow(v[n], cmap='gray')
        ax[int(np.floor(n/w)),wi].set_title(n,fontsize=8)
        ax[int(np.floor(n/w)),wi].axis('off')
        n+=1
        wi = wi+1 if wi<w-1 else 0
    
        
#=================  Ploting 3D   ==================================
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.6
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def plot_3d(im_Label,CTs, alpas = [0.8], colors = [(0.5,0.5,0.9)], xyz=False, bright=True):
    if not type(im_Label) is list:
        Exception("Sorry, input should in list type ")
    p = im_Label[0]
    x,y,z = p.shape
    n = len(im_Label)
    if n!=len(alpas):
        alpas = np.linspace(0, 1, n+1) [1:]
    if n!=len(colors):
        colors = random_colors(n, bright)
    mlab.figure(bgcolor=(0, 0, 0), size=(500, 500))
    if xyz == True :
        # alpa_xyz = 1
        # imx = np.zeros((x,y,z), dtype = bool)
        # imx[int(round(x/2)), :,:] = True
        # imy = np.zeros((x,y,z), dtype = bool)
        # imy[: , int(round(y/2)),:] = True
        # imz = np.zeros((x,y,z), dtype = bool)
        # imz[:, :,int(round(z/2))] = True
        # mlab.contour3d(imz*1, color = (0.8,0.8,0.2),opacity =  alpa_xyz)
        # mlab.contour3d(imx*1, color = (0.8,0.3,0.8),opacity =  alpa_xyz)
        # mlab.contour3d(imy*1, color = (0.3,0.8,0.9),opacity =  alpa_xyz)
        mlab.volume_slice(CTs, plane_orientation='x_axes', slice_index=round(x/2), colormap='gray')
        mlab.volume_slice(CTs, plane_orientation='y_axes', slice_index=round(y/2), colormap='gray')
        mlab.volume_slice(CTs, plane_orientation='z_axes', slice_index=round(z/2), colormap='gray')
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    
    mlab.zlabel('Z'); mlab.xlabel('X'); mlab.ylabel('Y')
    for i in range(n):
        p = im_Label[i]
        mlab.contour3d(p, color = colors[i], opacity =  alpas[i])

def plot_3d_point(points, CTs,  alpas = [0.8], colors = [(0.5,0.5,0.9)], modes = ['cube'], xyz=False, bright=True):
    # points = [[x,y,z,r]]
    if not type(points) is list:
        Exception("Sorry, input should in list type ")
    x,y,z = CTs.shape
    n = len(points)
    if n!=len(alpas):
        alpas = np.linspace(0, 1, n+1) [1:]
    if n!=len(colors):
        colors = random_colors(n, bright)
    mlab.figure(bgcolor=(0, 0, 0), size=(500, 500))
    if xyz == True :
        mlab.volume_slice(CTs, plane_orientation='x_axes', slice_index=round(x/2), colormap='gray')
        mlab.volume_slice(CTs, plane_orientation='y_axes', slice_index=round(y/2), colormap='gray')
        mlab.volume_slice(CTs, plane_orientation='z_axes', slice_index=round(z/2), colormap='gray')
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    
    mlab.zlabel('Z'); mlab.xlabel('X'); mlab.ylabel('Y')
    for i in range(n):
        x, y, z, r = points[i]
        mlab.points3d(x, y, z, r, scale_factor=1, mode = modes[i] , opacity =  alpas[i], color = colors[i])
        
# ================  3D sliced multi orientations  ============================              
def rotxyz(CT):
    p = CT.transpose(2,1,0)
    return p[:,:,::-1]# z,x,y-> x,y,z

def slice45_v2(iso_dicom, ddd = [1,0,1]):
    alpha = 0.2
    ddd = np.array([1,0,1])
    sn = np.array(iso_dicom.shape)
    #potong slice ke sbz
    #mulai 10%z 
    if len(np.where(ddd==0))!=1:
        raise Exception("Sorry, you should choose 1 as a rotary axis ")
    else:
        dr = np.where(ddd!=0)[0] #--> find planes -> m,n
        ndr= np.where(ddd==0)[0]#-> mn
    #--> m,n simbol x,y,z
    m = sn[dr[0]]; n = sn[dr[1]]; mn = sn[ndr[0]]
    md = ddd[dr[0]];   nd= ddd[dr[1]]; #md and nd ->> direction of m and n

    d =  round((m+n)*alpha/2) # berhenti 2 sisi , sisi awal dan akhir
    puter = list(range(m+n))[d:-d]

    tm = list(range(m)) if md == 1 else list(reversed(range(m)))
    tn = list(range(n)) if nd == 1 else list(reversed(range(n)))

    tm1 = tm+list(np.ones(n, dtype=int)*tm[-1])
    tm2 = list(np.ones(n, dtype=int)*tm[0])+tm

    tn1 = list(np.ones(m, dtype=int)*tn[0])+tn
    tn2 = tn+list(np.ones(m, dtype=int)*tn[-1])
    slices = []
    for p in puter:
        vm = 1 if tm2[p]-tm1[p]>0 else -1
        vn = 1 if tn2[p]-tn1[p]>0 else -1
        
        mm = list(range(tm1[p], tm2[p]+vm, vm))
        nn = list(range(tn1[p], tn2[p]+vn, vn))
        mmnn = list(range(mn))
         
        #panjang mm dan nn harusnya sama
        mmnn = [list(range(sn[ndr[0]])) for ln in range(len(nn))]
        mnj = [[],[],[]]
        mnj[dr[0]] = mm; mnj[dr[1]] = nn; mnj[ndr[0]] = mmnn;
        
        imi = [iso_dicom[j,jj,jjj]  for j,jj,jjj in zip(mnj[0], mnj[1], mnj[2])]
        imi = np.array(imi)
        # imi, _ = resample(imi, old_spacing=[np.sqrt(spacingV[dr[0]]+spacingV[dr[1]]),spacingV[ndr[0]]], new_spacing=[new_spacingV[ndr[0]],new_spacingV[ndr[0]]])
        slices.append({'array':imi, 'meta': mnj})
    return slices

def returnSlice45_v2(slices, iso_dicom):
    boxbis = (iso_dicom*0)+iso_dicom[0,0,0]
    for i in slices:
        arr = i['array']
        meta = i['meta']
        ke = 0
        for j,jj,jjj in zip(meta[0], meta[1], meta[2]):
            boxbis[j,jj,jjj] = arr[ke,:]
            ke+=1
    return boxbis

def gt_boxes(gt_slices, alpa_sampling = [1,1]): #alpa_sampling->> old_spacing/new_spacing
    boxes = []
    for gt in gt_slices:
        if np.max(gt)==0:
            boxes.append([])
            continue
        c_gt= label(gt)
        props = regionprops(c_gt)
        bbx = []
        for pi in props:
            minr, minc, maxr, maxc = pi.bbox #https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html
            minr, minc, maxr, maxc = minr*alpa_sampling[0], minc*alpa_sampling[1], maxr*alpa_sampling[0], maxc*alpa_sampling[1]
            bbx.append([int(round(minr)), int(round(minc)), int(round(maxr)), int(round(maxc))])
        boxes.append(bbx)
    return boxes

def slice45 (CT3D, ddd = [-1,0,1], spacingV = [1,1,1], new_spacingV = [1,1,1],  alpha = 0.1, plott = True):
    #sudah di uji coba, dan ok smua
    ddd = np.array(ddd)
    L3D = CT3D.copy()
    boxx = np.zeros(L3D.shape)
    s = np.array(L3D.shape)-1
    
    #potong slice ke sbz
    #mulai 10%z 
    if len(np.where(ddd==0))!=1:
        raise Exception("Sorry, you should choose 1 as a rotary axis ")
    else:
        dr = np.where(ddd!=0)[0] #--> find planes
        ndr= np.where(ddd==0)[0]
    #--> m,n simbol x,y,z
    m = s[dr[0]]; n=  s[dr[1]];
    md = ddd[dr[0]];   nd= ddd[dr[1]]; #md and nd ->> direction of m and n

    d =  round((m+n)*alpha/2) # berhenti 2 sisi , sisi awal dan akhir

    N = list(np.arange(d, n+m-d, 1))
    nh = N[round(len(N)/2)]
    
    m0 = 0 if md == 1 else m
    n0 = 0 if nd == 1 else n
    
    all3D = []; coll = []
    for d in N:
        mi = d if md == 1 else m-d
        ni = d if nd == 1 else n-d
        if mi<m and mi>0:
            mi+=md
        else:
            if md == -1:
                mi = 0
            else:
                mi = m
            n0+=nd        
        if ni<n and ni>0:
            ni+=nd
        else:
            if nd == -1:
                ni = 0
            else:
                ni = n   
            m0+=md
        coll.append([m0,mi,n0,ni])
        mm = list(range(m0, mi, md))
        nn = list(range(n0, ni, nd))
        #panjang mm dan nn harusnya sama
        mmnn = [list(range(s[ndr[0]])) for ln in range(len(nn))]
        mnj = [[],[],[]]
        mnj[dr[0]] = mm; mnj[dr[1]] = nn; mnj[ndr[0]] = mmnn;
        
        imi = [L3D[j,jj,jjj]  for j,jj,jjj in zip(mnj[0], mnj[1], mnj[2])]
        imi = np.array(imi)
        imi, _ = resample(imi, old_spacing=[np.sqrt(spacingV[dr[0]]+spacingV[dr[1]]),spacingV[ndr[0]]], new_spacing=[new_spacingV[ndr[0]],new_spacingV[ndr[0]]])
        # io.imshow(imi)
        # plt.show()
        all3D.append(imi)
        # d += 1
        if plott :
            if d==nh:
                for j,jj,jjj in zip(mnj[0], mnj[1], mnj[2]):
                    boxx[j,jj,jjj]=1
    if plott :
        enh = CT3D.copy()
        enh[enh<-400] = -400
        enh[enh>700] = 700
        enh = streching(enh, [-400, 700], [0,255], 1)
        mlab.figure(bgcolor=(0, 0, 0), size=(500, 500))
        mlab.volume_slice(enh, plane_orientation='x_axes', slice_index=round(s[0]/2), colormap='gray')
        mlab.volume_slice(enh, plane_orientation='y_axes', slice_index=round(s[1]/2), colormap='gray')
        mlab.volume_slice(enh, plane_orientation='z_axes', slice_index=round(s[2]/2), colormap='gray')
        mlab.contour3d(boxx, color = (0.8,0.8,0.2),opacity =  0.3)
        mlab.zlabel('Z'); mlab.xlabel('X'); mlab.ylabel('Y')
    
    details= {'shape': L3D.shape, 'directon': ddd, 'space' : spacingV, 'new_space' : new_spacingV, 'alpa': alpha}
    return all3D,details