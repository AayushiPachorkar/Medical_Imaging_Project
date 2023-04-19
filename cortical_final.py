# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 13:55:32 2022

@author: ishan-gulati and aayushi pachorkar
"""

import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from nilearn import plotting
import pylab as plt
from nilearn import image as nli
from skimage.util import montage 
from skimage.transform import rotate
from nilearn.masking import apply_mask
from skimage import morphology
from skimage import measure
from skimage.segmentation import find_boundaries
from skimage.segmentation import mark_boundaries
from math import sqrt
from skimage.measure import label
from scipy import ndimage

path = "C:\\Users\\pacho\\Downloads\\Medical Imaging\\"
#Loading dataset #orignal image 
data_frame=nib.load(path+"raw_t1_subject_02.nii.gz")
plot_data=data_frame.get_fdata() #Getting nifti data
print(plot_data.shape)  #check shape of the data
plt.imshow(plot_data[:,:,110])   #Plotting Image
plt.title("Original Image")
plt.show()      

#Applying greyscale to the plot data from above
figure, Axis1 = plt.subplots(1, 1, figsize = (15,15)) #Setting Image Size
Axis1.imshow(rotate(montage(plot_data[50:-50,:,:]), 
90, resize=True), cmap ='gray')
plt.title("Applying GrayScale")

sub_img = data_frame
mean_img = nli.mean_img(sub_img)
#Plotting Mean of subject02
plotting.plot_img(mean_img, bg_img=mean_img)
plt.imshow(mean_img.get_fdata()[:,:,110])
plt.title("Plot for Mean of Subject 2")
plt.show()

meaneds = nli.resample_to_img(sub_img, mean_img)
print(meaneds.shape)


plotting.plot_anat(mean_img, title='original nifti ', display_mode='z', dim=-1,cut_coords=[-20, -10, 0, 10, 20, 30])
plotting.plot_anat(meaneds, title='Mean of nifti', display_mode='z', dim=-1, cut_coords=[-20, -10, 0, 10, 20, 30])

for fm in range(1, 12, 5):
    smoothed_img = nli.smooth_img(mean_img, fm)
    plotting.plot_epi(smoothed_img, title="Smooth %imm" % fm,
                     display_mode='z', cmap='viridis')

subimg_Threshold = nli.threshold_img(mean_img, threshold='93%')
plotting.plot_img(subimg_Threshold, bg_img=mean_img)
plt.imshow(subimg_Threshold.get_fdata()[:,:,110])


white_matter = np.array([0]*256**3).reshape(256,256,256)
grey_mat = np.array([0]*256**3).reshape(256,256,256)
white_points= np.array([0]*256**3).reshape(256,256,256)
grey_points= np.array([0]*256**3).reshape(256,256,256)
masked_matter = np.array([0]*256**3).reshape(256,256,256)
for i in np.arange(subimg_Threshold.shape[0]):
    for j in np.arange(subimg_Threshold.shape[1]):
        for k in np.arange(subimg_Threshold.shape[2]):
            if(subimg_Threshold.get_fdata()[i,j,k]>90):
                white_matter[i,j,k]=subimg_Threshold.get_fdata()[i,j,k]
                white_points[i,j,k]=255;




for i in np.arange(subimg_Threshold.shape[0]):
    for j in np.arange(subimg_Threshold.shape[1]):
        for k in np.arange(subimg_Threshold.shape[2]):
            if(subimg_Threshold.get_fdata()[i,j,k]>50 and subimg_Threshold.get_fdata()[i,j,k]<90):
                if(subimg_Threshold.get_fdata()[i+13,j,k]>90 or subimg_Threshold.get_fdata()[i-13,j,k]>90):    
                    grey_mat[i,j,k]=subimg_Threshold.get_fdata()[i,j,k]
                    grey_points[i,j,k]=180;
                elif(subimg_Threshold.get_fdata()[i,j+13,k]>90 or subimg_Threshold.get_fdata()[i,j-13,k]>90):
                    grey_mat[i,j,k]=subimg_Threshold.get_fdata()[i,j,k]
                    grey_points[i,j,k]=180;
                elif(subimg_Threshold.get_fdata()[i+13,j+13,k]>90 or subimg_Threshold.get_fdata()[i-13,j-13,k]>90):
                    grey_mat[i,j,k]=subimg_Threshold.get_fdata()[i,j,k]
                    grey_points[i,j,k]=180;
                elif(subimg_Threshold.get_fdata()[i-13,j+13,k]>90 or subimg_Threshold.get_fdata()[i+13,j-13,k]>90):
                    grey_mat[i,j,k]=subimg_Threshold.get_fdata()[i,j,k]
                    grey_points[i,j,k]=180;
            elif(subimg_Threshold.get_fdata()[i,j,k]>90):
                grey_mat[i,j,k]=subimg_Threshold.get_fdata()[i,j,k]
                grey_points[i,j,k]=180;
                
for i in np.arange(subimg_Threshold.shape[0]):
    for j in np.arange(subimg_Threshold.shape[1]):
        for k in np.arange(subimg_Threshold.shape[2]):
                if(subimg_Threshold.get_fdata()[i,j,k]>50 and subimg_Threshold.get_fdata()[i,j,k]<90):
                    if(subimg_Threshold.get_fdata()[i+4,j,k]<30 and subimg_Threshold.get_fdata()[i-4,j,k]<30):    
                        grey_mat[i,j,k]=0
                        grey_points[i,j,k]=0;
                        
grey_mat=morphology.dilation(grey_mat)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(grey_mat[:,:,110])
plt.title("White and grey image")
plt.subplot(1,2,2)
plt.imshow(white_matter[:,:,110], interpolation='none')
plt.title("White matter")
plt.show()

white_points_d=morphology.dilation(white_points)
grey_points_d=morphology.dilation(grey_points)
grey_seg = grey_mat-white_matter
dd_bound = find_boundaries(white_points, mode='outer').astype(np.uint8)
white_bound = find_boundaries(white_points_d, mode='inner').astype(np.uint8)

grey_bound = find_boundaries(grey_points_d, mode='outer').astype(np.uint8)

for i in np.arange(grey_seg.shape[0]):
    for j in np.arange(grey_seg.shape[1]):
        for k in np.arange(grey_seg.shape[2]):
                if(grey_seg[i,j,k]<40):
                    grey_seg[i,j,k]=0;
    
plt.figure()
plt.subplot(1,2,1)
plt.imshow(grey_seg[:,:,110])
plt.title("Grey matter image")
plt.subplot(1,2,2)
plt.imshow(white_bound[:,:,110], interpolation='none')
plt.imshow(grey_bound[:,:,110],  interpolation='none', alpha=0.7)
plt.title("Grey and White matter outer boundaries")
plt.show()


plt.figure()
plt.subplot(1,2,1)
plt.imshow(white_bound[:,:,110], interpolation='none')
plt.title("White matter boundary")
plt.subplot(1,2,2)
plt.imshow(grey_bound[:,:,110],  interpolation='none')
plt.title("Grey matter outer boundaries")
plt.show()



dd_label = label(grey_seg,connectivity=3)                
dd_label_l = dd_label/10000
plt.figure()
plt.imshow(dd_label[:,:,110])
plt.title("Label image")
plt.show()


wh_b = set()
wh_b_l=[]
for i in np.arange(white_bound.shape[0]):
    for j in np.arange(white_bound.shape[1]):
        for k in np.arange(white_bound.shape[2]):
                if(white_bound[i,j,k]==1):
                    wh_b.add(tuple([i,j,k]))
                    wh_b_l.append([i,j,k])
                    
gr_b = set()
gr_b_l=[]
for i in np.arange(grey_bound.shape[0]):
    for j in np.arange(grey_bound.shape[1]):
        for k in np.arange(grey_bound.shape[2]):
                if(grey_bound[i,j,k]!=0):
                    gr_b.add(tuple([i,j,k])) 
                    gr_b_l.append([i,j,k])
                    
gr_b=gr_b-wh_b
gr_h = list(gr_b)   
wh_h=list(wh_b)


from sklearn.neighbors import KDTree
tree = KDTree(wh_b_l, leaf_size=2)
u=0;
neigh = []
dist_n=[]

for gx in np.arange(grey_seg.shape[0]):
    for gy in np.arange(grey_seg.shape[1]):
        for gz in np.arange(grey_seg.shape[2]):
            if(grey_seg[gx,gy,gz]>0):
                l=[]
                l.append([gx,gy,gz])
                h =[]
                h.append(list(l))
                dist, ind = tree.query(l)
                if(np.min(dist)==0):
                    grey_seg[gx,gy,gz] = 1
                else:
                    grey_seg[gx,gy,gz] = np.min(dist)
  
plt.figure()
plt.imshow(grey_seg[:,:,106],cmap="hot",alpha=1,interpolation='none')
plt.title("Output image with KDTREE")
plt.show()        

affine = np.eye(4)
thickness_KD = nib.Nifti1Image(grey_seg,affine)
nib.save(thickness_KD,path+"thickness_KDTree_subject_02.nii")




grey_dist = ndimage.distance_transform_edt(grey_seg)
plt.figure()
plt.imshow(grey_dist[:,:,110],cmap="magma",alpha=1,interpolation='none')
plt.title("Output image with distance")
plt.show()


affine = np.eye(4)
thickness = nib.Nifti1Image(grey_dist,affine)
nib.save(thickness,path+"thickness_DISTANCE_EDT_subject_02.nii")
      

#Bresenham#D algo to get all points between two 3d points.
"""
def Bresenham3D(x1, y1, z1, x2, y2, z2):
    ListOfPoints = []
    ListOfPoints.append((x1, y1, z1))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if (x2 > x1):
        xs = 1
    else:
        xs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    if (z2 > z1):
        zs = 1
    else:
        zs = -1
  
    # Driving axis is X-axis"
    if (dx >= dy and dx >= dz):        
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while (x1 != x2):
            x1 += xs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))
  
    # Driving axis is Y-axis"
    elif (dy >= dx and dy >= dz):       
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while (y1 != y2):
            y1 += ys
            if (p1 >= 0):
                x1 += xs
                p1 -= 2 * dy
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

for i in wh_h:
    u=u+1
    minD = 0
    l = []
    l.append(list(i))
    dist, ind = tree.query(l)
    #print(np.where(dist==np.min(dist))[1][0])
    c = np.where(dist==np.min(dist))[1][0]
    neigh.append(ind[0][c])
    dist_n.append(np.min(dist))
    
    for i in np.arange(len(wh_h)):
        ListOfPoints=[]
        wh = wh_h[i]
        gy = gr_h[neigh[i]]
        ListOfPoints = Bresenham3D(wh[0], wh[1], wh[2], gy[0], gy[1], gy[2]);
        for j in np.arange(len(ListOfPoints)):
            masked_matter[ListOfPoints[j][0],ListOfPoints[j][1],ListOfPoints[j][2]]=dist_n[i]
        
plt.figure()
plt.subplot(1,2,1)
plt.imshow(masked_matter[:,:,110],'jet')
plt.subplot(1,2,2)
plt.imshow(grey_bound[:,:,110], interpolation='none')
plt.imshow(white_bound[:,:,110], 'jet', interpolation='none', alpha=0.7)
plt.show()
"""          