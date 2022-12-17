# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 15:01:39 2022

@author: vince
"""

import numpy as np
import SimpleITK as sitk
import os
from matplotlib import pyplot as plt


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

dir = "C:/Users/vince/OneDrive/Unibe/Semester_3/MIA_Lab/code_env/MIALab/data/test/117122"
image = sitk.ReadImage(os.path.join(dir, "T1native.nii"), sitk.sitkFloat32)
img = sitk.GetArrayFromImage(image)
imi = sitk.GetImageFromArray(img)
imi .CopyInformation(image)
sample = img[:,:,100]

#normalization as in exercise: 
sample_norm = sample/np.max(sample)

#z-score
mue = np.mean(sample)
sigma = np.std(sample)
sample_z = (sample-mue)/sigma

#histogramm equalization
hist,bins = np.histogram(sample,256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

sample_temp = sample.astype('uint8')
sample_hist, cdf = image_histogram_equalization(sample)


#min-max normalization
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

sample_minmax_1 = scaler.fit_transform(sample) #scheisse?


sample_minmax_2 = (sample-np.min(sample))/(np.max(sample)-np.min(sample))


#logscale normalization

c = 255 / np.log(1 + np.max(image))
log_image = c * (np.log(image + 1))