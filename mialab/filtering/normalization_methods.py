# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 15:01:39 2022

@author: vince
"""

import numpy as np
import SimpleITK as sitk
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def no_normalization(image):
    return image

#histogramm equalization
def histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


#normalization as in exercise, z-score: 
def z_score_normalization(image):
    img_out = image        
    mue = np.mean(img_out)
    sigma = np.std(img_out)
    img_out = (img_out-mue)/sigma

    return img_out

#min-max normalization
def min_max_normalization(image):

    #scaler = MinMaxScaler()
    #sample_minmax_1 = scaler.fit_transform(image) #scheisse?
    img_out = (image-np.min(image))/(np.max(image)-np.min(image))

    return img_out

#logscale normalization
def log_normalization(image):
    c = 255 / np.log(1 + np.max(image))
    img_out = c * (np.log(image + 1))
    return img_out

#local testing
#dir = "C:/Users/vince/OneDrive/Unibe/Semester_3/MIA_Lab/code_env/MIALab/data/test/117122"
#image = sitk.ReadImage(os.path.join(dir, "T1native.nii"), sitk.sitkFloat32)
#img = sitk.GetArrayFromImage(image)
#imi = sitk.GetImageFromArray(img)
#imi .CopyInformation(image)
#sample = img[:,:,100]