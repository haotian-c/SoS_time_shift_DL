
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import glob
import cv2
from skimage import color
import pandas as pd
#from numba import cuda 
from PIL import Image , ImageFilter 

from scipy.signal import hilbert    


import torch.nn as nn 
import torch.nn.functional as F
import random
import scipy.io
import zipfile

import torch

# print('if available',torch.cuda.is_available())
import math



import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as compare_ssim


###########################################
###  Evaluation metrics
############################################
def calculate_recon_metrics(recon_img,gt_img):

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(gt_img, recon_img)
    # Calculate Root Mean Squared Error (RMSE)
    
    rmse = np.sqrt(mse)
    rmse = round(rmse,2)
    print("Root Mean Squared Error (RMSE):", rmse)
    
    
    mae = np.mean(np.abs(gt_img - recon_img))
    mae = round(mae,2)
    print("Mean Absolute Error (MAE):", mae)
    
    ssim_score = compare_ssim(gt_img.astype('int'), recon_img.astype('int'),data_range=200)
    ssim_score = round(ssim_score,3)
    print("Structural Similarity Index (SSIM):", ssim_score)
    

############################################
####    Create Gaussian mask
###############################################
def create_rand_gaussian_mask(image_size, sigma):
    """
    Create a 2D Gaussian mask.

    Parameters:
    - image_size: Tuple (height, width) specifying the dimensions of the mask.
    - sigma: Standard deviation of the Gaussian distribution.

    Returns:
    - Gaussian mask.
    """
    x, y = np.meshgrid(np.arange(0, image_size[1]), np.arange(0, image_size[0]))
    x_center = (image_size[1] - 1) / 2 + random.randint(-image_size[1]//3, image_size[1]//3)
    y_center = (image_size[0] - 1) / 2 + random.randint(-image_size[0]//3, image_size[0]//3)
    
    # Calculate Gaussian mask
    gaussian_mask = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma ** 2))
    gaussian_mask /= 2 * np.pi * sigma ** 2
    return gaussian_mask

    
    
    
############################################
###   Remove margins on the blind zones
##############################################
def remove_blind_region_psf0(time_lag):
    height = len(time_lag)
    width = len(time_lag[0])

    for height_i in range(height):
        width_i = int(np.floor(height_i*np.tan(15/180*np.pi)))
        for width_i_curr in range(width_i+1):
#             print(height_i,width_i_curr)
            time_lag[height_i][width_i_curr] = 0
            time_lag[height_i][width-1-width_i_curr] =0
        
    return time_lag

def remove_blind_region_psf7p5(time_lag):
    height = len(time_lag)
    width = len(time_lag[0])

    for height_i in range(height):
        width_i = int(np.floor(height_i*np.tan(15/180*np.pi)))
        for width_i_curr in range(width_i+1):
#             print(height_i,width_i_curr)
            time_lag[height_i][width_i_curr] = 0
#             time_lag[height_i][width-1-width_i_curr] =0
        
    return time_lag


def remove_blind_region_minuspsf7p5(time_lag):
    height = len(time_lag)
    width = len(time_lag[0])

    for height_i in range(height):
        width_i = int(np.floor(height_i*np.tan(15/180*np.pi)))
        for width_i_curr in range(width_i+1):
#             print(height_i,width_i_curr)
#             time_lag[height_i][width_i_curr] = 0
            time_lag[height_i][width-1-width_i_curr] =0
        
    return time_lag


##########################################
####   Total variation loss
#########################################
def calculate_total_variation_loss(img, weight_h,weight_w):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = (torch.abs(img[:,:,1:,:]-img[:,:,:-1,:])).sum()
     tv_w = (torch.abs(img[:,:,:,1:]-img[:,:,:,:-1])).sum()
     return (weight_h*tv_h+weight_w*tv_w)/(bs_img*c_img*h_img*w_img)

##########################################
####   set requires grad
#########################################

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


############################################
### project from SoS map to time shit , differentiable
############################################
mask_foi = torch.ones((73, 89))
class project_to_time_lag_0psf(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_model_origin = torch.tensor(scipy.io.loadmat(
            ('./physics_forward_model/d11_11_psf0_forward_modelmatrix.mat'))['L'],dtype=torch.float)
        self.linear_forward_model = nn.Linear(len(self.forward_model_origin), len(self.forward_model_origin), bias=False)
        with torch.no_grad():
            self.linear_forward_model.weight = torch.nn.Parameter(torch.tensor(self.forward_model_origin))

    def forward(self,SlownessRelat):
        flattened_slowness =  torch.flatten(SlownessRelat.transpose(2, 3), start_dim=2)
        projected_timelag = self.linear_forward_model(flattened_slowness)
        product = projected_timelag.view(len(SlownessRelat),1,89,73).transpose(2, 3)
        product = torch.mul(product,mask_foi.to(product.device))
        return product
    
class project_to_time_lag_7p5psf(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_model_origin = torch.tensor(scipy.io.loadmat(
            ('./physics_forward_model/d11_11_7p5_forward_modelmatrix.mat'))['L'],dtype=torch.float)
        self.linear_forward_model = nn.Linear(len(self.forward_model_origin), len(self.forward_model_origin), bias=False)
        with torch.no_grad():
            self.linear_forward_model.weight = torch.nn.Parameter(torch.tensor(self.forward_model_origin))

    def forward(self,SlownessRelat):
        flattened_slowness =  torch.flatten(SlownessRelat.transpose(2, 3), start_dim=2)
        projected_timelag = self.linear_forward_model(flattened_slowness)
        product = projected_timelag.view(len(SlownessRelat),1,89,73).transpose(2, 3)
        product = torch.mul(product,mask_foi.to(product.device))
        return product
    
class project_to_time_lag_minus7p5psf(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_model_origin = torch.tensor(scipy.io.loadmat(
            ('./physics_forward_model/d11_11_minus7p5_forward_modelmatrix.mat'))['L'],dtype=torch.float)
        self.linear_forward_model = nn.Linear(len(self.forward_model_origin), len(self.forward_model_origin), bias=False)
        with torch.no_grad():
            self.linear_forward_model.weight = torch.nn.Parameter(torch.tensor(self.forward_model_origin))

    def forward(self,SlownessRelat):
        flattened_slowness =  torch.flatten(SlownessRelat.transpose(2, 3), start_dim=2)
        projected_timelag = self.linear_forward_model(flattened_slowness)
        product = projected_timelag.view(len(SlownessRelat),1,89,73).transpose(2, 3)
        product = torch.mul(product,mask_foi.to(product.device))
        return product
    
    
    
###############################
### Convert SoS to time-shift map
################################


forward_model_transform_matrix_7p5psf = scipy.io.loadmat(('./physics_forward_model/d11_11_7p5_forward_modelmatrix.mat'))['L']
def sos2timelag_7p5psf(sos_matrix): ## Input is a SoS matrix
    return np.matmul(forward_model_transform_matrix_7p5psf,1/sos_matrix.flatten('F')-1/1540).reshape((89,73)).T

forward_model_transform_matrix_0psf = scipy.io.loadmat(('./physics_forward_model/d11_11_psf0_forward_modelmatrix.mat'))['L']
def sos2timelag_0psf(sos_matrix): ## Input is a SoS matrix
    return np.matmul(forward_model_transform_matrix_0psf,1/sos_matrix.flatten('F')-1/1540).reshape((89,73)).T


forward_model_transform_matrix_minus7p5psf = scipy.io.loadmat(('./physics_forward_model/d11_11_minus7p5_forward_modelmatrix.mat'))['L']
def sos2timelag_minus7p5psf(sos_matrix): ## Input is a SoS matrix
    return np.matmul(forward_model_transform_matrix_minus7p5psf,1/sos_matrix.flatten('F')-1/1540).reshape((89,73)).T
    
    
    
    
    
    
def rf_to_bmode(rf_data, dynamic_range=40):
    """
    Convert beamformed RF data to a B-mode ultrasound image.
    
    Parameters:
        rf_data (ndarray): 2D numpy array of beamformed RF data (depth × lateral).
        dynamic_range (int): Dynamic range in dB for log compression.
    
    Returns:
        bmode_image (ndarray): Processed B-mode ultrasound image.
    """

    # 1. Envelope detection using Hilbert transform
    analytic_signal = hilbert(rf_data, axis=0)  # Apply along depth
    envelope = np.abs(analytic_signal)

    # 2. Log compression (convert envelope to dB scale)
    envelope = envelope / np.max(envelope)  # Normalize
    bmode_image = 20 * np.log10(envelope + 1e-6)  # Convert to dB scale

    # 3. Normalize for display (0 to 255 grayscale)
    bmode_image = np.clip((bmode_image + dynamic_range) / dynamic_range, 0, 1)

    return (bmode_image * dynamic_range - dynamic_range)

    
    
    
    
downsample_size = 11
lateral_length = 51
axial_length = 73
def extract_center_columns(array,width=axial_length):
    """
    Extracts the center (height × width) region from a 2D NumPy array.

    Parameters:
        array (ndarray): Input 2D NumPy array.
        height (int): Number of rows to preserve.
        width (int): Number of columns to preserve.

    Returns:
        ndarray: Extracted center region.
    """

    rows, cols = array.shape  # Get original dimensions

    # Compute center indices

    col_start = (cols - width) // 2
    col_end = col_start + width

    # Extract center region
    center_region = array[:, col_start:col_end]

    return center_region
    