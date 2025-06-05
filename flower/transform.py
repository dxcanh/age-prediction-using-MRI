import os
from tkinter import NE
import numpy as np
import scipy.ndimage
import nibabel as nib
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.restoration import denoise_nl_means, denoise_bilateral
from scipy.ndimage import median_filter
import logging
import numpy.polynomial.polynomial as poly
import random

# Transformation Functions
def apply_rician_noise(data, config):
    """Apply Rician noise to the input data using NumPy."""
    if random.random() > config['apply_prob']:
        return data
    std = random.uniform(*config['noise_std_range'])
    noise_real = np.random.normal(0, std, data.shape)
    noise_imag = np.random.normal(0, std, data.shape)
    noisy_data = np.sqrt((data + noise_real)**2 + noise_imag**2)
    return noisy_data

def apply_gaussian_blur(data, original_pixdim, sigma_mm=0.8):
    sigma_voxels = [sigma_mm / dim for dim in original_pixdim]
    return scipy.ndimage.gaussian_filter(data, sigma=sigma_voxels)

def apply_synthetic_bias_field(data, order=3):
    X, Y, Z = data.shape
    x = np.linspace(-1, 1, X)
    y = np.linspace(-1, 1, Y)
    z = np.linspace(-1, 1, Z)
    coeffs_x = np.random.normal(0, 0.05, order + 1)
    coeffs_y = np.random.normal(0, 0.05, order + 1)
    coeffs_z = np.random.normal(0, 0.05, order + 1)
    coeffs_x[0] += 1
    coeffs_y[0] += 1
    coeffs_z[0] += 1
    poly_x = poly.polyval(x, coeffs_x)
    poly_y = poly.polyval(y, coeffs_y)
    poly_z = poly.polyval(z, coeffs_z)
    bias_field = poly_x[:, np.newaxis, np.newaxis] * poly_y[np.newaxis, :, np.newaxis] * poly_z[np.newaxis, np.newaxis, :]
    return data * bias_field

def apply_gamma_correction(image, gamma=0.5, epsilon=1e-8):
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    if (max_val - min_val) < epsilon:
        return image
    normalized = (image - min_val) / (max_val - min_val)
    gamma_corrected = np.power(normalized, gamma)
    return gamma_corrected * (max_val - min_val) + min_val

def resample(image, original_pixdim, new_spacing=[1, 1, 1]):
    resize_factor = np.array(original_pixdim) / np.array(new_spacing)
    return scipy.ndimage.zoom(image, resize_factor, order=3, mode='nearest')

def downsample_to_shape(image, target_shape):
    zoom_factors = np.array(target_shape) / np.array(image.shape)
    return scipy.ndimage.zoom(image, zoom_factors, order=3)