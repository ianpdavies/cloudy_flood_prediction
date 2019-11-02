from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
import rasterio
import random
from noise import snoise3
from math import sqrt
import time

# ======================================================================================================================

def cloud_generator_small(img, data_path, seed=None, octaves=10, overwrite=False, alt=None):
    """
    *Creates smaller clouds than the other generator.*
    Creates a random cloud image using Simplex noise, and saves that as a numpy binary file.
    The cloud image can then be thresholded by varying % cloud cover to create a binary cloud mask.
    See example.

    Reqs: random, rasterio, os, snoise3 from noise, numpy, sqrt from math

    Parameters
    ----------
    seed : int
        If no seed provided, a random integer between 1-10000 is used.
    data_path :str
        pathlib.Path pointing to data directory
    img : str
        Image name that will be used to name cloudmask
    overwrite: true/false
        Whether existing cloud image should be overwritten

    Returns
    ----------
    clouds : array
        Cloud image as a numpy array; also saved as a numpy binary file

    Example
    ----------
    data_path = Path('C:/Users/ipdavies/CPR/data')
    img = '4337_LC08_026038_20160325_1'

    myClouds = cloudGenerator(img = img, path = path)

    # Create cloudmask with 90% cloud cover
    cloudmask_20 = myClouds < np.percentile(clouds, 90)
    """

    stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'
    file_name = img + '_clouds.npy'
    cloud_dir = data_path / 'clouds' / 'small'
    cloud_file = cloud_dir / file_name

    if overwrite:
        try:
            cloud_file.unlink()
            print('Removing existing cloud image for ' + img + ' and creating new one')
        except FileNotFoundError:
            print('No existing cloud image for ' + img + '. Creating new one')
    if not overwrite:
        if cloud_file.exists():
            print('Cloud image already exists for ' + img)
            return
        else:
            print('No cloud image for ' + img + ', creating one')

    # Make directory for clouds if none exists
    if cloud_dir.is_dir() == False:
        cloud_dir.mkdir()
        print('Creating cloud imagery directory')

    if seed is None:
        seed = (random.randint(1, 10000))

    # Get shape of input image to be masked
    with rasterio.open(stack_path) as ds:
        shape = ds.shape

    # Create empty array of zeros to generate clouds on
    clouds = np.zeros(shape)
    freq = np.ceil(sqrt(np.sum(shape))) * 2  # Frequency calculated based on shape of image

    # Generate 2D (technically 3D, but uses a scalar for z) simplex noise
    for y in range(shape[1]):
        for x in range(shape[0]):
            clouds[x, y] = snoise3(x / freq, y / freq, seed, octaves)

    # Save cloud file as
    np.save(cloud_file, clouds)

    # Return clouds
    return clouds


# ======================================================================================================================


def preprocessing_small_clouds(data_path, img, pctl, gaps, normalize=True):
    """
    *This preprocessing uses randomly generated clouds
    Masks stacked image with cloudmask by converting cloudy values to NaN

    Parameters
    ----------
    data_path : str
        Path to image folder
    img : str
        Name of image file (without file extension)
    pctl : list of int
        List of integers of cloud cover percentages to mask image with (10, 20, 30, etc.)
    gaps : bool
        Preprocessing cloud gaps or clouds? Determines how to mask out image
    normalize : bool
        Whether to normalize data or not. Default is true.

    Returns
    ----------
    data : array
        3D array identical to input stacked image but with cloudy pixels removed
    data_vector : array
        2D array of data, standardized, with NaNs removed
    data_ind : tuple
        Tuple of row/col indices in 'data' where cloudy pixels/cloud gaps were masked. Used for reconstructing the image later.
    """

    img_path = data_path / 'images' / img
    stack_path = img_path / 'stack' / 'stack.tif'

    # Get local image
    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)

    # load cloudmasks
    cloudmask_dir = data_path / 'clouds' / 'small'

    cloudmask = np.load(cloudmask_dir / '{0}'.format(img+'_clouds.npy'))

    # Check for any features that have all zeros/same value and remove. This only matters with the training data
    cloudmask = cloudmask < np.percentile(cloudmask, pctl)
    data_check = data.copy()
    data_check[cloudmask] = -999999
    data_check[data_check == -999999] = np.nan
    data_check[np.isneginf(data_check)] = np.nan
    data_check_vector = data_check.reshape([data_check.shape[0] * data_check.shape[1], data_check.shape[2]])
    data_check_vector = data_check_vector[~np.isnan(data_check_vector).any(axis=1)]
    data_std = data_check_vector[:, 0:data_check_vector.shape[1] - 1].std(0)

    # Just adding this next line in to correctly remove the deleted feat from feat_list_new during training
    # Should remove once I've decided whether to train with or without perm water
    feat_keep = [a for a in range(data.shape[2])]
    if 0 in data_std.tolist():
        zero_feat = data_std.tolist().index(0)
        data = np.delete(data, zero_feat, axis=2)
        feat_keep.pop(zero_feat)

    if gaps:
        cloudmask = cloudmask < np.percentile(cloudmask, pctl)
    if not gaps:
        cloudmask = cloudmask > np.percentile(cloudmask, pctl)

    # Convert -999999 and -Inf to Nans
    data[cloudmask] = -999999
    data[data == -999999] = np.nan
    data[np.isneginf(data)] = np.nan

    # Get indices of non-nan values. These are the indices of the original image array
    data_ind = np.where(~np.isnan(data[:, :, 1]))

    # Reshape into a 2D array, where rows = pixels and cols = features
    data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
    shape = data_vector.shape

    # Remove NaNs
    data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]

    data_mean = data_vector[:, 0:shape[1] - 1].mean(0)
    data_std = data_vector[:, 0:shape[1] - 1].std(0)

    # Normalize data - only the non-binary variables
    if normalize:
        data_vector[:, 0:shape[1]-1] = (data_vector[:, 0:shape[1]-1] - data_mean) / data_std

    return data, data_vector, data_ind, feat_keep


# ======================================================================================================================


def preprocessing_rand_clouds(data_path, img, pctl, trial, gaps, normalize=True):
    """
    *This preprocessing uses randomly generated clouds
    Masks stacked image with cloudmask by converting cloudy values to NaN

    Parameters
    ----------
    data_path : str
        Path to image folder
    img : str
        Name of image file (without file extension)
    pctl : list of int
        List of integers of cloud cover percentages to mask image with (10, 20, 30, etc.)
    gaps : bool
        Preprocessing cloud gaps or clouds? Determines how to mask out image
    trial : int
        Trial number for random cloud testing
    normalize : bool
        Whether to normalize data or not. Default is true.

    Returns
    ----------
    data : array
        3D array identical to input stacked image but with cloudy pixels removed
    data_vector : array
        2D array of data, standardized, with NaNs removed
    data_ind : tuple
        Tuple of row/col indices in 'data' where cloudy pixels/cloud gaps were masked. Used for reconstructing the image later.
    """

    img_path = data_path / 'images' / img
    stack_path = img_path / 'stack' / 'stack.tif'

    # Get local image
    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)

    # load cloudmasks
    cloudmask_dir = data_path / 'clouds' / '{}'.format('random_' + str(trial))

    cloudmask = np.load(cloudmask_dir / '{0}'.format(img+'_clouds.npy'))

    # Check for any features that have all zeros/same value and remove. This only matters with the training data
    cloudmask = cloudmask < np.percentile(cloudmask, pctl)
    data_check = data.copy()
    data_check[cloudmask] = -999999
    data_check[data_check == -999999] = np.nan
    data_check[np.isneginf(data_check)] = np.nan
    data_check_vector = data_check.reshape([data_check.shape[0] * data_check.shape[1], data_check.shape[2]])
    data_check_vector = data_check_vector[~np.isnan(data_check_vector).any(axis=1)]
    data_std = data_check_vector[:, 0:data_check_vector.shape[1] - 1].std(0)

    # Just adding this next line in to correctly remove the deleted feat from feat_list_new during training
    # Should remove once I've decided whether to train with or without perm water
    feat_keep = [a for a in range(data.shape[2])]
    if 0 in data_std.tolist():
        zero_feat = data_std.tolist().index(0)
        data = np.delete(data, zero_feat, axis=2)
        feat_keep.pop(zero_feat)

    if gaps:
        cloudmask = cloudmask < np.percentile(cloudmask, pctl)
    if not gaps:
        cloudmask = cloudmask > np.percentile(cloudmask, pctl)

    # Convert -999999 and -Inf to Nans
    data[cloudmask] = -999999
    data[data == -999999] = np.nan
    data[np.isneginf(data)] = np.nan

    # Get indices of non-nan values. These are the indices of the original image array
    data_ind = np.where(~np.isnan(data[:, :, 1]))

    # Reshape into a 2D array, where rows = pixels and cols = features
    data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
    shape = data_vector.shape

    # Remove NaNs
    data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]

    data_mean = data_vector[:, 0:shape[1] - 1].mean(0)
    data_std = data_vector[:, 0:shape[1] - 1].std(0)

    # Normalize data - only the non-binary variables
    if normalize:
        data_vector[:, 0:shape[1]-1] = (data_vector[:, 0:shape[1]-1] - data_mean) / data_std

    return data, data_vector, data_ind, feat_keep
