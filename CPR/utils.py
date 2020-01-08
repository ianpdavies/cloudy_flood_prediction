from pathlib import Path
import zipfile
import pandas as pd
import time

# ======================================================================================================================


def tif_stacker(data_path, img, feat_list_new, features, overwrite=False):
    """
    Reorders the tifs (i.e. individual bands) downloaded from GEE according to feature order in feat_list_new,
    then stacks them all into one multiband image called 'stack.tif' located in input path. Requires rasterio,
    os, from zipfile import *

    Reqs: zipfile, Path from pathlib

    Parameters
    ----------
    data_path : str
        Path to image folder
    img :str
        Name of image file (without file extension)
    feat_list_new : list
        List of feature names (str) to be the desired order of the output stacked .tif - target feature must be last
    features : Bool
        Whether stacking feature layers (True) or spectra layers (False)
    overwrite : Bool
        Whether existing stacked image should be overwritten (True)

    Returns
    ----------
    "stack.tif" in 'path' location
    feat_list_files : list
        Not sure what that is or what it's for

    """

    file_list = []
    img_path = data_path / 'images' / img

    if features:
        stack_path = img_path / 'stack' / 'stack.tif'
        img_file = img_path / img
    else:
        stack_path = img_path / 'stack' / 'spectra_stack.tif'
        img_file = img_path / '{}'.format('spectra_' + img)

    # This gets the name of all files in the zip folder, and formats them into a full path readable by rasterio.open()
    with zipfile.ZipFile(str(img_file.with_suffix('.zip')), 'r') as f:
        names = f.namelist()
        names = [str(img_file.with_suffix('.zip!')) + name for name in names]
        names = ['zip://' + name for name in names]
        for file in names:
            if file.endswith('.tif'):
                file_list.append(file)

    feat_list_files = list(map(lambda x: x.split('.')[-2], file_list))  # Grabs a list of features in file order

    if not overwrite:
        if stack_path.exists():
            print('Stack already exists for ' + img)
            return
        else:
            print('No existing stack for ' + img + ', creating one')

    if overwrite:
        # Remove stack file if already exists
        try:
            stack_path.unlink()
            print('Removing existing stack and creating new one')
        except FileNotFoundError:
            print('No existing stack for ' + img + ', creating one')

    # Create 1 row df of file names where each col is a feature name, in the order files are stored locally
    file_arr = pd.DataFrame(data=[file_list], columns=feat_list_files)

    # Then index the file list by the ordered list of feature names used in training
    file_arr = file_arr.loc[:, feat_list_new]

    # The take this re-ordered row as a list - the new file_list
    file_list = list(file_arr.iloc[0, :])

    print(file_list)
    # Read metadata of first file.
    # This needs to be a band in float32 dtype, because it sets the metadata for the entire stack
    # and we are converting the other bands to float64
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta
        meta['dtype'] = 'float32'
    #         print(meta)

    # Update meta to reflect the number of layers
    meta.update(count=len(file_list))

    # Read each layer, convert to float, and write it to stack
    # There's also a gdal way to do this, but unsure how to convert to float:
    # https://gis.stackexchange.com/questions/223910/using-rasterio-or-gdal-to-stack-multiple-bands-without-using-subprocess-commands

    # Make new directory for stacked tif if it doesn't already exist
    try:
        (img_path / 'stack').mkdir()
    except FileExistsError:
        print('Stack directory already exists')

    with rasterio.open(stack_path, 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=0):
            with rasterio.open(layer) as src1:
                dst.write_band(id + 1, src1.read(1).astype('float32'))

    return feat_list_files

# ======================================================================================================================


def preprocessing(data_path, img, pctl, test, normalize=True):
    """
    Masks stacked image with cloudmask by converting cloudy values to NaN

    Parameters
    ----------
    data_path : str
        Path to image folder
    img : str
        Name of image file (without file extension)
    pctl : list of int
        List of integers of cloud cover percentages to mask image with (10, 20, 30, etc.)
    test : bool
        Preprocessing for training or testing data? Test=False means 100-pctl data is returned (i.e. clear pixels)
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

    # load cloudmasks
    clouds_dir = data_path / 'clouds'

    clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))

    # Check for any features that have all zeros/same value and remove. This only matters with the training data
    # Get local image
    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
        data[data == -999999] = np.nan
        data[np.isneginf(data)] = np.nan
        # Now remove NaNs (real clouds, ice, missing data, etc). from cloudmask
        clouds[np.isnan(data[:, :, 0])] = np.nan
        cloudmask = clouds > np.nanpercentile(clouds, pctl)
        data[cloudmask] = -999999
        data[data == -999999] = np.nan
        data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
        data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
        data_std = data_vector[:, 0:data_vector.shape[1] - 1].std(0)

    # Just adding this next line in to correctly remove the deleted feat from feat_list_new during training
    # Should remove once I've decided whether to train with or without perm water
    feat_keep = [a for a in range(data.shape[2])]
    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)

    if 0 in data_std.tolist():
        zero_feat = data_std.tolist().index(0)
        data = np.delete(data, zero_feat, axis=2)
        feat_keep.pop(zero_feat)

    # Convert -999999 and -Inf to Nans
    data[data == -999999] = np.nan
    data[np.isneginf(data)] = np.nan
    # Now remove NaNs (real clouds, ice, missing data, etc). from cloudmask
    clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
    clouds[np.isnan(data[:, :, 0])] = np.nan
    if test:
        cloudmask = clouds > np.nanpercentile(clouds, pctl)  # Data, data_vector, etc = pctl
    if not test:
        cloudmask = clouds < np.nanpercentile(clouds, pctl)  # Data, data_vector, etc = 1 - pctl

    # And mask clouds
    data[cloudmask] = -999999
    data[data == -999999] = np.nan

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


def preprocessing2(data_path, img, pctl, feat_list_new, test, normalize=True):
    """
    Removes ALL pixels that are over permanent water

    Masks stacked image with cloudmask by converting cloudy values to NaN

    Parameters
    ----------
    data_path : str
        Path to image folder
    img : str
        Name of image file (without file extension)
    pctl : list of int
        List of integers of cloud cover percentages to mask image with (10, 20, 30, etc.)
    test : bool
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

    # load cloudmasks
    cloudmask_dir = data_path / 'clouds'

    cloudmask = np.load(cloudmask_dir / '{0}'.format(img+'_clouds.npy'))

    # Check for any features that have all zeros/same value and remove. This only matters with the training data
    cloudmask = cloudmask > np.percentile(cloudmask, pctl)
    # Get local image
    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
        data[cloudmask] = -999999
        data[data == -999999] = np.nan
        data[np.isneginf(data)] = np.nan
        data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
        data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
        data_std = data_vector[:, 0:data_vector.shape[1] - 1].std(0)

    # Just adding this next line in to correctly remove the deleted feat from feat_list_new during training
    # Should remove once I've decided whether to train with or without perm water
    feat_keep = [a for a in range(data.shape[2])]
    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)

    if 0 in data_std.tolist():
        zero_feat = data_std.tolist().index(0)
        data = np.delete(data, zero_feat, axis=2)
        feat_keep.pop(zero_feat)

    cloudmask = np.load(cloudmask_dir / '{0}'.format(img + '_clouds.npy'))
    if test:
        cloudmask = cloudmask < np.percentile(cloudmask, pctl)  # Data, data_vector, etc = pctl
    if not test:
        cloudmask = cloudmask > np.percentile(cloudmask, pctl)  # Data, data_vector, etc = 1 - pctl

    perm_index = feat_list_new.index('GSW_perm')

    # Convert -999999 and -Inf to Nans
    data[cloudmask] = -999999
    data[data[:, :, perm_index] == 1] = -999999
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

def train_val(data_vector, holdout):
    """
    Splits data into train/validation sets after standardizing and removing NaNs
    
    Parameters
    ----------
    data_vector : np.arr 
        Output of preprocessing(). 
    holdout : float
        Fraction of data to be used for validation (e.g. 0.3)   
        
    Returns
    ----------
    data_vector : np.array
        2D array of all data, standardized, with NaNs removed
    training_data : np.array
        2D array of training data, standardized, with NaNs removed
    validation_data : np.array
        2D array of validation data, standardized, with NaNs removed
    training_size : int
        Number of observations in training data
        
    """    
    HOLDOUT_FRACTION = holdout

    # Hold out a fraction of the labeled data for validation.
    training_size = int(data_vector.shape[0] * (1 - HOLDOUT_FRACTION))
    training_data = data_vector[0:training_size,:]
    validation_data = data_vector[training_size:-1,:]
    
    return [training_data, validation_data]

# ======================================================================================================================

import random
import rasterio
import os
from noise import snoise3
import numpy as np
from math import sqrt
from pathlib import Path


def cloud_generator(img, data_path, seed=None, octaves=10, overwrite=False):
    """
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
    cloud_dir = data_path / 'clouds'
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
            print('No cloud image for '+img+', creating one')

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
    freq = np.ceil(sqrt(np.sum(shape)/2)) * octaves  # Frequency calculated based on shape of image

    # Generate 2D (technically 3D, but uses a scalar for z) simplex noise
    for y in range(shape[1]):
        for x in range(shape[0]):
              clouds[x,y] = snoise3(x/freq, y/freq, seed, octaves)

    # Save cloud file as 
    np.save(cloud_file, clouds)

    # Return clouds
    return clouds


# ======================================================================================================================


def timer(start,end, formatted = True):
    """
    Returns formatted elapsed time of time.time() calls.

    Parameters
    ----------
    start : float
        Starting time
    end : float
        Ending time
    formatted : bool
        If True, returns formatted time in hours, minutes, seconds. If False, returns minutes plus remainder as fraction of a minute. 

    Returns
    ----------
    See 'formatted' parameter

    Example
    ----------
    start = time.time()
    end = time.time()
    timer(start, end, formatted = True)
    """
    if formatted == True: # Returns full formated time in hours, minutes, seconds
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        return str("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    else: # Returns minutes + fraction of minute
        minutes, seconds = divmod(time.time() - start, 60)
        seconds = seconds/60
        minutes = minutes + seconds
        return str(minutes)



