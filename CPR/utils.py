from pathlib import Path
import zipfile
import pandas as pd

def tif_stacker(data_path, img, feat_list_new, overwrite=False): 
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
    
    Returns
    ----------
    "stack.tif" in 'path' location 
    feat_list_files : list 
        Not sure what that is or what it's for 
    
    """
    
    file_list = []
    img_path = data_path / 'images' / img
    stack_path = img_path / 'stack' / 'stack.tif'

    # This gets the name of all files in the zip folder, and formats them into a full path readable by rasterio.open()
    with zipfile.ZipFile(str((img_path / img).with_suffix('.zip')), 'r') as f:
        names = f.namelist()
        zip_path = Path('zip:/'+str(img_path)) / img # Format needed by rasterio
        zip_path = zip_path.with_suffix('.zip!')
        names = [str(zip_path) + name for name in names]
        for file in names:
            if file.endswith('.tif'):
                file_list.append(file)

    feat_list_files = list(map(lambda x: x.split('.')[-2], file_list)) # Grabs a list of features in file order        
    
    if overwrite==False:
            if stack_path.exists() == True:
                print('"stack.tif" already exists for '+ img)
                return
            else:
                print('No existing "stack.tif" for '+img+', creating one')
        
    if overwrite==True:
        # Remove stack file if already exists
        try:
            stack_path.unlink()
            print('Removing existing "stack.tif" and creating new one')
        except FileNotFoundError:
            print('No existing "stack.tif" for '+img+', creating one')
            
    # Create 1 row df of file names where each col is a feature name, in the order files are stored locally
    file_arr = pd.DataFrame(data=[file_list], columns=feat_list_files)

    # Then index the file list by the ordered list of feature names used in training
    file_arr = file_arr.loc[:, feat_list_new]

    # The take this re-ordered row as a list - the new file_list
    file_list = list(file_arr.iloc[0,:])

    print(file_list)
    # Read metadata of first file. This needs to be a band in float32 dtype, because it sets the metadata for the entire stack
    # and we are converting the other bands to float64
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta
        meta['dtype'] = 'float32'
    #         print(meta)

    # Update meta to reflect the number of layers
    meta.update(count = len(file_list))

    # Read each layer, convert to float, and write it to stack
    # There's also a gdal way to do this, but unsure how to convert to float: https://gis.stackexchange.com/questions/223910/using-rasterio-or-gdal-to-stack-multiple-bands-without-using-subprocess-commands

    # Make new directory for stacked tif if it doesn't already exist
    try:
        (img_path / 'stack').mkdir()
    except FileExistsError:
        print('Stack directory already exists') 

    with rasterio.open(stack_path, 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=0):
            with rasterio.open(layer) as src1:
                dst.write_band(id+1, src1.read(1).astype('float32'))

    return feat_list_files

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
from pathlib import Path
def preprocessing(data_path, img, pctl, gaps):
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
    gaps : bool
        Preprocessing cloud gaps or clouds? Determines how to mask out image 
    
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
        data = data.transpose((1, -1, 0)) # Not sure why the rasterio.read output is originally (D, W, H)
    
    # load cloudmasks
    cloudMaskDir = data_path / 'clouds'
    
    cloudMask = np.load(cloudMaskDir / '{0}'.format(img+'_clouds.npy'))

    if gaps==False:
        cloudMask = cloudMask < np.percentile(cloudMask, pctl)
    
    if gaps==True:
        cloudMask = cloudMask > np.percentile(cloudMask, pctl)
    
    # Need to remove NaNs because any arithmetic operation involving an NaN will result in NaN
    data[cloudMask] = -999999
    
    # Convert -999999 to None
    data[data == -999999] = np.nan

    # Get indices of non-nan values. These are the indices of the original image array
    data_ind = np.where(~np.isnan(data[:,:,1]))
    
    # Reshape into a 2D array, where rows = pixels and cols = features
    data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])

    # Remove NaNs
    data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]

    # Compute per-band means and standard deviations of the input bands.
    data_mean = data_vector[:,0:14].mean(0)
    data_std = data_vector[:,0:14].std(0)
    
    # Normalize data - only the non-binary variables
    data_vector[:,0:14] = (data_vector[:,0:14] - data_mean) / data_std

    return data, data_vector, data_ind

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

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

    if overwrite==False:
            if cloud_file.exists() == True:
                print('Cloud image already exists for '+ img)
                return
            else:
                print('No cloud image for '+img+', creating one')

    if overwrite==True:
        try:
            cloud_file.remove()
            print('Removing existing cloud image for '+img+' and creating new one')
        except FileNotFoundError:
            print('No existing cloud image for '+img+'. Creating new one')

    # Make directory for clouds if none exists
    if cloud_dir.is_dir() == False:
        cloud_dir.mkdir()
        print('Creating cloud imagery directory')

    if seed==None:
        seed = (random.randint(1,10000))

    # Get shape of input image to be masked
    with rasterio.open(stack_path) as ds:
        shape = ds.shape

    # Create empty array of zeros to generate clouds on
    clouds = np.zeros(shape)
    freq = np.ceil(sqrt(np.sum(shape)/2)) * octaves # Frequency calculated based on shape of image

    # Generate 2D (technically 3D, but uses a scalar for z) simplex noise
    for y in range(shape[1]):
        for x in range(shape[0]):
              clouds[x,y] = snoise3(x/freq, y/freq, seed, octaves)

    # Save cloud file as 
    np.save(cloud_file, clouds)

    # Return clouds
    return clouds

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
import time

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

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


