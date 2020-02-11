import __init__
import pandas as pd
import numpy as np
import seaborn as sns
import rasterio
from zipfile import ZipFile
import sys
import os
import tensorflow as tf
from scipy.special import softmax
from scipy.stats import entropy

sys.path.append('../../')
from CPR.configs import data_path

# Version numbers
print('Python Version:', sys.version)

# ======================================================================================================================
# Performance metrics vs. image metadata (dry/flood pixels, image size)
pctls = [10, 30, 50, 70, 90]

# To get list of all folders (images) in directory
img_list = os.listdir(data_path / 'images')
img_list.remove('4115_LC08_021033_20131227_test')

feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

batch = 'RCTs'
trials = ['trial1', 'trial2', 'trial3', 'trial4', 'trial5']
exp_path = data_path / batch / 'results'
try:
    (data_path / batch).mkdir(parents=True)
except FileExistsError:
    pass
try:
    exp_path.mkdir(parents=True)
except FileExistsError:
    pass

# ======================================================================================================================


def preprocessing_random_clouds(data_path, img, pctl, trial):
    """
    Preprocessing but gets cloud image from random cloud file directories
    """
    test = True
    img_path = data_path / 'images' / img
    stack_path = img_path / 'stack' / 'stack.tif'
    clouds_dir = data_path / 'clouds' / 'random' / trial

    # Check for any features that have all zeros/same value and remove. For both train and test sets.
    # Get local image
    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
        data[data == -999999] = np.nan
        data[np.isneginf(data)] = np.nan

    # Convert -999999 and -Inf to Nans
    data[data == -999999] = np.nan
    data[np.isneginf(data)] = np.nan
    # Now remove NaNs (real clouds, ice, missing data, etc). from cloudmask
    clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
    clouds[np.isnan(data[:, :, 0])] = np.nan
    if test:
        cloudmask = np.greater(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))
    if not test:
        cloudmask = np.less(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))

    # And mask clouds
    data[cloudmask] = -999999
    data[data == -999999] = np.nan

    # Get indices of non-nan values. These are the indices of the original image array
    nans = np.sum(data, axis=2)
    data_ind = np.where(~np.isnan(nans))

    # Reshape into a 2D array, where rows = pixels and cols = features
    data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
    shape = data_vector.shape

    # Remove NaNs
    data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]

    # Make sure NaNs are in the same position element-wise in image
    mask = np.sum(data, axis=2)
    data[np.isnan(mask)] = np.nan

    return data, data_vector, data_ind

# ======================================================================================================================
# Getting image-wide mean, variance, entropy of each variable in each trial
means_train = []
variances_train = []
entropies_train = []

# Extract cloud files
clouds_dir = data_path / 'clouds'
for trial in trials:
    trial_clouds_dir = clouds_dir / 'random' / trial
    if os.path.isdir(trial_clouds_dir):
        try:
            trial_clouds_dir.mkdir(parents=True)
        except FileExistsError:
            pass
    trial_clouds_zip = clouds_dir / 'random' / '{}'.format(trial + '.zip')
    with ZipFile(trial_clouds_zip, 'r') as src:
        src.extractall(trial_clouds_dir)

for img in img_list:
    print('Getting means for', img)
    for trial in trials:
        print(trial)
        for pctl in pctls:
            print(pctl)
            data_train, data_vector_train, data_ind_train = preprocessing_random_clouds(data_path, img, pctl, trial)
            perm_index = feat_list_new.index('GSW_perm')
            p = softmax(data_vector_train, axis=1)
            for feat in feat_list_new:
                index = feat_list_new.index(feat)
                means_train.append(np.mean(data_vector_train[:, index]))
                variances_train.append(np.var(data_vector_train[:, index]))
                entropies_train.append(entropy(p[:, index]))

np.savetxt(exp_path / 'means_train.csv', means_train, delimiter=",")
np.savetxt(exp_path / 'variances_train.csv', variances_train, delimiter=",")
np.savetxt(exp_path / 'entropies_train.csv', entropies_train, delimiter=",")

means_test = []
variances_test = []
entropies_test = []

for img in img_list:
    print('Getting means for', img)
    for trial in trials:
        print(trial)
        for pctl in pctls:
            print(pctl)
            data_test, data_vector_test, data_ind_test = preprocessing_random_clouds(data_path, img, pctl, trial)
            perm_index = feat_list_new.index('GSW_perm')
            p = softmax(data_vector_test, axis=1)
            for feat in feat_list_new:
                index = feat_list_new.index(feat)
                means_test.append(np.mean(data_vector_test[:, index]))
                variances_test.append(np.var(data_vector_test[:, index]))
                entropies_test.append(entropy(p[:, index]))

np.savetxt(exp_path / 'means_test.csv', means_test, delimiter=",")
np.savetxt(exp_path / 'variances_test.csv', variances_test, delimiter=",")
np.savetxt(exp_path / 'entropies_test.csv', entropies_test, delimiter=",")

# ======================================================================================================================


def preprocessing_random_clouds_standard(data_path, img, pctl, trial):
    """
    Preprocessing but gets cloud image from random cloud file directories
    Does standardize features; any feature with original std=0 becomes the average standardized value
    """
    test = True
    img_path = data_path / 'images' / img
    stack_path = img_path / 'stack' / 'stack.tif'
    clouds_dir = data_path / 'clouds' / 'random' / trial

    # Check for any features that have all zeros/same value and remove. For both train and test sets.
    # Get local image
    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
        data[data == -999999] = np.nan
        data[np.isneginf(data)] = np.nan

        # Getting std of train dataset
        # Remove NaNs (real clouds, ice, missing data, etc). from cloudmask
        clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
        clouds[np.isnan(data[:, :, 0])] = np.nan
        cloudmask = np.less(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))
        data[cloudmask] = -999999
        data[data == -999999] = np.nan
        data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
        data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
        train_std = data_vector[:, 0:data_vector.shape[1] - 2].std(0)

        # Getting std of test dataset
        # Remove NaNs (real clouds, ice, missing data, etc). from cloudmask
        data = ds.read()
        data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
        data[data == -999999] = np.nan
        data[np.isneginf(data)] = np.nan
        clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
        clouds[np.isnan(data[:, :, 0])] = np.nan
        cloudmask = np.greater(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))
        data[cloudmask] = -999999
        data[data == -999999] = np.nan
        data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
        data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
        test_std = data_vector[:, 0:data_vector.shape[1] - 2].std(0)

    # Now adjust feat_list_new to account for a possible removed feature because of std=0
    feat_keep = feat_list_new.copy()
    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)

    if 0 in train_std.tolist():
        print('Removing', feat_keep[train_std.tolist().index(0)], 'because std=0 in training data')
        zero_feat = train_std.tolist().index(0)
        data = np.delete(data, zero_feat, axis=2)
        feat_keep.pop(zero_feat)

    # Now checking stds of test data if not already removed because of train data
    if 0 in test_std.tolist():
        zero_feat_ind = test_std.tolist().index(0)
        zero_feat = feat_list_new[zero_feat_ind]
        try:
            zero_feat_ind = feat_keep.index(zero_feat)
            feat_keep.pop(feat_list_new.index(zero_feat))
            removed_feat = data[:, :, zero_feat_ind]
            data = np.delete(data, zero_feat_ind, axis=2)
        except ValueError:
            pass

    # Convert -999999 and -Inf to Nans
    data[data == -999999] = np.nan
    data[np.isneginf(data)] = np.nan
    # Now remove NaNs (real clouds, ice, missing data, etc). from cloudmask
    clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
    clouds[np.isnan(data[:, :, 0])] = np.nan
    if test:
        cloudmask = np.greater(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))
    if not test:
        cloudmask = np.less(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))

    # And mask clouds
    data[cloudmask] = -999999
    data[data == -999999] = np.nan

    # Get indices of non-nan values. These are the indices of the original image array
    nans = np.sum(data, axis=2)
    data_ind = np.where(~np.isnan(nans))

    # Reshape into a 2D array, where rows = pixels and cols = features
    data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
    shape = data_vector.shape

    # Remove NaNs
    data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]

    data_mean = data_vector[:, 0:shape[1] - 2].mean(0)
    data_std = data_vector[:, 0:shape[1] - 2].std(0)

    # Normalize data - only the non-binary variables
    data_vector[:, 0:shape[1] - 2] = (data_vector[:, 0:shape[1] - 2] - data_mean) / data_std

    # Add back removed feature with 0 (mean of standardized values)
    if 0 in test_std.tolist() or 0 in train_std.tolist():
        try:
            removed_feat = removed_feat * 0
            data = np.insert(data, zero_feat_ind, removed_feat, axis=2)
            removed_feat_vector = removed_feat.reshape([removed_feat.shape[0] * removed_feat.shape[1], ])
            data_vector = np.insert(data_vector, zero_feat_ind, removed_feat_vector, axis=1)
        except ValueError:
            pass

    # Make sure NaNs are in the same position element-wise in image
    mask = np.sum(data, axis=2)
    data[np.isnan(mask)] = np.nan

    return data, data_vector, data_ind


# ======================================================================================================================
# With standardized data

exp_path = data_path / batch / 'results' / 'standardized'
try:
    (data_path / batch).mkdir(parents=True)
except FileExistsError:
    pass
try:
    exp_path.mkdir(parents=True)
except FileExistsError:
    pass

means_train = []
variances_train = []
entropies_train = []

# Extract cloud files
clouds_dir = data_path / 'clouds'
for trial in trials:
    trial_clouds_dir = clouds_dir / 'random' / trial
    if os.path.isdir(trial_clouds_dir):
        try:
            trial_clouds_dir.mkdir(parents=True)
        except FileExistsError:
            pass
    trial_clouds_zip = clouds_dir / 'random' / '{}'.format(trial + '.zip')
    with ZipFile(trial_clouds_zip, 'r') as src:
        src.extractall(trial_clouds_dir)

for img in img_list:
    print('Getting means for', img)
    for trial in trials:
        print(trial)
        for pctl in pctls:
            print(pctl)
            data_train, data_vector_train, data_ind_train = \
                preprocessing_random_clouds_standard(data_path, img, pctl, trial)
            perm_index = feat_list_new.index('GSW_perm')
            p = softmax(data_vector_train, axis=1)
            for feat in feat_list_new:
                index = feat_list_new.index(feat)
                means_train.append(np.mean(data_vector_train[:, index]))
                variances_train.append(np.var(data_vector_train[:, index]))
                entropies_train.append(entropy(p[:, index]))

np.savetxt(exp_path / 'means_train.csv', means_train, delimiter=",")
np.savetxt(exp_path / 'variances_train.csv', variances_train, delimiter=",")
np.savetxt(exp_path / 'entropies_train.csv', entropies_train, delimiter=",")

means_test = []
variances_test = []
entropies_test = []

for img in img_list:
    print('Getting means for', img)
    for trial in trials:
        print(trial)
        for pctl in pctls:
            print(pctl)
            data_test, data_vector_test, data_ind_test = \
                preprocessing_random_clouds_standard(data_path, img, pctl, trial)
            perm_index = feat_list_new.index('GSW_perm')
            p = softmax(data_vector_test, axis=1)
            for feat in feat_list_new:
                index = feat_list_new.index(feat)
                means_test.append(np.mean(data_vector_test[:, index]))
                variances_test.append(np.var(data_vector_test[:, index]))
                entropies_test.append(entropy(p[:, index]))

np.savetxt(exp_path / 'means_test.csv', means_test, delimiter=",")
np.savetxt(exp_path / 'variances_test.csv', variances_test, delimiter=",")
np.savetxt(exp_path / 'entropies_test.csv', entropies_test, delimiter=",")