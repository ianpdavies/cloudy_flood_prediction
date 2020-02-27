# For each RCT, computes mean, variance, and entropy of test and train data separately

import __init__
import numpy as np
import rasterio
from zipfile import ZipFile
import sys
import os
import scipy.special as sc
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
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]

feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

batch = 'RCTs'
trials = ['trial1', 'trial2', 'trial3', 'trial4', 'trial5', 'trial6', 'trial7', 'trial8', 'trial9', 'trial10']
exp_path = data_path / batch / 'results'
try:
    (data_path / batch).mkdir(parents=True)
except FileExistsError:
    pass
try:
    exp_path.mkdir(parents=True)
except FileExistsError:
    pass

dtypes = ['float32', 'float32', 'float32', 'float32', 'int', 'float32', 'int', 'float32', 'int', 'int', 'float32',
          'float32', 'float32', 'int', 'int', 'int']

# ======================================================================================================================


def preprocessing_random_clouds(data_path, img, pctl, trial, test):
    """
    Preprocessing but gets cloud image from random cloud file directories
    """
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

# Extract cloud files
clouds_dir = data_path / 'clouds'
for trial in trials:
    trial_clouds_dir = clouds_dir / 'random' / trial
    if not os.path.isdir(trial_clouds_dir):
        trial_clouds_dir.mkdir(parents=True)
        trial_clouds_zip = clouds_dir / 'random' / '{}'.format(trial + '.zip')
        with ZipFile(trial_clouds_zip, 'r') as src:
            src.extractall(trial_clouds_dir)

def binary_variance(x):
    p = np.sum(x) / x.shape[0]
    bin_var = p*(1-p)
    return bin_var

def binary_entropy(x):
    return -np.sum(sc.xlogy(x, x) + sc.xlog1py(1 - x, -x))/np.log(2)

for img in img_list:
    print('Getting train means for', img)
    means_train_water = []
    variances_train_water = []
    entropies_train_water = []
    means_train_flood = []
    variances_train_flood = []
    entropies_train_flood = []
    for trial in trials:
        print(trial)
        for pctl in pctls:
            print(pctl)
            data_train, data_vector_train, data_ind_train = preprocessing_random_clouds(data_path, img, pctl, trial, test=False)
            perm_index = feat_list_new.index('GSW_perm')
            flooded = feat_list_new.index('flooded')
            data_vector_train = data_vector_train[data_vector_train[:, flooded] == 1]
            p = sc.softmax(data_vector_train, axis=1)
            for i, feat in enumerate(feat_list_new):
                index = feat_list_new.index(feat)
                x = data_vector_train[:, index]
                px = p[:, index]
                means_train_water.append(np.mean(x))
                if dtypes[i] is 'int':
                    variances_train_water.append(binary_variance(x))
                    entropies_train_water.append(binary_entropy(px))
                if dtypes[i] is 'float32':
                    variances_train_water.append(np.var(x))
                    entropies_train_water.append(entropy(px[px != 0]))

            data_vector_train = data_vector_train[data_vector_train[:, perm_index] == 0]
            for i, feat in enumerate(feat_list_new):
                index = feat_list_new.index(feat)
                x = data_vector_train[:, index]
                px = p[:, index]
                means_train_flood.append(np.mean(x))
                if dtypes[i] is 'int':
                    variances_train_flood.append(binary_variance(x))
                    entropies_train_flood.append(binary_entropy(px))
                if dtypes[i] is 'float32':
                    variances_train_flood.append(np.var(x))
                    entropies_train_flood.append(entropy(px[px != 0]))
    with open(exp_path / 'means_train.csv', 'ab') as f:
        np.savetxt(f, np.array(means_train_water), delimiter=',')
    with open(exp_path / 'variances_train.csv', 'ab') as f:
        np.savetxt(f, np.array(variances_train_water), delimiter=',')
    with open(exp_path / 'entropies_train.csv', 'ab') as f:
        np.savetxt(f, np.array(entropies_train_water), delimiter=',')
    with open(exp_path / 'means_train.csv', 'ab') as f:
        np.savetxt(f, np.array(means_train_flood), delimiter=',')
    with open(exp_path / 'variances_train.csv', 'ab') as f:
        np.savetxt(f, np.array(variances_train_flood), delimiter=',')
    with open(exp_path / 'entropies_train.csv', 'ab') as f:
        np.savetxt(f, np.array(entropies_train_flood), delimiter=',')

for img in img_list:
    print('Getting test means for', img)
    means_test_water = []
    variances_test_water = []
    entropies_test_water = []
    means_test_flood = []
    variances_test_flood = []
    entropies_test_flood = []
    for trial in trials:
        print(trial)
        for pctl in pctls:
            print(pctl)
            data_test, data_vector_test, data_ind_test = preprocessing_random_clouds(data_path, img, pctl, trial, test=True)
            perm_index = feat_list_new.index('GSW_perm')
            flooded = feat_list_new.index('flooded')
            data_vector_test = data_vector_test[data_vector_test[:, flooded] == 1]
            p = sc.softmax(data_vector_test, axis=1)
            for i, feat in enumerate(feat_list_new):
                index = feat_list_new.index(feat)
                x = data_vector_test[:, index]
                px = p[:, index]
                means_test_water.append(np.mean(x))
                if dtypes[i] is 'int':
                    variances_test_water.append(binary_variance(x))
                    entropies_test_water.append(binary_entropy(px))
                if dtypes[i] is 'float32':
                    variances_test_water.append(np.var(x))
                    entropies_test_water.append(entropy(px[px != 0]))

            data_vector_test = data_vector_test[data_vector_test[:, perm_index] == 0]
            for i, feat in enumerate(feat_list_new):
                index = feat_list_new.index(feat)
                x = data_vector_test[:, index]
                px = p[:, index]
                means_test_flood.append(np.mean(x))
                if dtypes[i] is 'int':
                    variances_test_flood.append(binary_variance(x))
                    entropies_test_flood.append(binary_entropy(px))
                if dtypes[i] is 'float32':
                    variances_test_flood.append(np.var(x))
                    entropies_test_flood.append(entropy(px[px != 0]))
    with open(exp_path / 'means_test.csv', 'ab') as f:
        np.savetxt(f, np.array(means_test_water), delimiter=',')
    with open(exp_path / 'variances_test.csv', 'ab') as f:
        np.savetxt(f, np.array(variances_test_water), delimiter=',')
    with open(exp_path / 'entropies_test.csv', 'ab') as f:
        np.savetxt(f, np.array(entropies_test_water), delimiter=',')
    with open(exp_path / 'means_test.csv', 'ab') as f:
        np.savetxt(f, np.array(means_test_flood), delimiter=',')
    with open(exp_path / 'variances_test.csv', 'ab') as f:
        np.savetxt(f, np.array(variances_test_flood), delimiter=',')
    with open(exp_path / 'entropies_test.csv', 'ab') as f:
        np.savetxt(f, np.array(entropies_test_flood), delimiter=',')