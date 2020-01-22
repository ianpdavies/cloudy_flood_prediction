import __init__
import rasterio
import numpy as np
import pandas as pd
import sys

sys.path.append('../')
from CPR.configs import data_path

# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pointbiserialr.html
# Strong assumptions about normality and homoscedasticity
# What about a plot?

from scipy.stats import pointbiserialr, pearsonr

feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

dtypes = ['float32', 'float32', 'float32', 'float32', 'int', 'float32', 'int', 'float32', 'int', 'int', 'float32',
          'float32', 'float32', 'int', 'int', 'int']

img_list = ['4444_LC08_044033_20170222_2',
            '4101_LC08_027038_20131103_1',
            '4101_LC08_027038_20131103_2',
            '4101_LC08_027039_20131103_1',
            '4115_LC08_021033_20131227_1',
            '4115_LC08_021033_20131227_2',
            '4337_LC08_026038_20160325_1',
            '4444_LC08_043034_20170303_1',
            '4444_LC08_043035_20170303_1',
            '4444_LC08_044032_20170222_1',
            '4444_LC08_044033_20170222_1',
            '4444_LC08_044033_20170222_3',
            '4444_LC08_044033_20170222_4',
            '4444_LC08_044034_20170222_1',
            '4444_LC08_045032_20170301_1',
            '4468_LC08_022035_20170503_1',
            '4468_LC08_024036_20170501_1',
            '4468_LC08_024036_20170501_2',
            '4469_LC08_015035_20170502_1',
            '4469_LC08_015036_20170502_1',
            '4477_LC08_022033_20170519_1',
            '4514_LC08_027033_20170826_1']


# # Correlation of ALL images
# with rasterio.open(str(data_path / 'images' / img_list[0] / 'stack' / 'stack.tif'), 'r') as ds:
    # data_vectors = np.zeros((0, ds.count))

# for img in img_list:
    # img_path = data_path / 'images' / img
    # stack_path = img_path / 'stack' / 'stack.tif'
    # with rasterio.open(str(stack_path), 'r') as ds:
        # data = ds.read()
        # data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
        # data[data == -999999] = np.nan
        # data[np.isneginf(data)] = np.nan
        # data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
        # data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
        # data_vectors = np.append(data_vectors, data_vector, axis=0)

# cor_arr = []
# j = 0
# for i in range(0, data_vectors.shape[1], 1):
    # for k in range(0, data_vectors.shape[1], 1):
        # if dtypes[j] != dtypes[k]:
            # if dtypes[i] is 'int':
                # cor_arr.append(pointbiserialr(data_vectors[:, j], data_vectors[:, k])[0])
            # else:
                # cor_arr.append(pointbiserialr(data_vectors[:, k], data_vectors[:, j])[0])
        # else:
            # cor_arr.append(pearsonr(data_vectors[:, j], data_vectors[:, k])[0])
    # j += 1

# cor_arr = np.round(cor_arr, 3).reshape([data_vectors.shape[1], data_vectors.shape[1]])
# np.save(data_path / 'images' / 'corr_matrix.npy', cor_arr)

# Correlation of ALL images, perm water = 0
with rasterio.open(str(data_path / 'images' / img_list[0] / 'stack' / 'stack.tif'), 'r') as ds:
    data_vectors = np.zeros((0, ds.count))

perm_index = feat_list_new.index('GSW_perm')
flood_index = feat_list_new.index('flooded')

for img in img_list:
    img_path = data_path / 'images' / img
    stack_path = img_path / 'stack' / 'stack.tif'
    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
        data[data == -999999] = np.nan
        data[np.isneginf(data)] = np.nan
        data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
        data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
        data_vector[data_vector[:, perm_index] == 1, flood_index] = 0
        data_vectors = np.append(data_vectors, data_vector, axis=0)

cor_arr = []
j = 0
for i in range(0, data_vectors.shape[1], 1):
    for k in range(0, data_vectors.shape[1], 1):
        if dtypes[j] != dtypes[k]:
            if dtypes[i] is 'int':
                cor_arr.append(pointbiserialr(data_vectors[:, j], data_vectors[:, k])[0])
            else:
                cor_arr.append(pointbiserialr(data_vectors[:, k], data_vectors[:, j])[0])
        else:
            cor_arr.append(pearsonr(data_vectors[:, j], data_vectors[:, k])[0])
    j += 1

cor_arr = np.round(cor_arr, 3).reshape([data_vectors.shape[1], data_vectors.shape[1]])
np.save(data_path / 'images' / 'corr_matrix_permzero.npy', cor_arr)


# Correlation of ALL images, perm water = Removed
with rasterio.open(str(data_path / 'images' / img_list[0] / 'stack' / 'stack.tif'), 'r') as ds:
    data_vectors = np.zeros((0, ds.count))

for img in img_list:
    img_path = data_path / 'images' / img
    stack_path = img_path / 'stack' / 'stack.tif'
    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
        data[data == -999999] = np.nan
        data[np.isneginf(data)] = np.nan
        data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
        data_vector[data_vector[:, perm_index] == 1, flood_index] = np.nan
        data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
        data_vectors = np.append(data_vectors, data_vector, axis=0)

cor_arr = []
j = 0
for i in range(0, data_vectors.shape[1], 1):
    for k in range(0, data_vectors.shape[1], 1):
        if dtypes[j] != dtypes[k]:
            if dtypes[i] is 'int':
                cor_arr.append(pointbiserialr(data_vectors[:, j], data_vectors[:, k])[0])
            else:
                cor_arr.append(pointbiserialr(data_vectors[:, k], data_vectors[:, j])[0])
        else:
            cor_arr.append(pearsonr(data_vectors[:, j], data_vectors[:, k])[0])
    j += 1

cor_arr = np.round(cor_arr, 3).reshape([data_vectors.shape[1], data_vectors.shape[1]])
np.save(data_path / 'images' / 'corr_matrix_permremoved.npy', cor_arr)