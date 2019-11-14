import sys
sys.path.append('../')
import tensorflow as tf
import sys
import rasterio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from CPR.utils import preprocessing, tif_stacker
from PIL import Image, ImageEnhance
import h5py
sys.path.append('../')

sys.path.append('../../')
from CPR.configs import data_path

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)

# ==================================================================================
# Training on an image using MCD uncertainty estimation
# Batch size = 8192
# ==================================================================================
# Parameters

uncertainty = True
batch = 'v20'
# pctls = [10, 20, 30, 40, 50, 60, 70, 80, 90]
pctls = [50]
BATCH_SIZE = 8192
EPOCHS = 100
DROPOUT_RATE = 0.3  # Dropout rate for MCD
HOLDOUT = 0.3  # Validation data size
NUM_PARALLEL_EXEC_UNITS = 4
remove_perm = True
MC_PASSES = 100

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# To get list of all folders (images) in directory
# img_list = os.listdir(data_path / 'images')

img_list = ['4444_LC08_044033_20170222_2']
            # '4101_LC08_027038_20131103_1',
            # '4101_LC08_027038_20131103_2',
            # '4101_LC08_027039_20131103_1',
            # '4115_LC08_021033_20131227_1',
            # '4115_LC08_021033_20131227_2',
            # '4337_LC08_026038_20160325_1',
            # '4444_LC08_043034_20170303_1',
            # '4444_LC08_043035_20170303_1',
            # '4444_LC08_044032_20170222_1',
            # '4444_LC08_044033_20170222_1',
            # '4444_LC08_044033_20170222_3',
            # '4444_LC08_044033_20170222_4',
            # '4444_LC08_044034_20170222_1',
            # '4444_LC08_045032_20170301_1',
            # '4468_LC08_022035_20170503_1',
            # '4468_LC08_024036_20170501_1',
            # '4468_LC08_024036_20170501_2',
            # '4469_LC08_015035_20170502_1',
            # '4469_LC08_015036_20170502_1',
            # '4477_LC08_022033_20170519_1',
            # '4514_LC08_027033_20170826_1']

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'GSW_perm', 'aspect', 'curve', 'developed', 'elevation',
                 'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'flooded']

model_params = {'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'verbose': 2,
                'use_multiprocessing': True}

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'uncertainty': uncertainty,
              'batch': batch,
              'feat_list_new': feat_list_new}

# ======================================================================================================================
# Plot variance and predictions
for i, img in enumerate(img_list):
    print('Creating FN/FP map for {}'.format(img))
    plot_path = data_path / batch / 'plots' / 'nn' / img
    vars_bin_file = data_path / batch / 'variances' / 'nn' / img / 'variances.h5'
    preds_bin_file = data_path / batch / 'predictions' / 'nn' / img / 'predictions.h5'
    stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

# Reshape variance values back into image band
    with rasterio.open(stack_path, 'r') as ds:
        shape = ds.read(1).shape  # Shape of full original image

    for j, pctl in enumerate(pctls):
        print('Fetching flood predictions for', str(pctl)+'{}'.format('%'))
        # Read predictions
        with h5py.File(vars_bin_file, 'r') as f:
            variances = f[str(pctl)]
            variances = np.array(variances)  # Copy h5 dataset to array

        data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, gaps=True,
                                                                   normalize=False)
        var_img = np.zeros(shape)
        var_img[:] = np.nan
        rows, cols = zip(data_ind_test)
        var_img[rows, cols] = variances

# Reshape predicted values back into image band
with rasterio.open(stack_path, 'r') as ds:
    shape = ds.read(1).shape  # Shape of full original image

for j, pctl in enumerate(pctls):
    print('Fetching flood predictions for', str(pctl) + '{}'.format('%'))
    # Read predictions
    with h5py.File(preds_bin_file, 'r') as f:
        predictions = f[str(pctl)]
        predictions = np.array(predictions)  # Copy h5 dataset to array

    prediction_img = np.zeros(shape)
    prediction_img[:] = np.nan
    rows, cols = zip(data_ind_test)
    prediction_img[rows, cols] = predictions

plt.figure()
plt.imshow(var_img)
plt.figure()
plt.imshow(prediction_img)

# ======================================================================================================================
# Plot variance and incorrect

# Correlation of variance and FP/FN? Maybe using https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pointbiserialr.html
# Strong assumptions about normality and homoscedasticity
# Also logistic regression?
# What about a plot?

# Get FP and FN
floods = data_test[:, :, 15]
# TP
tp = np.logical_and(prediction_img == 1, floods == 1).astype('int')
# TN
tn = np.logical_and(prediction_img == 0, floods == 0).astype('int')
# FP
fp = np.logical_and(prediction_img == 1, floods == 0).astype('int')
#FN
fn = np.logical_and(prediction_img == 0, floods == 1).astype('int')
