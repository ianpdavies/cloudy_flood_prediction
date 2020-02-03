import sys

sys.path.append('../')
import tensorflow as tf
import sys
import rasterio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from CPR.utils import preprocessing, tif_stacker, timer
from PIL import Image, ImageEnhance
import h5py

sys.path.append('../')
import numpy.ma as ma
import time
import dask.array as da

sys.path.append('../../')
from CPR.configs import data_path

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)

# ==================================================================================
# Parameters

uncertainty = True
batch = 'v30'
# pctls = [10, 20, 30, 40, 50, 60, 70, 80, 90]
pctls = [40]
BATCH_SIZE = 8192
EPOCHS = 100
DROPOUT_RATE = 0.3  # Dropout rate for MCD
HOLDOUT = 0.3  # Validation data size
NUM_PARALLEL_EXEC_UNITS = 4
remove_perm = True
MC_PASSES = 20

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# To get list of all folders (images) in directory
# img_list = os.listdir(data_path / 'images')

img_list = ['4444_LC08_044033_20170222_2',
            '4101_LC08_027038_20131103_2',
            '4115_LC08_021033_20131227_1',
            '4337_LC08_026038_20160325_1',
            '4444_LC08_043035_20170303_1',
            '4444_LC08_044033_20170222_1',
            '4444_LC08_044033_20170222_4',
            '4444_LC08_045032_20170301_1',
            '4468_LC08_024036_20170501_1',
            '4469_LC08_015035_20170502_1',
            '4514_LC08_027033_20170826_1']

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

myDpi = 400

# ======================================================================================================================
# Get predictions and variances
for i, img in enumerate(img_list):
    print('Creating FN/FP map for {}'.format(img))
    plot_path = data_path / batch / 'plots' / img
    vars_bin_file = data_path / batch / 'variances' / img / 'variances.h5'
    preds_bin_file = data_path / batch / 'predictions' / img / 'predictions.h5'
    stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

    # Reshape variance values back into image band
    with rasterio.open(stack_path, 'r') as ds:
        shape = ds.read(1).shape  # Shape of full original image

    for j, pctl in enumerate(pctls):
        print('Fetching prediction variances for', str(pctl) + '{}'.format('%'))
        # Read predictions
        with h5py.File(vars_bin_file, 'r') as f:
            variances = f[str(pctl)]
            variances = np.array(variances)  # Copy h5 dataset to array

        data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, feat_list_new,
                                                                              test=True)
        var_img = np.zeros(shape)
        var_img[:] = np.nan
        rows, cols = zip(data_ind_test)
        var_img[rows, cols] = variances

        print('Fetching flood predictions for', str(pctl) + '{}'.format('%'))
        # Read predictions
        with h5py.File(preds_bin_file, 'r') as f:
            predictions = f[str(pctl)]
            predictions = np.array(predictions)  # Copy h5 dataset to array

        prediction_img = np.zeros(shape)
        prediction_img[:] = np.nan
        rows, cols = zip(data_ind_test)
        prediction_img[rows, cols] = predictions

        # ----------------------------------------------------------------
        # Plotting
        plt.ioff()
        plot_path = data_path / batch / 'plots' / img
        try:
            plot_path.mkdir(parents=True)
        except FileExistsError:
            pass
        # ----------------------------------------------------------------
        # # Plot predicted floods
        # colors = ['saddlebrown', 'blue']
        # class_labels = ['Predicted No Flood', 'Predicted Flood']
        # legend_patches = [Patch(color=icolor, label=label)
        #                   for icolor, label in zip(colors, class_labels)]
        # cmap = ListedColormap(colors)
        #
        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.imshow(prediction_img, cmap=cmap)
        # ax.legend(handles=legend_patches,
        #           facecolor='white',
        #           edgecolor='white')
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        #
        # myFig = ax.get_figure()
        # myFig.savefig(plot_path / 'predictions.png', dpi=myDpi)
        # plt.close('all')

        # ----------------------------------------------------------------
        # Plot variance
        # fig, ax = plt.subplots(figsize=(10, 10))
        fig, ax = plt.subplots()
        img = ax.imshow(var_img, cmap='plasma')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im_ratio = var_img.shape[0] / var_img.shape[1]
        fig.colorbar(img, ax=ax, fraction=0.046*im_ratio, pad=0.04*im_ratio)
        myFig = ax.get_figure()
        myFig.savefig(plot_path / 'variance.png', dpi=myDpi)
        plt.close('all')

        # ----------------------------------------------------------------
        # Plot trues and falses
        floods = data_test[:, :, 15]
        tp = np.logical_and(prediction_img == 1, floods == 1).astype('int')
        tn = np.logical_and(prediction_img == 0, floods == 0).astype('int')
        fp = np.logical_and(prediction_img == 1, floods == 0).astype('int')
        fn = np.logical_and(prediction_img == 0, floods == 1).astype('int')
        falses = fp + fn
        trues = tp + tn
        # Mask out clouds, etc.
        tp = ma.masked_array(tp, mask=np.isnan(prediction_img))
        tn = ma.masked_array(tn, mask=np.isnan(prediction_img))
        fp = ma.masked_array(fp, mask=np.isnan(prediction_img))
        fn = ma.masked_array(fn, mask=np.isnan(prediction_img))
        falses = ma.masked_array(falses, mask=np.isnan(prediction_img))
        trues = ma.masked_array(trues, mask=np.isnan(prediction_img))

        true_false = fp + (fn * 2) + (tp * 3)
        colors = ['saddlebrown',
                  'red',
                  'limegreen',
                  'blue']
        class_labels = ['True Negatives',
                        'False Floods',
                        'Missed Floods',
                        'True Floods']
        legend_patches = [Patch(color=icolor, label=label)
                          for icolor, label in zip(colors, class_labels)]
        cmap = ListedColormap(colors)

        # fig, ax = plt.subplots(figsize=(10, 10))
        fig, ax = plt.subplots()
        ax.imshow(true_false, cmap=cmap)
        ax.legend(handles=legend_patches,
                  facecolor='white',
                  edgecolor='white')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        myFig = ax.get_figure()
        myFig.savefig(plot_path / 'truefalse.png', dpi=myDpi)
        plt.close('all')

        # ======================================================================================================================
        # Plot variance and predictions but with perm water noted
        feat_list_keep = [feat_list_new[i] for i in feat_keep]  # Removed if feat was deleted in preprocessing
        perm_index = feat_list_keep.index('GSW_perm')
        perm_water = (data_test[:, :, perm_index] == 1)
        perm_water_mask = np.ones(shape)
        perm_water_mask = ma.masked_array(perm_water_mask, mask=~perm_water)

        # ----------------------------------------------------------------
        # Plot predicted floods with perm water noted
        colors = ['gray', 'saddlebrown', 'blue']
        class_labels = ['Permanent Water', 'Predicted No Flood', 'Predicted Flood']
        # Would be nice to add hatches over permanent water
        legend_patches = [Patch(color=icolor, label=label)
                          for icolor, label, in zip(colors, class_labels)]
        cmap = ListedColormap(colors)
        prediction_img_mask = prediction_img.copy()
        prediction_img_mask[perm_water] = -1

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(prediction_img_mask, cmap=cmap)
        ax.legend(handles=legend_patches,
                  facecolor='white',
                  edgecolor='white')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        myFig = ax.get_figure()
        myFig.savefig(plot_path / 'predictions_perm.png', dpi=myDpi)
        plt.close('all')

        # ----------------------------------------------------------------
        # Plot variance with perm water noted
        colors = ['darkgray']
        legend_patches = [Patch(color=colors[0], label='Permanent Water')]
        cmap = ListedColormap(colors)

        # fig, ax = plt.subplots(figsize=(10, 10))
        fig, ax = plt.subplots()
        ax.imshow(var_img, cmap='plasma')
        ax.imshow(perm_water_mask, cmap=cmap)
        ax.legend(handles=legend_patches, facecolor='white', edgecolor='white')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im_ratio = var_img.shape[0] / var_img.shape[1]
        fig.colorbar(img, ax=ax, fraction=0.046*im_ratio, pad=0.04*im_ratio)
        myFig = ax.get_figure()
        myFig.savefig(plot_path / 'variance_perm.png', dpi=myDpi)
        plt.close('all')

        # ----------------------------------------------------------------
        # Plot trues and falses with perm noted
        floods = data_test[:, :, 15]
        tp = np.logical_and(prediction_img == 1, floods == 1).astype('int')
        tn = np.logical_and(prediction_img == 0, floods == 0).astype('int')
        fp = np.logical_and(prediction_img == 1, floods == 0).astype('int')
        fn = np.logical_and(prediction_img == 0, floods == 1).astype('int')
        falses = fp + fn
        trues = tp + tn
        # Mask out clouds, etc.
        tp = ma.masked_array(tp, mask=np.isnan(prediction_img))
        tn = ma.masked_array(tn, mask=np.isnan(prediction_img))
        fp = ma.masked_array(fp, mask=np.isnan(prediction_img))
        fn = ma.masked_array(fn, mask=np.isnan(prediction_img))
        falses = ma.masked_array(falses, mask=np.isnan(prediction_img))
        trues = ma.masked_array(trues, mask=np.isnan(prediction_img))

        true_false = fp + (fn * 2) + (tp * 3)
        true_false[perm_water] = -1
        colors = ['darkgray',
                  'saddlebrown',
                  'red',
                  'limegreen',
                  'blue']
        class_labels = ['Permanent Water',
                        'True Negatives',
                        'False Floods',
                        'Missed Floods',
                        'True Floods']
        legend_patches = [Patch(color=icolor, label=label)
                          for icolor, label in zip(colors, class_labels)]
        cmap = ListedColormap(colors)

        # fig, ax = plt.subplots(figsize=(10, 10))
        fig, ax = plt.subplots()
        ax.imshow(true_false, cmap=cmap)
        ax.legend(handles=legend_patches,
                  facecolor='white',
                  edgecolor='white')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        myFig = ax.get_figure()
        myFig.savefig(plot_path / 'truefalse_perm.png', dpi=myDpi)
        plt.close('all')
# ======================================================================================================================
# Plot variance and incorrect

# Correlation of variance and FP/FN? Maybe using https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pointbiserialr.html
# Strong assumptions about normality and homoscedasticity
# Also logistic regression?
# What about a plot?


# Reshape into 1D arrays
var_img_1d = var_img.reshape([var_img.shape[0] * var_img.shape[1], ])
false_1d = falses.reshape([falses.shape[0] * falses.shape[1], ])

# Remove NaNs
nan_ind = np.where(~np.isnan(var_img_1d))[0]
false_1d = false_1d[~np.isnan(var_img_1d)]
var_img_1d = var_img_1d[~np.isnan(var_img_1d)]

# Convert perm water pixels in variance to NaN
feat_list_keep = [feat_list_new[i] for i in feat_keep]  # Removed if feat was deleted in preprocessing
perm_index = feat_list_keep.index('GSW_perm')
not_perm = np.bitwise_not(data_vector_test[:, perm_index] == 1)
var_img_1d[not_perm] = np.nan

# Remove NaNs due to perm water
false_1d = false_1d[~np.isnan(var_img_1d)]
var_img_1d = var_img_1d[~np.isnan(var_img_1d)]

from scipy.stats import pointbiserialr

pointbiserialr(var_img_1d, false_1d)

# Plotting variance vs. falses
# import pandas as pd
# df = pd.DataFrame()
# import seaborn as sns
# sns.pointplot(x=false_cat, y=var_img_1d)

# ======================================================================================================================
# Correlation matrix of variance with features

feats_var = np.dstack([var_img, data_test])
feats_var_vector = feats_var.reshape([feats_var.shape[0] * feats_var.shape[1], feats_var.shape[2]])
feats_var_vector = feats_var_vector[~np.isnan(feats_var_vector).any(axis=1)]

feat_list_var = feat_list_new
feat_list_var.insert(0, 'variance')
feats_var_df = pd.DataFrame(feats_var_vector, columns=feat_list_var)

corr_matrix = feats_var_df.corr()
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 15))
heatmap = sns.heatmap(corr_matrix,
                      mask=mask,
                      square=True,
                      linewidths=.5,
                      cmap='coolwarm',
                      cbar_kws={'shrink': .4,
                                'ticks': [-1, -.5, 0, 0.5, 1]},
                      vmin=-1,
                      vmax=1,
                      annot=True,
                      annot_kws={'size': 12})
# add the column names as labels
ax.set_yticklabels(corr_matrix.columns, rotation=0)
ax.set_xticklabels(corr_matrix.columns)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

# ======================================================================================================================
# Performance metrics for binned variances

# Get predictions and variances
for i, img in enumerate(img_list):
    print('Creating FN/FP map for {}'.format(img))
    plot_path = data_path / batch / 'plots' / img
    vars_bin_file = data_path / batch / 'variances' / img / 'variances.h5'
    preds_bin_file = data_path / batch / 'predictions' / img / 'predictions.h5'
    stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

    # Reshape variance values back into image band
    with rasterio.open(stack_path, 'r') as ds:
        shape = ds.read(1).shape  # Shape of full original image

    for j, pctl in enumerate(pctls):
        print('Fetching prediction variances for', str(pctl) + '{}'.format('%'))
        # Read predictions
        with h5py.File(vars_bin_file, 'r') as f:
            variances = f[str(pctl)]
            variances = np.array(variances)  # Copy h5 dataset to array

        data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, feat_list_new,
                                                                              test=True)
        var_img = np.zeros(shape)
        var_img[:] = np.nan
        rows, cols = zip(data_ind_test)
        var_img[rows, cols] = variances

        print('Fetching flood predictions for', str(pctl) + '{}'.format('%'))
        # Read predictions
        with h5py.File(preds_bin_file, 'r') as f:
            predictions = f[str(pctl)]
            predictions = np.array(predictions)  # Copy h5 dataset to array

        prediction_img = np.zeros(shape)
        prediction_img[:] = np.nan
        rows, cols = zip(data_ind_test)
        prediction_img[rows, cols] = predictions

# ----------------------------------------------------------------
# Get true/falses
floods = data_test[:, :, 15]
floods = floods.reshape([floods.shape[0] * floods.shape[1], ])
predictions_mask = prediction_img.reshape([prediction_img.shape[0] * prediction_img.shape[1], ])
tp = np.logical_and(predictions_mask == 1, floods == 1).astype('int')
tn = np.logical_and(predictions_mask == 0, floods == 0).astype('int')
fp = np.logical_and(predictions_mask == 1, floods == 0).astype('int')
fn = np.logical_and(predictions_mask == 0, floods == 1).astype('int')
falses = fp + fn
trues = tp + tn
# Mask out clouds, etc.

tp = tp[~np.isnan(predictions_mask)]
tn = tn[~np.isnan(predictions_mask)]
fp = fp[~np.isnan(predictions_mask)]
fn = fn[~np.isnan(predictions_mask)]

var_img_mask = var_img.reshape([var_img.shape[0] * var_img.shape[1], ])
var_img_mask = var_img_mask[~np.isnan(predictions_mask)]

df = np.column_stack([data_vector_test, predictions, var_img_mask, tp, tn, fp, fn])

feat_list_var = feat_list_new + ['predictions', 'variances', 'tp', 'tn', 'fp', 'fn']
df = pd.DataFrame(df, columns=feat_list_var)

bins = [i for i in range(0, 205, 5)]
bins = [x / 1000 for x in bins]
df['var_bins'] = pd.cut(x=df['variances'], bins=bins)

# Sum of trues/falses
df_group = df[['var_bins', 'fn', 'fp', 'tp']].groupby('var_bins').sum()
df_group.plot(kind='bar', width=1)
plt.show()

plot_path = data_path / batch / 'plots'
myFig = plt.get_figure()
myFig.savefig(plot_path / 'variance_bins.png', dpi=myDpi)
