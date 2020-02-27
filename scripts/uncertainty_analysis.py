# For LR and BNN uncertainties
# Stack all image predictions and uncertainty into one hdf5 file
# Then create histogram of uncertainty binned for TP, FP, FN
# Warning - stacking and saving creatings a huge (30GB) file

import __init__
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
import os

sys.path.append('../')
import numpy.ma as ma
import time
import dask.array as da
import math
from CPR.configs import data_path

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# ==================================================================================
pctls = [10, 30, 50, 70, 90]

# Get all images in image directory
img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_distSeasonal', 'aspect', 'curve', 'developed', 'elevation',
                 'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm',
                 'flooded']

myDpi = 300


# ======================================================================================================================
# Get all uncertainties (10-90%) for each image, stack into HDF5 file


def stack_all_uncertainties(model, batch, data_path, img_list):
    uncertainty_all = []
    predictions_all = []
    tp_all = []
    tn_all = []
    fp_all = []
    fn_all = []
    if model is 'BNN':
        aleatoric_all = []
        epistemic_all = []
    plot_path = data_path / batch / 'plots'
    output_bin_file = data_path / batch / 'metrics' / 'uncertainty_fpfn.h5'
    for i, img in enumerate(img_list):
        print(img)
        preds_bin_file = data_path / batch / 'predictions' / img / 'predictions.h5'
        stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

        try:
            plot_path.mkdir(parents=True)
        except FileExistsError:
            pass

        # Reshape variance values back into image band
        with rasterio.open(stack_path, 'r') as ds:
            shape = ds.read(1).shape  # Shape of full original image

        for pctl in pctls:
            print(pctl)
            data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl,
                                                                                  feat_list_new, test=True)
            perm_index = feat_keep.index('GSW_perm')
            flood_index = feat_keep.index('flooded')
            floods = data_test[:, :, flood_index]
            perm = data_test[:, :, perm_index]

            if model is 'LR':
                se_lower_bin_file = data_path / batch / 'uncertainties' / img / 'se_lower.h5'
                se_upper_bin_file = data_path / batch / 'uncertainties' / img / 'se_upper.h5'
                with h5py.File(se_lower_bin_file, 'r') as f:
                    lower = f[str(pctl)]
                    lower = np.array(lower)

                with h5py.File(se_upper_bin_file, 'r') as f:
                    upper = f[str(pctl)]
                    upper = np.array(upper)

                uncertainties = upper - lower

            if model is 'BNN':
                aleatoric_bin_file = data_path / batch / 'uncertainties' / img / 'aleatoric_uncertainties.h5'
                epistemic_bin_file = data_path / batch / 'uncertainties' / img / 'epistemic_uncertainties.h5'
                with h5py.File(aleatoric_bin_file, 'r') as f:
                    aleatoric = f[str(pctl)]
                    aleatoric = np.array(aleatoric)

                with h5py.File(epistemic_bin_file, 'r') as f:
                    epistemic = f[str(pctl)]
                    epistemic = np.array(epistemic)

                aleatoric_image = np.zeros(shape)
                aleatoric_image[:] = np.nan
                rows, cols = zip(data_ind_test)
                aleatoric_image[rows, cols] = aleatoric

                epistemic_image = np.zeros(shape)
                epistemic_image[:] = np.nan
                rows, cols = zip(data_ind_test)
                epistemic_image[rows, cols] = epistemic

                uncertainties = aleatoric + epistemic

            unc_image = np.zeros(shape)
            unc_image[:] = np.nan
            rows, cols = zip(data_ind_test)
            unc_image[rows, cols] = uncertainties

            # unc_image[perm == 1] = 0
            # cutoff_value = np.nanpercentile(unc_image, 99.99)  # Truncate values so outliers don't skew colorbar
            # unc_image[unc_image > cutoff_value] = np.round(cutoff_value, 0)

            with h5py.File(preds_bin_file, 'r') as f:
                predictions = f[str(pctl)]
                if model is 'LR':
                    predictions = np.argmax(np.array(predictions), axis=1)  # Copy h5 dataset to array
                if model is 'BNN':
                    predictions = np.array(predictions)

            prediction_img = np.zeros(shape)
            prediction_img[:] = np.nan
            rows, cols = zip(data_ind_test)
            prediction_img[rows, cols] = predictions

            floods = floods.reshape([floods.shape[0] * floods.shape[1], ])
            predictions_mask = prediction_img.reshape([prediction_img.shape[0] * prediction_img.shape[1], ])
            tp = np.logical_and(predictions_mask == 1, floods == 1).astype('int')
            tn = np.logical_and(predictions_mask == 0, floods == 0).astype('int')
            fp = np.logical_and(predictions_mask == 1, floods == 0).astype('int')
            fn = np.logical_and(predictions_mask == 0, floods == 1).astype('int')

            # Mask out clouds, etc.
            tp = tp[~np.isnan(predictions_mask)]
            tn = tn[~np.isnan(predictions_mask)]
            fp = fp[~np.isnan(predictions_mask)]
            fn = fn[~np.isnan(predictions_mask)]

            unc_image_mask = unc_image.reshape([unc_image.shape[0] * unc_image.shape[1], ])
            unc_image_mask = unc_image_mask[~np.isnan(predictions_mask)]

            if model is 'BNN':
                aleatoric_image_mask = aleatoric_image.reshape([aleatoric_image.shape[0] * aleatoric_image.shape[1], ])
                aleatoric_image_mask = aleatoric_image_mask[~np.isnan(predictions_mask)]

                epistemic_image_mask = epistemic_image.reshape([epistemic_image.shape[0] * epistemic_image.shape[1], ])
                epistemic_image_mask = epistemic_image_mask[~np.isnan(predictions_mask)]

                aleatoric_all.append(aleatoric_image_mask)
                epistemic_all.append(epistemic_image_mask)

            predictions_all.append(predictions)
            uncertainty_all.append(unc_image_mask)
            tp_all.append(tp)
            tn_all.append(tn)
            fp_all.append(fp)
            fn_all.append(fn)

    # data_vector_all = np.concatenate(data_vector_all, axis=0)  # Won't work because some features are missing
    predictions_all = np.concatenate(predictions_all, axis=0)
    uncertainty_all = np.concatenate(uncertainty_all, axis=0)
    tp_all = np.concatenate(tp_all, axis=0)
    tn_all = np.concatenate(tn_all, axis=0)
    fp_all = np.concatenate(fp_all, axis=0)
    fn_all = np.concatenate(fn_all, axis=0)

    if model is 'BNN':
        aleatoric_all = np.concatenate(aleatoric_all, axis=0)
        epistemic_all = np.concatenate(epistemic_all, axis=0)

    # df = np.column_stack((data_vector_all, predictions_all, uncertainty_all, tp_all, tn_all, fp_all, fn_all))
    if model is 'LR':
        df = np.column_stack((predictions_all, uncertainty_all, tp_all, tn_all, fp_all, fn_all))

    if model is 'BNN':
        df = np.column_stack(
            (predictions_all, uncertainty_all, aleatoric_all, epistemic_all, tp_all, tn_all, fp_all, fn_all))

    with h5py.File(output_bin_file, 'a') as f:
        if 'uncertainty_fpfn' in f:
            print('Deleting earlier uncertainty/fpfn')
            del f['uncertainty_fpfn']
        f.create_dataset('uncertainty_fpfn', data=df)


# ======================================================================================================================
# # Logistic Regression
batch = 'LR_allwater'
try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# stack_all_uncertainties(model='LR', batch=batch, data_path=data_path, img_list=img_list)

# # Bayesian Neural Network
# batch = 'BNN_kwon'
# try:
# (data_path / batch).mkdir()
# except FileExistsError:
# pass

# stack_all_uncertainties(model='BNN', batch=batch, data_path=data_path, img_list=img_list)

# ======================================================================================================================
# Create histogram
# model = 'BNN'
# batch = 'BNN_kwon'

model = 'LR'
batch = 'LR_allwater'

output_bin_file = data_path / batch / 'metrics' / 'uncertainty_fpfn.h5'
plot_path = data_path / batch / 'plots'

with h5py.File(output_bin_file, 'r') as f:
    df = f['uncertainty_fpfn']
    df = np.array(df)

# feat_list_var = feat_list_new + ['predictions', 'uncertainty', 'tp', 'fp', 'fn']
if model is 'LR':
    feat_list_var = ['predictions', 'uncertainty', 'tp', 'tn', 'fp', 'fn']
    xlabel = 'Predictive Interval'
    plotname = 'LR_uncertainty_hist.png'
if model is 'BNN':
    feat_list_var = ['predictions', 'uncertainty', 'aleatoric', 'epistemic', 'tp', 'tn', 'fp', 'fn']
    xlabel = 'Aleatoric + Epistemic Uncertainty'
    plotname = 'BNN_uncertainty_hist.png'

df = pd.DataFrame(df, columns=feat_list_var)

max_decimal = (int(math.log10(abs(np.nanmax(df['uncertainty'])))) * -1) + 1
# bin_max = np.round(np.nanmax(df['uncertainty']), max_decimal)
bin_max = 0.001
bin_size = bin_max / 40

bin_decimal = (int(math.log10(abs(bin_size))) * -1) + 1
bin_size = np.round(bin_size, bin_decimal)

bins = [i for i in range(0, int(bin_max * 1e6), int(bin_size * 1e6))]
bins = [x / int(1e6) for x in bins]
# bins.insert(0, 1e-9)  # Add unique nearly zero value to catch all the zeros
df['var_bins'] = pd.cut(x=df['uncertainty'], bins=bins)

df_group = df[['var_bins', 'fn', 'fp', 'tp']].groupby('var_bins').sum().reset_index()
df_group['bins_float'] = bins[1:]

set1_colors = sns.color_palette("Set1", 8)
color_inds = [1, 2, 0]
colors = []
for j in range(3):
    for i in color_inds:
        colors.append(set1_colors[i])

class_labels = ['True Positive', 'False Positive', 'False Negative']
legend_patches = [Patch(color=icolor, label=label)
                  for icolor, label in zip(colors, class_labels)]

df_group['tp'] = df_group['tp'] / df_group['tp'].sum()
df_group['fp'] = df_group['fp'] / df_group['fp'].sum()
df_group['fn'] = df_group['fn'] / df_group['fn'].sum()

print('plotting uncertainty hist')
fig, ax = plt.subplots()
w = np.min(np.diff(bins[1:])) / 3
ax.bar(df_group['bins_float'] - w, df_group['tp'], width=w, color=colors[0], align='center')
ax.bar(df_group['bins_float'], df_group['fp'], width=w, color=colors[1], align='center')
ax.bar(df_group['bins_float'] + w, df_group['fn'], width=w, color=colors[2], align='center')
ax.legend(labels=class_labels, handles=legend_patches, loc='lower left', bbox_to_anchor=(0.1, 1),
          ncol=5, borderaxespad=0, frameon=False, prop={'size': 7})
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True, useOffset=True)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.autoscale(tight=True)
ax.set_xlabel(xlabel)
ax.set_ylabel('Prediction Type / Total')
ax.yaxis.offsetText.set_fontsize(SMALL_SIZE)
ax.xaxis.offsetText.set_fontsize(SMALL_SIZE)

plt.savefig(plot_path / plotname, dpi=myDpi)

# plt.close('all')

df_group.to_csv(data_path / batch / 'metrics' / 'uncertainty_binned_allwater.csv')

# ========================

if model is 'BNN':
    type = 'aleatoric'
    max_decimal = (int(math.log10(abs(np.nanmax(df[type])))) * -1) + 1
    bin_max = np.round(np.nanmax(df[type]), max_decimal)
    # bin_max = 0.001
    bin_size = bin_max / 40

    bin_decimal = (int(math.log10(abs(bin_size))) * -1) + 1
    bin_size = np.round(bin_size, bin_decimal)

    bins = [i for i in range(0, int(bin_max * 1e6), int(bin_size * 1e6))]
    bins = [x / int(1e6) for x in bins]
    # bins.insert(0, 1e-9)  # Add unique nearly zero value to catch all the zeros
    df['var_bins'] = pd.cut(x=df[type], bins=bins)

    df_group = df[['var_bins', 'fn', 'fp', 'tp']].groupby('var_bins').sum().reset_index()
    df_group['bins_float'] = bins[1:]

    set1_colors = sns.color_palette("Set1", 8)
    color_inds = [1, 2, 0]
    colors = []
    for j in range(3):
        for i in color_inds:
            colors.append(set1_colors[i])

    class_labels = ['True Positive', 'False Positive', 'False Negative']
    legend_patches = [Patch(color=icolor, label=label)
                      for icolor, label in zip(colors, class_labels)]

    df_group['tp'] = df_group['tp'] / df_group['tp'].sum()
    df_group['fp'] = df_group['fp'] / df_group['fp'].sum()
    df_group['fn'] = df_group['fn'] / df_group['fn'].sum()

    print('plotting aleatoric hist')
    fig, ax = plt.subplots()
    w = np.min(np.diff(bins[1:])) / 3
    ax.bar(df_group['bins_float'] - w, df_group['tp'], width=w, color=colors[0], align='center')
    ax.bar(df_group['bins_float'], df_group['fp'], width=w, color=colors[1], align='center')
    ax.bar(df_group['bins_float'] + w, df_group['fn'], width=w, color=colors[2], align='center')
    ax.legend(labels=class_labels, handles=legend_patches, loc='lower left', bbox_to_anchor=(0.1, 1),
              ncol=5, borderaxespad=0, frameon=False, prop={'size': 7})
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True, useOffset=True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.autoscale(tight=True)
    ax.set_xlabel('Aleatoric Uncertainty')
    ax.set_ylabel('Prediction Type / Total')
    ax.yaxis.offsetText.set_fontsize(SMALL_SIZE)
    ax.xaxis.offsetText.set_fontsize(SMALL_SIZE)

    plt.savefig(plot_path / 'aleatoric_hist.png', dpi=myDpi)
    plt.close('all')
    df_group.to_csv(data_path / batch / 'metrics' / 'aleatoric_binned.csv')

    type = 'epistemic'
    max_decimal = (int(math.log10(abs(np.nanmax(df[type])))) * -1) + 1
    bin_max = np.round(np.nanmax(df[type]), max_decimal)
    # bin_max = 0.001
    bin_size = bin_max / 40

    bin_decimal = (int(math.log10(abs(bin_size))) * -1) + 1
    bin_size = np.round(bin_size, bin_decimal)

    bins = [i for i in range(0, int(bin_max * 1e6), int(bin_size * 1e6))]
    bins = [x / int(1e6) for x in bins]
    # bins.insert(0, 1e-9)  # Add unique nearly zero value to catch all the zeros
    df['var_bins'] = pd.cut(x=df[type], bins=bins)

    df_group = df[['var_bins', 'fn', 'fp', 'tp']].groupby('var_bins').sum().reset_index()
    df_group['bins_float'] = bins[1:]

    set1_colors = sns.color_palette("Set1", 8)
    color_inds = [1, 2, 0]
    colors = []
    for j in range(3):
        for i in color_inds:
            colors.append(set1_colors[i])

    class_labels = ['True Positive', 'False Positive', 'False Negative']
    legend_patches = [Patch(color=icolor, label=label)
                      for icolor, label in zip(colors, class_labels)]

    df_group['tp'] = df_group['tp'] / df_group['tp'].sum()
    df_group['fp'] = df_group['fp'] / df_group['fp'].sum()
    df_group['fn'] = df_group['fn'] / df_group['fn'].sum()

    print('plotting epistemic hist')
    fig, ax = plt.subplots()
    w = np.min(np.diff(bins[1:])) / 3
    ax.bar(df_group['bins_float'] - w, df_group['tp'], width=w, color=colors[0], align='center')
    ax.bar(df_group['bins_float'], df_group['fp'], width=w, color=colors[1], align='center')
    ax.bar(df_group['bins_float'] + w, df_group['fn'], width=w, color=colors[2], align='center')
    ax.legend(labels=class_labels, handles=legend_patches, loc='lower left', bbox_to_anchor=(0.1, 1),
              ncol=5, borderaxespad=0, frameon=False, prop={'size': 7})
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True, useOffset=True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.autoscale(tight=True)
    ax.set_xlabel('Prediction Interval')
    ax.set_ylabel('Prediction Type / Total')
    ax.yaxis.offsetText.set_fontsize(SMALL_SIZE)
    ax.xaxis.offsetText.set_fontsize(SMALL_SIZE)

    plt.savefig(plot_path / 'epistemic_hist.png', dpi=myDpi)
    plt.close('all')
    df_group.to_csv(data_path / batch / 'metrics' / 'epistemic_binned.csv')

# ======================================================================================================================
# # Reshape into 1D arrays
# var_img_1d = var_img.reshape([var_img.shape[0] * var_img.shape[1], ])
# false_1d = falses.reshape([falses.shape[0] * falses.shape[1], ])
#
# # Remove NaNs
# nan_ind = np.where(~np.isnan(var_img_1d))[0]
# false_1d = false_1d[~np.isnan(var_img_1d)]
# var_img_1d = var_img_1d[~np.isnan(var_img_1d)]
#
# # Convert perm water pixels in variance to NaN
# feat_list_keep = [feat_list_new[i] for i in feat_keep]  # Removed if feat was deleted in preprocessing
# perm_index = feat_list_keep.index('GSW_perm')
# not_perm = np.bitwise_not(data_vector_test[:, perm_index] == 1)
# var_img_1d[not_perm] = np.nan
#
# # Remove NaNs due to perm water
# false_1d = false_1d[~np.isnan(var_img_1d)]
# var_img_1d = var_img_1d[~np.isnan(var_img_1d)]
#
# from scipy.stats import pointbiserialr
#
# pointbiserialr(var_img_1d, false_1d)
#
# # Plotting variance vs. falses
# # import pandas as pd
# # df = pd.DataFrame()
# # import seaborn as sns
# sns.pointplot(x=false_cat, y=var_img_1d)

# ======================================================================================================================
# # Correlation matrix of variance with features
#
# feats_var = np.dstack([var_img, data_test])
# feats_var_vector = feats_var.reshape([feats_var.shape[0] * feats_var.shape[1], feats_var.shape[2]])
# feats_var_vector = feats_var_vector[~np.isnan(feats_var_vector).any(axis=1)]
#
# feat_list_var = feat_list_new
# feat_list_var.insert(0, 'variance')
# feats_var_df = pd.DataFrame(feats_var_vector, columns=feat_list_var)
#
# corr_matrix = feats_var_df.corr()
# mask = np.zeros_like(corr_matrix, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
#
# f, ax = plt.subplots(figsize=(11, 15))
# heatmap = sns.heatmap(corr_matrix,
#                       mask=mask,
#                       square=True,
#                       linewidths=.5,
#                       cmap='coolwarm',
#                       cbar_kws={'shrink': .4,
#                                 'ticks': [-1, -.5, 0, 0.5, 1]},
#                       vmin=-1,
#                       vmax=1,
#                       annot=True,
#                       annot_kws={'size': 12})
# # add the column names as labels
# ax.set_yticklabels(corr_matrix.columns, rotation=0)
# ax.set_xticklabels(corr_matrix.columns)
# sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

# ======================================================================================================================
# Display histograms separately