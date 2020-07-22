# Using one image as a test
# Divide features into classes based on SD
# Calculate frequency ratio
# Identify low flood hazard areas to remove based on FR results

# Will eventually want to calculated FR for all images, then use those bins to threshold each image

import __init__
import os
import time
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from CPR.utils import tif_stacker, cloud_generator, preprocessing, timer
import pandas as pd
from results_viz import VizFuncs
import sys
import numpy as np
import h5py
from LR_conf_intervals import get_se, get_probs
import matplotlib.pyplot as plt
from pandas.plotting import table
from sklearn.preprocessing import KBinsDiscretizer
import rasterio

sys.path.append('../')
from CPR.configs import data_path

# ======================================================================================================================
# Parameters
# pctls = [10, 30, 50, 70, 90]
pctls = [50]

batch = 'LR_FR'

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# Get all images in image directory
img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test'}
img_list = [x for x in img_list if x not in removed]

feat_list_new = ['GSWDistSeasonal', 'aspect', 'curve', 'elevation', 'hand', 'slope',
                 'spi', 'twi', 'sti', 'precip', 'GSWPerm', 'flooded']

feat_list_all = ['developed', 'forest', 'planted', 'wetlands', 'openspace', 'carbonate', 'noncarbonate',
                 'akl_intrusive',
                 'silicic_resid', 'silicic_resid', 'extrusive_volcanic', 'colluvial_sed', 'glacial_till_clay',
                 'glacial_till_loam', 'glacial_till_coarse', 'glacial_lake_sed_fine', 'glacial_outwash_coarse',
                 'hydric', 'eolian_sed_coarse', 'eolian_sed_fine', 'saline_lake_sed', 'alluv_coastal_sed_fine',
                 'coastal_sed_coarse', 'GSW_distSeasonal', 'aspect', 'curve', 'elevation', 'hand', 'slope', 'spi',
                 'twi', 'sti', 'GSW_perm', 'flooded']

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'batch': batch,
              'feat_list_new': feat_list_new}

# ======================================================================================================================


def preprocessing_discrete_whole_image(data_path, img, feat_list_all, batch):
    img_path = data_path / 'images' / img
    stack_path = img_path / 'stack' / 'stack.tif'
    bins = 7

    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))
        data[data == -999999] = np.nan
        data[np.isneginf(data)] = np.nan
        data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
        data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]

    # Get indices of non-nan values
    nans = np.sum(data, axis=2)
    data_ind = np.where(~np.isnan(nans))
    rows, cols = zip(data_ind)

    # Discretize continuous features
    cts_feats = ['GSW_distSeasonal', 'aspect', 'curve', 'elevation', 'hand', 'slope', 'spi',
                 'twi', 'sti']
    non_cts_feats = ['developed', 'forest', 'planted', 'wetlands', 'openspace', 'carbonate', 'noncarbonate',
                     'akl_intrusive', 'silicic_resid', 'silicic_resid', 'extrusive_volcanic', 'colluvial_sed',
                     'glacial_till_clay', 'glacial_till_loam', 'glacial_till_coarse', 'glacial_lake_sed_fine',
                     'glacial_outwash_coarse', 'hydric', 'eolian_sed_coarse', 'eolian_sed_fine', 'saline_lake_sed',
                     'alluv_coastal_sed_fine', 'coastal_sed_coarse', 'GSW_perm', 'flooded']

    feats_disc = []
    all_edges = pd.DataFrame([])

    # GSW_distSeasonal
    discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='quantile')
    GSW_distSeasonal_disc = discretizer.fit_transform(
        data_vector[:, feat_list_all.index('GSW_distSeasonal')].reshape(-1, 1))
    for i in range(bins):
        feats_disc.append('GSW_distSeasonal_' + str(i + 1))

    disc_nan = np.zeros(data[:, :, 0:bins].shape)
    disc_nan[~np.isnan(disc_nan)] = np.nan
    for bin in range(bins):
        disc_nan[rows, cols, bin] = GSW_distSeasonal_disc[:, bin]

    GSW_distSeasonal_disc = disc_nan
    del disc_nan

    edges = []
    for arr in discretizer.bin_edges_:
        for edge in arr[:-1]:
            edges.append(edge)

    all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)

    # Elevation
    discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='quantile')
    elevation_disc = discretizer.fit_transform(data_vector[:, feat_list_all.index('elevation')].reshape(-1, 1))
    for i in range(bins):
        feats_disc.append('elevation' + str(i + 1))

    disc_nan = np.zeros(data[:, :, 0:bins].shape)
    disc_nan[~np.isnan(disc_nan)] = np.nan
    for bin in range(bins):
        disc_nan[rows, cols, bin] = elevation_disc[:, bin]

    elevation_disc = disc_nan
    del disc_nan

    edges = []
    for arr in discretizer.bin_edges_:
        for edge in arr[:-1]:
            edges.append(edge)

    all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)

    # Slope
    discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='quantile')
    slope_disc = discretizer.fit_transform(data_vector[:, feat_list_all.index('slope')].reshape(-1, 1))
    for i in range(bins):
        feats_disc.append('slope' + str(i + 1))

    disc_nan = np.zeros(data[:, :, 0:bins].shape)
    disc_nan[~np.isnan(disc_nan)] = np.nan
    for bin in range(bins):
        disc_nan[rows, cols, bin] = slope_disc[:, bin]

    slope_disc = disc_nan
    del disc_nan

    edges = []
    for arr in discretizer.bin_edges_:
        for edge in arr[:-1]:
            edges.append(edge)

    all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)

    # TWI
    discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='quantile')
    twi_disc = discretizer.fit_transform(data_vector[:, feat_list_all.index('twi')].reshape(-1, 1))
    for i in range(bins):
        feats_disc.append('twi' + str(i + 1))

    disc_nan = np.zeros(data[:, :, 0:bins].shape)
    disc_nan[~np.isnan(disc_nan)] = np.nan
    for bin in range(bins):
        disc_nan[rows, cols, bin] = twi_disc[:, bin]

    twi_disc = disc_nan
    del disc_nan

    edges = []
    for arr in discretizer.bin_edges_:
        for edge in arr[:-1]:
            edges.append(edge)

    all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)

    # SPI
    discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='quantile')
    spi_disc = discretizer.fit_transform(data_vector[:, feat_list_all.index('spi')].reshape(-1, 1))
    for i in range(bins):
        feats_disc.append('spi' + str(i + 1))

    disc_nan = np.zeros(data[:, :, 0:bins].shape)
    disc_nan[~np.isnan(disc_nan)] = np.nan
    for bin in range(bins):
        disc_nan[rows, cols, bin] = spi_disc[:, bin]

    spi_disc = disc_nan
    del disc_nan

    edges = []
    for arr in discretizer.bin_edges_:
        for edge in arr[:-1]:
            edges.append(edge)

    all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)

    # STI
    discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='quantile')
    sti_disc = discretizer.fit_transform(data_vector[:, feat_list_all.index('sti')].reshape(-1, 1))
    for i in range(bins):
        feats_disc.append('sti' + str(i + 1))

    disc_nan = np.zeros(data[:, :, 0:bins].shape)
    disc_nan[~np.isnan(disc_nan)] = np.nan
    for bin in range(bins):
        disc_nan[rows, cols, bin] = sti_disc[:, bin]

    sti_disc = disc_nan
    del disc_nan

    edges = []
    for arr in discretizer.bin_edges_:
        for edge in arr[:-1]:
            edges.append(edge)

    all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)

    # Curve (flat, convex, concave)
    convex = np.zeros((data_vector.shape[0],))
    concave = np.zeros((data_vector.shape[0],))
    flat = np.zeros((data_vector.shape[0],))
    convex[np.where(data_vector[:, feat_list_all.index('curve')] < 0)] = 1
    concave[np.where(data_vector[:, feat_list_all.index('curve')] > 0)] = 1
    flat[np.where(data_vector[:, feat_list_all.index('curve')] == 0)] = 1
    names = ['convex', 'concave', 'flat']
    bins = len(names)
    for name in names:
        feats_disc.append(name)

    curve = np.column_stack([convex, concave, flat])

    shape = data[:, :, 0:curve.shape[1]].shape
    disc_nan = np.zeros(shape)
    disc_nan[~np.isnan(disc_nan)] = np.nan
    for bin in range(bins):
        disc_nan[rows, cols, bin] = curve[:, bin]

    curve = disc_nan

    del disc_nan, convex, concave, flat

    edges = [0, 0, 0]
    all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)

    # Aspect (north, northeast, northwest, south, southeast, southwest, east, west)
    north = np.zeros((data_vector.shape[0],))
    northeast = np.zeros((data_vector.shape[0],))
    east = np.zeros((data_vector.shape[0],))
    southeast = np.zeros((data_vector.shape[0],))
    south = np.zeros((data_vector.shape[0],))
    southwest = np.zeros((data_vector.shape[0],))
    west = np.zeros((data_vector.shape[0],))
    northwest = np.zeros((data_vector.shape[0],))

    north[np.where(np.logical_or.reduce((data_vector[:, feat_list_all.index('aspect')] >= 337.5,
                                         data_vector[:, feat_list_all.index('aspect')] < 22.5)))] = 1
    northeast[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 22.5,
                                              data_vector[:, feat_list_all.index('aspect')] < 67.5)))] = 1
    east[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 67.5,
                                         data_vector[:, feat_list_all.index('aspect')] < 112.5)))] = 1
    southeast[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 112.5,
                                              data_vector[:, feat_list_all.index('aspect')] < 157.5)))] = 1
    south[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 157.5,
                                          data_vector[:, feat_list_all.index('aspect')] < 202.5)))] = 1
    southwest[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 202.5,
                                              data_vector[:, feat_list_all.index('aspect')] < 247.5)))] = 1
    west[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 247.5,
                                         data_vector[:, feat_list_all.index('aspect')] < 292.5)))] = 1
    northwest[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 292.5,
                                              data_vector[:, feat_list_all.index('aspect')] < 337.5)))] = 1
    names = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
    bins = len(names)
    for name in names:
        feats_disc.append(name)

    aspect = np.column_stack([north, northeast, east, southeast, south, southwest, west, northwest])

    shape = data[:, :, 0:aspect.shape[1]].shape
    disc_nan = np.zeros(shape)
    disc_nan[~np.isnan(disc_nan)] = np.nan
    for bin in range(bins):
        disc_nan[rows, cols, bin] = aspect[:, bin]

    aspect = disc_nan

    del disc_nan, north, northeast, east, southeast, south, southwest, west, northwest

    edges = [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]

    all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)

    # Get original discrete features
    orig_disc_inds = []
    for feat in non_cts_feats:
        orig_disc_inds.append(feat_list_all.index(feat))
    orig_disc_data = data[:, :, orig_disc_inds]

    # Combine with new discrete features
    new_disc_data = np.dstack([GSW_distSeasonal_disc, elevation_disc, slope_disc, twi_disc,
                               spi_disc, sti_disc, curve, aspect])
    data = np.dstack([new_disc_data, orig_disc_data])

    del orig_disc_data, new_disc_data

    # Combine all edges and features
    all_edges = all_edges.reset_index(drop=True)
    feature_edges = pd.concat([all_edges, pd.DataFrame(data=feats_disc)], axis=1)
    feature_edges.columns = ['edge', 'feature']

    # If a feat has only zeros or 1s, it is removed
    data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
    data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
    std = data_vector[:, 0:data_vector.shape[1] - 2].std(0)

    remove_inds = []
    if 0 in std.tolist():
        zero_inds = np.where(std == 0)[0].tolist()
        for ind in zero_inds:
            remove_inds.append(ind)

    remove_inds = np.unique(remove_inds).tolist()

    feat_list_stack = feats_disc + non_cts_feats
    remove_feats = [feat_list_stack[ind] for ind in remove_inds]
    data_vector_keep = np.delete(data_vector, remove_inds, axis=1)
    feat_keep = [x for x in feat_list_stack if x not in remove_feats]

    feature_edges_keep = feature_edges[~feature_edges.feature.isin(remove_feats)]

    # Save feature class bin edges
    filedir = data_path / batch / 'class_bins'
    try:
        filedir.mkdir(parents=True)
    except FileExistsError:
        pass
    filename = filedir / '{}'.format('feature_edges.csv')
    feature_edges_keep.to_csv(filename, index=False)

    return data, data_vector_keep, data_ind, feat_keep, feature_edges_keep


def frequency_ratio(data_vector, feat_keep, feature_edges_keep):
    """
    Calculate frequency ratio
    FRi = [(# flood pixels in class i)/(Total # flood pixels)] / [(# pixels of class i)/(Total # of pixels)]
    """
    classes = len(feat_keep)
    flood_index = feat_keep.index('flooded')
    perm_index = feat_keep.index('GSW_perm')
    data_vector[data_vector[:, perm_index] == 1, flood_index] = 0  # Remove detected water that is perm water
    frs = []
    for i in range(classes):
        NFi = np.sum(
            np.logical_and(data_vector[:, i] == 1, data_vector[:, flood_index] == 1))  # Num flood pixels in class i
        NFt = np.sum(data_vector[:, flood_index])  # Num total flood pixels
        NPi = np.sum(data_vector[:, i])  # Num pixels in class i
        NPt = data_vector.shape[0]  # Num total pixels
        fr = (NFi / NFt) / (NPi / NPt)
        frs.append(fr)

    fr_feats = pd.concat([pd.DataFrame(data=feat_keep, columns=['feature']),
                          pd.DataFrame(data=frs, columns=['FR']),
                          feature_edges_keep['edge']], axis=1)
    fr_feats = fr_feats.drop([flood_index, perm_index], axis=0)

    return fr_feats


def flood_frequency(data_vector, feat_keep, feature_edges_keep):
    """
    What percent of flooded pixels are in each class?
    """
    classes = len(feat_keep)
    flood_index = feat_keep.index('flooded')
    perm_index = feat_keep.index('GSW_perm')
    data_vector[data_vector[:, perm_index] == 1, flood_index] = 0  # Remove detected water that is perm water
    freqs = []
    for i in range(classes):
        NFi = np.sum(
            np.logical_and(data_vector[:, i] == 1, data_vector[:, flood_index] == 1))  # Num flood pixels in class
        NFt = np.sum(data_vector[:, flood_index])  # Num total flood pixels
        freq = NFi / NFt
        freqs.append(freq)

    freq_feats = pd.concat([pd.DataFrame(data=feat_keep, columns=['feature']),
                            pd.DataFrame(data=freqs, columns=['Frequency']),
                            feature_edges_keep['edge']], axis=1)
    freq_feats = freq_feats.drop([flood_index, perm_index], axis=0)

    return freq_feats


def save_tables_as_pngs(df, img, data_path, type):
    plt.ioff()
    df = df.round(3)
    if type is 'fr':
        file_path = data_path / 'image_info' / 'FR'
    if type is 'freq':
        file_path = data_path / 'image_info' / 'flood_freq'

    try:
        file_path.mkdir(parents=True)
    except FileExistsError:
        pass

    plt.figure(figsize=(10, 5))
    axes = [plt.subplot(1, 2, i + 1, frame_on=False) for i in range(2)]  # no visible frame
    axes[0].xaxis.set_visible(False)  # hide the x axis
    axes[0].yaxis.set_visible(False)  # hide the y axis
    axes[1].xaxis.set_visible(False)  # hide the x axis
    axes[1].yaxis.set_visible(False)  # hide the y axis

    first_half = int(np.floor(len(df) / 2))
    second_half = int(np.ceil(len(df) / 2))
    a = table(axes[0], df.iloc[:first_half], loc='center')  # where df is your data frame
    b = table(axes[1], df.iloc[second_half:], loc='center')  # where df is your data frame

    a.auto_set_font_size(False)
    a.set_fontsize(9)
    # a.scale(1.1, 1.1)

    b.auto_set_font_size(False)
    b.set_fontsize(9)
    # b.scale(1.1, 1.1)

    plt.subplots_adjust(wspace=0.2, top=0.9, bottom=0.1, left=0.07, right=0.94)
    plt.savefig(file_path / '{}'.format(img + '.png'))
    plt.close('all')


# ======================================================================================================================

# Calculate frequency ratios

# img_list = img_list[img_list.index('4468_LC08_024036_20170501_1')+1:]
for img in img_list:
    tif_stacker(data_path, img, feat_list_new, overwrite=False)
    data, data_vector, data_ind, feat_keep, feature_edges_keep = preprocessing_discrete_whole_image(data_path, img,
                                                                                                    feat_list_all, batch)
    fr_feats = frequency_ratio(data_vector, feat_keep, feature_edges_keep)
    save_tables_as_pngs(fr_feats, img, data_path, type='fr')
    fr_feats.to_csv(data_path / 'image_info' / 'FR' / '{}'.format(img + '.csv'))
    freq_feats = flood_frequency(data_vector, feat_keep, feature_edges_keep)
    freq_feats.to_csv(data_path / 'image_info' / 'flood_freq' / '{}'.format(img + '.csv'))
    save_tables_as_pngs(freq_feats, img, data_path, type='freq')

from osgeo import gdal
import gdal


# ======================================================================================================================
#
# img = img_list[0]
# img_path = data_path / 'images' / img
# stack_path = img_path / 'stack' / 'stack.tif'
#
# with rasterio.open(str(stack_path), 'r') as ds:
#     data = ds.read()
#     data = data.transpose((1, -1, 0))
#     data[data == -999999] = np.nan
#     data[np.isneginf(data)] = np.nan
#     data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
#     data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
#
#
# plt.hist(data_vector[:, feat_list_all.index('spi')][data_vector[:, feat_list_all.index('spi')] < 1e2])
#
# from CPR.utils import tif_stacker_spectra
# band_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
# for img in img_list:
#     tif_stacker_spectra(data_path, img, band_list, overwrite=True)
#
#
# data, data_vector, data_ind, feat_keep, feature_edges_keep = preprocessing_discrete_whole_image(data_path, img, feat_list_all, batch)
# fr_feats = frequency_ratio(data_vector, feat_keep, feature_edges_keep)
# freq_feats = flood_frequency(data_vector, feat_keep, feature_edges_keep)
#
# classes = len(feat_keep)
# flood_index = feat_keep.index('flooded')
# perm_index = feat_keep.index('GSW_perm')
# data_vector[data_vector[:, perm_index] == 1, flood_index] = 0  # Remove detected water that is perm water
# frs = []
# for i in range(classes):
#     print(i)
#     NFi = np.sum(
#         np.logical_and(data_vector[:, i] == 1, data_vector[:, flood_index] == 1))  # Num flood pixels in class i
#     NFt = np.sum(data_vector[:, flood_index])  # Num total flood pixels
#     NPi = np.sum(data_vector[:, i])  # Num pixels in class i
#     NPt = data_vector.shape[0]  # Num total pixels
#
#     fr = (NFi / NFt) / (NPi / NPt)
#     frs.append(fr)
#
# fr_feats = pd.concat([pd.DataFrame(data=feat_keep, columns=['feature']),
#                       pd.DataFrame(data=frs, columns=['FR']),
#                       feature_edges_keep['edge']], axis=1)
# fr_feats = fr_feats.drop([flood_index, perm_index], axis=0)








# img_path = data_path / 'images' / img
# stack_path = img_path / 'stack' / 'stack.tif'
# bins = 7
#
# with rasterio.open(str(stack_path), 'r') as ds:
#     data = ds.read()
#     data = data.transpose((1, -1, 0))
#     data[data == -999999] = np.nan
#     data[np.isneginf(data)] = np.nan
#     data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
#     data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
#
# # Get indices of non-nan values
# nans = np.sum(data, axis=2)
# data_ind = np.where(~np.isnan(nans))
# rows, cols = zip(data_ind)
#
# # Discretize continuous features
# cts_feats = ['GSW_distSeasonal', 'aspect', 'curve', 'elevation', 'hand', 'slope', 'spi',
#              'twi', 'sti']
# non_cts_feats = ['developed', 'forest', 'planted', 'wetlands', 'openspace', 'carbonate', 'noncarbonate',
#                  'akl_intrusive', 'silicic_resid', 'silicic_resid', 'extrusive_volcanic', 'colluvial_sed',
#                  'glacial_till_clay', 'glacial_till_loam', 'glacial_till_coarse', 'glacial_lake_sed_fine',
#                  'glacial_outwash_coarse', 'hydric', 'eolian_sed_coarse', 'eolian_sed_fine', 'saline_lake_sed',
#                  'alluv_coastal_sed_fine', 'coastal_sed_coarse', 'GSW_perm', 'flooded']
#
# feats_disc = []
# all_edges = pd.DataFrame([])
#
# # GSW_distSeasonal
# discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='quantile')
# GSW_distSeasonal_disc = discretizer.fit_transform(
#     data_vector[:, feat_list_all.index('GSW_distSeasonal')].reshape(-1, 1))
# for i in range(bins):
#     feats_disc.append('GSW_distSeasonal_' + str(i + 1))
#
# disc_nan = np.zeros(data[:, :, 0:bins].shape)
# disc_nan[~np.isnan(disc_nan)] = np.nan
# for bin in range(bins):
#     disc_nan[rows, cols, bin] = GSW_distSeasonal_disc[:, bin]
#
# GSW_distSeasonal_disc = disc_nan
# del disc_nan
#
# edges = []
# for arr in discretizer.bin_edges_:
#     for edge in arr[:-1]:
#         edges.append(edge)
#
# all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)
#
# # Elevation
# discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='quantile')
# elevation_disc = discretizer.fit_transform(data_vector[:, feat_list_all.index('elevation')].reshape(-1, 1))
# for i in range(bins):
#     feats_disc.append('elevation' + str(i + 1))
#
# disc_nan = np.zeros(data[:, :, 0:bins].shape)
# disc_nan[~np.isnan(disc_nan)] = np.nan
# for bin in range(bins):
#     disc_nan[rows, cols, bin] = elevation_disc[:, bin]
#
# elevation_disc = disc_nan
# del disc_nan
#
# edges = []
# for arr in discretizer.bin_edges_:
#     for edge in arr[:-1]:
#         edges.append(edge)
#
# all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)
#
# # Slope
# discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='quantile')
# slope_disc = discretizer.fit_transform(data_vector[:, feat_list_all.index('slope')].reshape(-1, 1))
# for i in range(bins):
#     feats_disc.append('slope' + str(i + 1))
#
# disc_nan = np.zeros(data[:, :, 0:bins].shape)
# disc_nan[~np.isnan(disc_nan)] = np.nan
# for bin in range(bins):
#     disc_nan[rows, cols, bin] = slope_disc[:, bin]
#
# slope_disc = disc_nan
# del disc_nan
#
# edges = []
# for arr in discretizer.bin_edges_:
#     for edge in arr[:-1]:
#         edges.append(edge)
#
# all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)
#
# # TWI
# discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='quantile')
# twi_disc = discretizer.fit_transform(data_vector[:, feat_list_all.index('twi')].reshape(-1, 1))
# for i in range(bins):
#     feats_disc.append('twi' + str(i + 1))
#
# disc_nan = np.zeros(data[:, :, 0:bins].shape)
# disc_nan[~np.isnan(disc_nan)] = np.nan
# for bin in range(bins):
#     disc_nan[rows, cols, bin] = twi_disc[:, bin]
#
# twi_disc = disc_nan
# del disc_nan
#
# edges = []
# for arr in discretizer.bin_edges_:
#     for edge in arr[:-1]:
#         edges.append(edge)
#
# all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)
#
# # SPI
# discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='quantile')
# spi_disc = discretizer.fit_transform(data_vector[:, feat_list_all.index('spi')].reshape(-1, 1))
# for i in range(bins):
#     feats_disc.append('spi' + str(i + 1))
#
# disc_nan = np.zeros(data[:, :, 0:bins].shape)
# disc_nan[~np.isnan(disc_nan)] = np.nan
# for bin in range(bins):
#     disc_nan[rows, cols, bin] = spi_disc[:, bin]
#
# spi_disc = disc_nan
# del disc_nan
#
# edges = []
# for arr in discretizer.bin_edges_:
#     for edge in arr[:-1]:
#         edges.append(edge)
#
# all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)
#
# # STI
# discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='quantile')
# sti_disc = discretizer.fit_transform(data_vector[:, feat_list_all.index('sti')].reshape(-1, 1))
# for i in range(bins):
#     feats_disc.append('sti' + str(i + 1))
#
# disc_nan = np.zeros(data[:, :, 0:bins].shape)
# disc_nan[~np.isnan(disc_nan)] = np.nan
# for bin in range(bins):
#     disc_nan[rows, cols, bin] = sti_disc[:, bin]
#
# sti_disc = disc_nan
# del disc_nan
#
# edges = []
# for arr in discretizer.bin_edges_:
#     for edge in arr[:-1]:
#         edges.append(edge)
#
# all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)
#
# # Curve (flat, convex, concave)
# convex = np.zeros((data_vector.shape[0],))
# concave = np.zeros((data_vector.shape[0],))
# flat = np.zeros((data_vector.shape[0],))
# convex[np.where(data_vector[:, feat_list_all.index('curve')] < 0)] = 1
# concave[np.where(data_vector[:, feat_list_all.index('curve')] > 0)] = 1
# flat[np.where(data_vector[:, feat_list_all.index('curve')] == 0)] = 1
# names = ['convex', 'concave', 'flat']
# bins = len(names)
# for name in names:
#     feats_disc.append(name)
#
# curve = np.column_stack([convex, concave, flat])
#
# shape = data[:, :, 0:curve.shape[1]].shape
# disc_nan = np.zeros(shape)
# disc_nan[~np.isnan(disc_nan)] = np.nan
# for bin in range(bins):
#     disc_nan[rows, cols, bin] = curve[:, bin]
#
# curve = disc_nan
#
# del disc_nan, convex, concave, flat
#
# edges = [0, 0, 0]
# all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)
#
# # Aspect (north, northeast, northwest, south, southeast, southwest, east, west)
# north = np.zeros((data_vector.shape[0],))
# northeast = np.zeros((data_vector.shape[0],))
# east = np.zeros((data_vector.shape[0],))
# southeast = np.zeros((data_vector.shape[0],))
# south = np.zeros((data_vector.shape[0],))
# southwest = np.zeros((data_vector.shape[0],))
# west = np.zeros((data_vector.shape[0],))
# northwest = np.zeros((data_vector.shape[0],))
#
# north[np.where(np.logical_or.reduce((data_vector[:, feat_list_all.index('aspect')] >= 337.5,
#                                      data_vector[:, feat_list_all.index('aspect')] < 22.5)))] = 1
# northeast[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 22.5,
#                                           data_vector[:, feat_list_all.index('aspect')] < 67.5)))] = 1
# east[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 67.5,
#                                      data_vector[:, feat_list_all.index('aspect')] < 112.5)))] = 1
# southeast[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 112.5,
#                                           data_vector[:, feat_list_all.index('aspect')] < 157.5)))] = 1
# south[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 157.5,
#                                       data_vector[:, feat_list_all.index('aspect')] < 202.5)))] = 1
# southwest[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 202.5,
#                                           data_vector[:, feat_list_all.index('aspect')] < 247.5)))] = 1
# west[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 247.5,
#                                      data_vector[:, feat_list_all.index('aspect')] < 292.5)))] = 1
# northwest[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 292.5,
#                                           data_vector[:, feat_list_all.index('aspect')] < 337.5)))] = 1
# names = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
# bins = len(names)
# for name in names:
#     feats_disc.append(name)
#
# aspect = np.column_stack([north, northeast, east, southeast, south, southwest, west, northwest])
#
# shape = data[:, :, 0:aspect.shape[1]].shape
# disc_nan = np.zeros(shape)
# disc_nan[~np.isnan(disc_nan)] = np.nan
# for bin in range(bins):
#     disc_nan[rows, cols, bin] = aspect[:, bin]
#
# aspect = disc_nan
#
# del disc_nan, north, northeast, east, southeast, south, southwest, west, northwest
#
# edges = [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
#
# all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)
#
# # Get original discrete features
# orig_disc_inds = []
# for feat in non_cts_feats:
#     orig_disc_inds.append(feat_list_all.index(feat))
# orig_disc_data = data[:, :, orig_disc_inds]
#
# # Combine with new discrete features
# new_disc_data = np.dstack([GSW_distSeasonal_disc, elevation_disc, slope_disc, twi_disc,
#                            spi_disc, sti_disc, curve, aspect])
# data = np.dstack([new_disc_data, orig_disc_data])
#
# del orig_disc_data, new_disc_data
#
# # Combine all edges and features
# all_edges = all_edges.reset_index(drop=True)
# feature_edges = pd.concat([all_edges, pd.DataFrame(data=feats_disc)], axis=1)
# feature_edges.columns = ['edge', 'feature']
#
# # If a feat has only zeros or 1s, it is removed
# data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
# data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
# std = data_vector[:, 0:data_vector.shape[1] - 2].std(0)
#
# remove_inds = []
# if 0 in std.tolist():
#     zero_inds = np.where(std == 0)[0].tolist()
#     for ind in zero_inds:
#         remove_inds.append(ind)
#
# remove_inds = np.unique(remove_inds).tolist()
#
# feat_list_stack = feats_disc + non_cts_feats
# remove_feats = [feat_list_stack[ind] for ind in remove_inds]
# data_vector_keep = np.delete(data_vector, remove_inds, axis=1)
# feat_keep = [x for x in feat_list_stack if x not in remove_feats]
#
# feature_edges_keep = feature_edges[~feature_edges.feature.isin(remove_feats)]
#
# # Save feature class bin edges
# filedir = data_path / batch / 'class_bins'
# try:
#     filedir.mkdir(parents=True)
# except FileExistsError:
#     pass
# filename = filedir / '{}'.format('feature_edges.csv')
# feature_edges_keep.to_csv(filename, index=False)


# def lithology_dummy(data_path, img):
#     img_path = data_path / 'images' / img
#     img_file = img_path / img
#
#     lith_file = 'zip://' + str(img_file.with_suffix('.zip!')) + img + '.lith.tif'
#
#     with rasterio.open(lith_file, 'r') as src:
#         lith = src.read().squeeze()
#         lith[lith == -999999] = np.nan
#         lith[np.isneginf(lith)] = np.nan
#
#     carbonate = np.zeros(lith.shape)
#     noncarbonate = np.zeros(lith.shape)
#     akl_intrusive = np.zeros(lith.shape)
#     silicic_resid = np.zeros(lith.shape)
#     extrusive_volcanic = np.zeros(lith.shape)
#     colluvial_sed = np.zeros(lith.shape)
#     glacial_till_clay = np.zeros(lith.shape)
#     glacial_till_loam = np.zeros(lith.shape)
#     glacial_till_coarse = np.zeros(lith.shape)
#     glacial_lake_sed_fine = np.zeros(lith.shape)
#     glacial_outwash_coarse = np.zeros(lith.shape)
#     hydric = np.zeros(lith.shape)
#     eolian_sed_coarse = np.zeros(lith.shape)
#     eolian_sed_fine = np.zeros(lith.shape)
#     saline_lake_sed = np.zeros(lith.shape)
#     alluv_coastal_sed_fine = np.zeros(lith.shape)
#     coastal_sed_coarse = np.zeros(lith.shape)
#
#     lith_feat_list = ['carbonate', 'noncarbonate', 'akl_intrusive', 'silicic_resid', 'silicic_resid',
#                       'extrusive_volcanic', 'colluvial_sed', 'glacial_till_clay', 'glacial_till_loam',
#                       'glacial_till_coarse', 'glacial_lake_sed_fine', 'glacial_outwash_coarse', 'hydric',
#                       'eolian_sed_coarse', 'eolian_sed_fine', 'saline_lake_sed', 'alluv_coastal_sed_fine',
#                       'coastal_sed_coarse']
#
#     carbonate[np.where(lith == 1)] = 1
#     noncarbonate[np.where(lith == 3)] = 1
#     akl_intrusive[np.where(lith == 4)] = 1
#     silicic_resid[np.where(lith == 5)] = 1
#     extrusive_volcanic[np.where(lith == 7)] = 1
#     colluvial_sed[np.where(lith == 8)] = 1
#     glacial_till_clay[np.where(lith == 9)] = 1
#     glacial_till_loam[np.where(lith == 10)] = 1
#     glacial_till_coarse[np.where(lith == 11)] = 1
#     glacial_lake_sed_fine[np.where(lith == 13)] = 1
#     glacial_outwash_coarse[np.where(lith == 14)] = 1
#     hydric[np.where(lith == 15)] = 1
#     eolian_sed_coarse[np.where(lith == 16)] = 1
#     eolian_sed_fine[np.where(lith == 17)] = 1
#     saline_lake_sed[np.where(lith == 18)] = 1
#     alluv_coastal_sed_fine[np.where(lith == 19)] = 1
#     coastal_sed_coarse[np.where(lith == 20)] = 1
#
#     lith_list_all = [carbonate, noncarbonate, akl_intrusive, silicic_resid, silicic_resid,
#                         extrusive_volcanic, colluvial_sed, glacial_till_clay, glacial_till_loam,
#                         glacial_till_coarse, glacial_lake_sed_fine, glacial_outwash_coarse, hydric,
#                         eolian_sed_coarse, eolian_sed_fine, saline_lake_sed, alluv_coastal_sed_fine,
#                         coastal_sed_coarse]
#
#     # # This removes features that aren't here, but would require some tweaking of feat_list_new in other scripts
#     # # Can do that later, for now preprocessing will remove feats if they have all zeroes
#     # lith_list = []
#     # lith_not_present = []
#     # for i, lith_class in enumerate(lith_list_all):
#     #     if np.nansum(lith_class) > 0.0:
#     #         lith_list.append(lith_class)
#     #     else:
#     #         lith_not_present.append(lith_feat_list[i])
#     #
#     # lith_feat_list = [x for x in lith_feat_list if x not in lith_not_present]
#     #
#     # del lith_list_all
#     lith_list = lith_list_all
#
#     # Add NaNs, convert to -999999
#     nan_mask = lith.copy()
#     nan_mask = (nan_mask * 0) + 1
#
#     for lith_class in lith_list:
#         lith_class = np.multiply(lith_class, nan_mask)
#         lith_class[np.isnan(lith_class)] = -999999
#
#     return lith_list, lith_feat_list
#
#
# def landcover_dummy(data_path, img):
#     img_path = data_path / 'images' / img
#     img_file = img_path / img
#
#     lulc_file = 'zip://' + str(img_file.with_suffix('.zip!')) + img + '.landcover.tif'
#
#     with rasterio.open(lulc_file, 'r') as src:
#         lulc = src.read()
#         lulc[lulc == -999999] = np.nan
#         lulc[np.isneginf(lulc)] = np.nan
#
#     developed = np.zeros(lulc.shape)
#     forest = np.zeros(lulc.shape)
#     planted = np.zeros(lulc.shape)
#     wetlands = np.zeros(lulc.shape)
#     openspace = np.zeros(lulc.shape)
#
#     developed[np.where(np.logical_or.reduce((lulc == 22, lulc == 23, lulc == 24)))] = 1
#     forest[np.where(np.logical_or.reduce((lulc == 41, lulc == 42, lulc == 43)))] = 1
#     planted[np.where(np.logical_or.reduce((lulc == 81, lulc == 82)))] = 1
#     wetlands[np.where(np.logical_or.reduce((lulc == 90, lulc == 95)))] = 1
#     openspace[np.where(np.logical_or.reduce((lulc == 21, lulc == 31, lulc == 71)))] = 1
#
#     # Add NaNs, convert to -999999
#     nan_mask = lulc.copy()
#     nan_mask = (nan_mask * 0) + 1
#     lulc_list = [developed, forest, planted, wetlands, openspace]
#     lulc_feat_list = ['developed', 'forest', 'planted', 'wetlands', 'openspace']
#
#     for lulc_class in lulc_list:
#         lulc_class = np.multiply(lulc_class, nan_mask)
#         lulc_class[np.isnan(lulc_class)] = -999999
#
#     return lulc_list, lulc_feat_list
#
#
# import zipfile
#
#
# file_list = []
# img_path = data_path / 'images' / img
#
# stack_path = img_path / 'stack' / 'stack.tif'
# img_file = img_path / img
#
# # This gets the name of all files in the zip folder, and formats them into a full path readable by rasterio.open()
# with zipfile.ZipFile(str(img_file.with_suffix('.zip')), 'r') as f:
#     names = f.namelist()
#     names = [str(img_file.with_suffix('.zip!')) + name for name in names]
#     names = ['zip://' + name for name in names]
#     for file in names:
#         if file.endswith('.tif'):
#             file_list.append(file)
#
# feat_list_files = list(map(lambda x: x.split('.')[-2], file_list))  # Grabs a list of features in file order
#
#
# # Create 1 row df of file names where each col is a feature name, in the order files are stored locally
# file_arr = pd.DataFrame(data=[file_list], columns=feat_list_files)
#
# # Then index the file list by the ordered list of feature names used in training
# file_arr = file_arr.loc[:, feat_list_new]
#
# # The take this re-ordered row as a list - the new file_list
# file_list = list(file_arr.iloc[0, :])
#
# # Put all layers into a list
# layers = []
# for file in file_list:
#     with rasterio.open(file, 'r') as ds:
#         layers.append(ds.read())
#
# # Get LULC layers
# lulc_list, lulc_feat_list = landcover_dummy(data_path, img)
#
# # Get lithology layers
# lith_list, lith_feat_list = lithology_dummy(data_path, img)
#
# # Combine
# layers = lulc_list + lith_list + layers
# layers = [layer.squeeze() for layer in layers]
#
# feat_list = lulc_feat_list + lith_feat_list + feat_list_new
#
# # Make new directory for stacked tif if it doesn't already exist
# try:
#     (img_path / 'stack').mkdir()
# except FileExistsError:
#     print('Stack directory already exists')
#
# # Read metadata of first file.
# # This needs to be a band in float32 dtype, because it sets the metadata for the entire stack
# # and we are converting the other bands to float64
# with rasterio.open(file_list[0]) as src0:
#     meta = src0.meta
#     meta['dtype'] = 'float32'
# #         print(meta)
#
# # Update meta to reflect the number of layers
# meta.update(count=len(feat_list))
#
# with rasterio.open(stack_path, 'w', **meta) as dst:
#     for ind, layer in enumerate(layers):
#         dst.write_band(ind + 1, layer.astype('float32'))
