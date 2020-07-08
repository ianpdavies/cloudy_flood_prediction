from pathlib import Path
import zipfile
import pandas as pd
import time
import random
import rasterio
from noise import snoise3
import numpy as np
from math import sqrt
import os
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

# ======================================================================================================================
def gdrive_unstack(data_path, img, feat_list):
    """
    feat_list must be the order that the tifs are stacked in originally from GEE
    """

    img_path = data_path / 'images' / img
    stack_path = img_path / 'stack' / '{}'.format(img + '.tif')
    img_zip = img_path / '{}'.format(img + '.zip')

    # Write each band as a new tif, add to zip, delete
    temp_tifs = []
    with rasterio.open(str(stack_path), 'r') as ds:
        for i, feat in enumerate(feat_list):
            data = ds.read(i + 1)
            meta = ds.meta
            meta.update(count=1)
            name = str(img + '.' + feat + '.tif')
            temp_tif = img_path / name
            temp_tifs.append(temp_tif)
            with rasterio.open(temp_tif, 'w', **meta) as tmp:
                tmp.write_band(1, data)
        with zipfile.ZipFile(str(img_zip), 'w') as dst:
            for temp_tif in temp_tifs:
                dst.write(temp_tif, os.path.basename(temp_tif))
                os.remove(temp_tif)


def preprocessing_discrete(data_path, img, pctl, feat_list_all, batch, test):
    img_path = data_path / 'images' / img
    stack_path = img_path / 'stack' / 'stack.tif'

    # load cloudmasks
    clouds_dir = data_path / 'clouds'

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
    bins = 5
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
    bins = 5
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
    bins = 5
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
    bins = 5
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
    bins = 5
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
    bins = 2
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

    edges = []
    for arr in discretizer.bin_edges_:
        for edge in arr[:-1]:
            edges.append(edge)

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

    north[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 337.5,
                                          data_vector[:, feat_list_all.index('aspect')] < 22.5)))] = 1
    northeast[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 22.5,
                                              data_vector[:, feat_list_all.index('aspect')] < 67.5)))] = 1
    east[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 67.5,
                                         data_vector[:, feat_list_all.index('aspect')] < 112.5)))] = 1
    southeast[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 157.5,
                                              data_vector[:, feat_list_all.index('aspect')] < 157.5)))] = 1
    south[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 202.5,
                                          data_vector[:, feat_list_all.index('aspect')] < 202.5)))] = 1
    southwest[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 247.5,
                                              data_vector[:, feat_list_all.index('aspect')] < 247.5)))] = 1
    west[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 292.5,
                                         data_vector[:, feat_list_all.index('aspect')] < 337.5)))] = 1
    northwest[np.where(np.logical_and.reduce((data_vector[:, feat_list_all.index('aspect')] >= 337.5,
                                              data_vector[:, feat_list_all.index('aspect')] < 360.5)))] = 1
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

    edges = []
    for arr in discretizer.bin_edges_:
        for edge in arr[:-1]:
            edges.append(edge)

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

    edges = []
    for arr in discretizer.bin_edges_:
        for edge in arr[:-1]:
            edges.append(edge)

    all_edges = pd.concat([all_edges, pd.DataFrame(edges)], axis=0)

    # Combine all edges and features
    feature_edges = pd.concat([all_edges, pd.DataFrame(data=feats_disc)], axis=1)
    feature_edges.columns = ['edge', 'feature']

    # If a feat has only zeros or 1s in test OR train set, it is removed from both
    # Check train set
    clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
    clouds[np.isnan(data[:, :, 0])] = np.nan
    cloudmask = np.less(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))
    data_train = data.copy()
    data_train[cloudmask] = -999999
    data_train[data_train == -999999] = np.nan
    data_vector_train = data_train.reshape([data.shape[0] * data_train.shape[1], data_train.shape[2]])
    data_vector_train = data_vector_train[~np.isnan(data_vector_train).any(axis=1)]
    train_std = data_vector_train[:, 0:data_vector_train.shape[1] - 2].std(0)
    del data_train, data_vector_train

    # Check test set
    clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
    clouds[np.isnan(data[:, :, 0])] = np.nan
    cloudmask = np.less(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))
    data_test = data.copy()
    data_test[cloudmask] = -999999
    data_test[data_test == -999999] = np.nan
    data_vector_test = data_test.reshape([data.shape[0] * data_test.shape[1], data_test.shape[2]])
    data_vector_test = data_vector_test[~np.isnan(data_vector_test).any(axis=1)]
    test_std = data_vector_test[:, 0:data_vector_test.shape[1] - 2].std(0)
    del data_test, data_vector_test

    remove_inds = []
    if 0 in train_std.tolist():
        zero_inds = np.where(train_std == 0)[0].tolist()
        for ind in zero_inds:
            remove_inds.append(ind)

    if 0 in test_std.tolist():
        zero_inds = np.where(test_std == 0)[0].tolist()
        for ind in zero_inds:
            remove_inds.append(ind)

    remove_inds = np.unique(remove_inds).tolist()

    # Mask clouds
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

    # Create data vector
    data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
    data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]

    feat_list_stack = feats_disc + feat_list_all
    remove_feats = [feat_list_stack[ind] for ind in remove_inds]
    data_vector = np.delete(data_vector, remove_inds, axis=1)
    feat_keep = [x for x in feat_list_all if x not in remove_feats]

    feature_edges_keep = feature_edges[~feature_edges.feature.isin(remove_feats)]

    # Save feature class bin edges
    if test:
        filedir = data_path / batch / 'class_bins' / 'test'
    else:
        filedir = data_path / batch / 'class_bins' / 'train'

    try:
        filedir.mkdir()
    except FileExistsError:
        pass

    filename = filedir / '{}'.format('feature_edges_' + str(pctl) + '.csv')
    feature_edges_keep.to_csv(filename, index=False)

    return data, data_vector, data_ind, feat_keep, feature_edges_keep


def soil_dummy(data_path, img):
    # Discretizes soil image into discrete hydrological groups
    img_path = data_path / 'images' / img
    img_file = img_path / img

    soil_file = 'zip://' + str(img_file.with_suffix('.zip!')) + img + '.soil.tif'

    with rasterio.open(soil_file, 'r') as src:
        soil = src.read().squeeze()
        soil[soil == -999999] = np.nan
        soil[np.isneginf(soil)] = np.nan

    hydgrp_a = np.zeros(soil.shape)
    hydgrp_ad = np.zeros(soil.shape)
    hydgrp_b = np.zeros(soil.shape)
    hydgrp_bd = np.zeros(soil.shape)
    hydgrp_c = np.zeros(soil.shape)
    hydgrp_cd = np.zeros(soil.shape)
    hydgrp_d = np.zeros(soil.shape)

    soil_feat_list = ['hydgrp_a', 'hydgrp_ad', 'hydgrp_b', 'hydgrp_bd', 'hydgrp_c', 'hydgrp_cd', 'hydgrp_d']

    hydgrp_a[np.where(soil == 1)] = 1
    hydgrp_ad[np.where(soil == 2)] = 1
    hydgrp_b[np.where(soil == 3)] = 1
    hydgrp_bd[np.where(soil == 4)] = 1
    hydgrp_c[np.where(soil == 5)] = 1
    hydgrp_cd[np.where(soil == 6)] = 1
    hydgrp_d[np.where(soil == 7)] = 1

    soil_list_all = [hydgrp_a, hydgrp_ad, hydgrp_b, hydgrp_bd, hydgrp_c, hydgrp_cd, hydgrp_d]

    soil_list = soil_list_all

    # Add NaNs, convert to -999999
    nan_mask = soil.copy()
    nan_mask = (nan_mask * 0) + 1

    for soil_class in soil_list:
        soil_class = np.multiply(soil_class, nan_mask)
        soil_class[np.isnan(soil_class)] = -999999

    return soil_list, soil_feat_list


def lithology_dummy(data_path, img):
    img_path = data_path / 'images' / img
    img_file = img_path / img

    lith_file = 'zip://' + str(img_file.with_suffix('.zip!')) + img + '.lith.tif'

    with rasterio.open(lith_file, 'r') as src:
        lith = src.read().squeeze()
        lith[lith == -999999] = np.nan
        lith[np.isneginf(lith)] = np.nan

    carbonate = np.zeros(lith.shape)
    noncarbonate = np.zeros(lith.shape)
    akl_intrusive = np.zeros(lith.shape)
    silicic_resid = np.zeros(lith.shape)
    extrusive_volcanic = np.zeros(lith.shape)
    colluvial_sed = np.zeros(lith.shape)
    glacial_till_clay = np.zeros(lith.shape)
    glacial_till_loam = np.zeros(lith.shape)
    glacial_till_coarse = np.zeros(lith.shape)
    glacial_lake_sed_fine = np.zeros(lith.shape)
    glacial_outwash_coarse = np.zeros(lith.shape)
    hydric = np.zeros(lith.shape)
    eolian_sed_coarse = np.zeros(lith.shape)
    eolian_sed_fine = np.zeros(lith.shape)
    saline_lake_sed = np.zeros(lith.shape)
    alluv_coastal_sed_fine = np.zeros(lith.shape)
    coastal_sed_coarse = np.zeros(lith.shape)

    lith_feat_list = ['carbonate', 'noncarbonate', 'akl_intrusive', 'silicic_resid', 'silicic_resid',
                      'extrusive_volcanic', 'colluvial_sed', 'glacial_till_clay', 'glacial_till_loam',
                      'glacial_till_coarse', 'glacial_lake_sed_fine', 'glacial_outwash_coarse', 'hydric',
                      'eolian_sed_coarse', 'eolian_sed_fine', 'saline_lake_sed', 'alluv_coastal_sed_fine',
                      'coastal_sed_coarse']

    carbonate[np.where(lith == 1)] = 1
    noncarbonate[np.where(lith == 3)] = 1
    akl_intrusive[np.where(lith == 4)] = 1
    silicic_resid[np.where(lith == 5)] = 1
    extrusive_volcanic[np.where(lith == 7)] = 1
    colluvial_sed[np.where(lith == 8)] = 1
    glacial_till_clay[np.where(lith == 9)] = 1
    glacial_till_loam[np.where(lith == 10)] = 1
    glacial_till_coarse[np.where(lith == 11)] = 1
    glacial_lake_sed_fine[np.where(lith == 13)] = 1
    glacial_outwash_coarse[np.where(lith == 14)] = 1
    hydric[np.where(lith == 15)] = 1
    eolian_sed_coarse[np.where(lith == 16)] = 1
    eolian_sed_fine[np.where(lith == 17)] = 1
    saline_lake_sed[np.where(lith == 18)] = 1
    alluv_coastal_sed_fine[np.where(lith == 19)] = 1
    coastal_sed_coarse[np.where(lith == 20)] = 1

    lith_list_all = [carbonate, noncarbonate, akl_intrusive, silicic_resid, silicic_resid,
                        extrusive_volcanic, colluvial_sed, glacial_till_clay, glacial_till_loam,
                        glacial_till_coarse, glacial_lake_sed_fine, glacial_outwash_coarse, hydric,
                        eolian_sed_coarse, eolian_sed_fine, saline_lake_sed, alluv_coastal_sed_fine,
                        coastal_sed_coarse]

    # # This removes features that aren't here, but would require some tweaking of feat_list_new in other scripts
    # # Can do that later, for now preprocessing will remove feats if they have all zeroes
    # lith_list = []
    # lith_not_present = []
    # for i, lith_class in enumerate(lith_list_all):
    #     if np.nansum(lith_class) > 0.0:
    #         lith_list.append(lith_class)
    #     else:
    #         lith_not_present.append(lith_feat_list[i])
    #
    # lith_feat_list = [x for x in lith_feat_list if x not in lith_not_present]
    #
    # del lith_list_all
    lith_list = lith_list_all

    # Add NaNs, convert to -999999
    nan_mask = lith.copy()
    nan_mask = (nan_mask * 0) + 1

    for lith_class in lith_list:
        lith_class = np.multiply(lith_class, nan_mask)
        lith_class[np.isnan(lith_class)] = -999999

    return lith_list, lith_feat_list


def landcover_dummy(data_path, img):
    img_path = data_path / 'images' / img
    img_file = img_path / img

    lulc_file = 'zip://' + str(img_file.with_suffix('.zip!')) + img + '.landcover.tif'

    with rasterio.open(lulc_file, 'r') as src:
        lulc = src.read()
        lulc[lulc == -999999] = np.nan
        lulc[np.isneginf(lulc)] = np.nan

    developed = np.zeros(lulc.shape)
    forest = np.zeros(lulc.shape)
    planted = np.zeros(lulc.shape)
    wetlands = np.zeros(lulc.shape)
    openspace = np.zeros(lulc.shape)

    developed[np.where(np.logical_or.reduce((lulc == 22, lulc == 23, lulc == 24)))] = 1
    forest[np.where(np.logical_or.reduce((lulc == 41, lulc == 42, lulc == 43)))] = 1
    planted[np.where(np.logical_or.reduce((lulc == 81, lulc == 82)))] = 1
    wetlands[np.where(np.logical_or.reduce((lulc == 90, lulc == 95)))] = 1
    openspace[np.where(np.logical_or.reduce((lulc == 21, lulc == 31, lulc == 71)))] = 1

    # Add NaNs, convert to -999999
    nan_mask = lulc.copy()
    nan_mask = (nan_mask * 0) + 1
    lulc_list = [developed, forest, planted, wetlands, openspace]
    lulc_feat_list = ['developed', 'forest', 'planted', 'wetlands', 'openspace']

    for lulc_class in lulc_list:
        lulc_class = np.multiply(lulc_class, nan_mask)
        lulc_class[np.isnan(lulc_class)] = -999999

    return lulc_list, lulc_feat_list


def truncate_values(data_vector, feat_list_all):
    """
    Some features have erroneously high values, e.g. SPI = 1e9
    These values get truncated in this function before normalization in preprocessing()
    Other possible solutions might be removal of outliers, or winsorizing
    See: https://stats.stackexchange.com/questions/90443/what-are-the-relative-merits-of-winsorizing-vs-trimming-data
    """

    data_vector[:, feat_list_all.index('spi')]


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

    stack_path = img_path / 'stack' / 'stack.tif'
    img_file = img_path / img

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

    # Put all layers into a list
    layers = []
    for file in file_list:
        with rasterio.open(file, 'r') as ds:
            layers.append(ds.read())

    # Get LULC layers
    lulc_list, lulc_feat_list = landcover_dummy(data_path, img)

    # Get soil layers
    soil_list, soil_feat_list = soil_dummy(data_path, img)

    # Combine
    layers = lulc_list + soil_list + layers
    layers = [layer.squeeze() for layer in layers]

    feat_list = lulc_feat_list + soil_feat_list + feat_list_new

    # Make new directory for stacked tif if it doesn't already exist
    try:
        (img_path / 'stack').mkdir()
    except FileExistsError:
        print('Stack directory already exists')

    # Read metadata of first file.
    # This needs to be a band in float32 dtype, because it sets the metadata for the entire stack
    # and we are converting the other bands to float64
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta
        meta['dtype'] = 'float32'
    #         print(meta)

    # Update meta to reflect the number of layers
    meta.update(count=len(feat_list))

    with rasterio.open(stack_path, 'w', **meta) as dst:
        for ind, layer in enumerate(layers):
            dst.write_band(ind + 1, layer.astype('float32'))

    return feat_list


def tif_stacker_spectra(data_path, img, band_list, overwrite=False):
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
    band_list : list
        List of spectral bands (str) to be the desired order of the output stacked .tif
    overwrite : Bool
        Whether existing stacked image should be overwritten (True)

    Returns
    ----------
    "spectra_stack.tif" in 'path' location

    """

    file_list = []
    img_path = data_path / 'images' / img

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
    file_arr = file_arr.loc[:, band_list]

    # The take this re-ordered row as a list - the new file_list
    file_list = list(file_arr.iloc[0, :])

    # Read metadata of first file.
    # This needs to be a band in float32 dtype, because it sets the metadata for the entire stack
    # and we are converting the other bands to float64
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta
        meta['dtype'] = 'float32'

    # Update meta to reflect the number of layers
    meta.update(count=len(file_list))

    # Make new directory for stacked tif if it doesn't already exist
    try:
        (img_path / 'stack').mkdir()
    except FileExistsError:
        print('Stack directory already exists')

    with rasterio.open(stack_path, 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=0):
            with rasterio.open(layer) as src1:
                dst.write_band(id + 1, src1.read(1).astype('float32'))


# ======================================================================================================================


def preprocessing(data_path, img, pctl, feat_list_all, test):
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

    # Check for any features that have all zeros/same value and remove. For both train and test sets.
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
    feat_keep = feat_list_all.copy()
    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)

    remove_inds = []
    if 0 in train_std.tolist():
        zero_inds = np.where(train_std == 0)[0].tolist()
        for ind in zero_inds:
            remove_inds.append(ind)

    if 0 in test_std.tolist():
        zero_inds = np.where(test_std == 0)[0].tolist()
        for ind in zero_inds:
            remove_inds.append(ind)

    remove_inds = np.unique(remove_inds).tolist()
    remove_feats = [feat_list_all[ind] for ind in remove_inds]
    data = np.delete(data, remove_inds, axis=2)
    feat_keep = [x for x in feat_list_all if x not in remove_feats]

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

    # Make sure NaNs are in the same position element-wise in image
    mask = np.sum(data, axis=2)
    data[np.isnan(mask)] = np.nan

    return data, data_vector, data_ind, feat_keep


# ======================================================================================================================


def preprocessing_gen_model(data_path, img_list_train):
    data_vector_list = []
    for img in img_list_train:
        img_path = data_path / 'images' / img
        stack_path = img_path / 'stack' / 'stack.tif'
        with rasterio.open(str(stack_path), 'r') as ds:
            data = ds.read()
            data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
            data[data == -999999] = np.nan
            data[np.isneginf(data)] = np.nan

        # Reshape into a 2D array, where rows = pixels and cols = features
        data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
        data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
        data_vector_list.append(data_vector)

    data_vector_all = np.concatenate(data_vector_list, axis=0)
    shape = data_vector_all.shape
    data_mean = data_vector_all[:, 0:shape[1] - 2].mean(0)
    data_std = data_vector_all[:, 0:shape[1] - 2].std(0)

    # Normalize data - only the non-binary variables
    data_vector_all[:, 0:shape[1] - 2] = (data_vector_all[:, 0:shape[1] - 2] - data_mean) / data_std

    return data_vector_all


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
    training_data = data_vector[0:training_size, :]
    validation_data = data_vector[training_size:-1, :]

    return [training_data, validation_data]


# ======================================================================================================================


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
    freq = np.ceil(sqrt(np.sum(shape) / 2)) * octaves  # Frequency calculated based on shape of image

    # Generate 2D (technically 3D, but uses a scalar for z) simplex noise
    for y in range(shape[1]):
        for x in range(shape[0]):
            clouds[x, y] = snoise3(x / freq, y / freq, seed, octaves)

    # Save cloud file as 
    np.save(cloud_file, clouds)

    # Return clouds
    return clouds


# ======================================================================================================================


def timer(start, end, formatted=True):
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
    if formatted == True:  # Returns full formated time in hours, minutes, seconds
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        return str("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    else:  # Returns minutes + fraction of minute
        minutes, seconds = divmod(time.time() - start, 60)
        seconds = seconds / 60
        minutes = minutes + seconds
        return str(minutes)

# Add TFW to stack directory and create geotiff with nodata tags
# import zipfile
# from osgeo import gdal
#
# img_list = ['4101_LC08_027038_20131103_1']
#
# for img in img_list:
#     img_path = data_path / 'images' / img
#     stack_path = img_path / 'stack' / 'stack.tif'
#     img_file = img_path / img
#     with zipfile.ZipFile(str(img_file.with_suffix('.zip')), 'r') as f:
#         tfw_old = f.read('{}'.format(img + '.aspect.tfw'))
#         tfw_new = open(str(stack_path.parent / 'stack.tfw'), 'wb')
#         tfw_new.write(tfw_old)
#         tfw_new.close()
#         src_ds = gdal.Open(str(stack_path))
#         print(src_ds.GetGeoTransform())
#         dst_filename = str(stack_path.parent / 'stack_gtiff.tif')
#         driver_format = 'GTiff'
#         driver = gdal.GetDriverByName(driver_format)
#         dst_ds = driver.CreateCopy(dst_filename, src_ds, 0)
#         dst_ds = None
#         src_ds = None
#
#     dst_filename2 = str(stack_path.parent / 'stack_gtiff2.tif')
#     with rasterio.open(dst_filename, 'r') as src_ds:
#         tags = src_ds.tags()
#         with rasterio.open(dst_filename2, 'w', **src_ds.meta) as dst_ds:
#             tags['TIFFTAG_GDAL_NODATA'] = -999999
#             dst_ds.update_tags(**tags)
#             dst_ds.write(src_ds.read())
