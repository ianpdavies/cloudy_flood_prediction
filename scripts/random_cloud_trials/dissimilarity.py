import __init__
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import sys
import os
import tensorflow as tf
from scipy.special import softmax
from scipy.stats import entropy
from CPR.utils import preprocessing

sys.path.append('../../')
from CPR.configs import data_path

# Version numbers
print('Python Version:', sys.version)

# ======================================================================================================================
# Performance metrics vs. image metadata (dry/flood pixels, image size)
pctls = [10, 20, 50, 70, 90]

img_list = ['4101_LC08_027038_20131103_1',
            '4101_LC08_027038_20131103_2',
            '4101_LC08_027039_20131103_1',
            '4115_LC08_021033_20131227_1',
            '4115_LC08_021033_20131227_2',
            '4337_LC08_026038_20160325_1',
            '4444_LC08_043034_20170303_1',
            '4444_LC08_043035_20170303_1',
            '4444_LC08_044032_20170222_1',
            '4444_LC08_044033_20170222_1',
            '4444_LC08_044033_20170222_2',
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

feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

batches = ['v13', 'v14', 'v15', 'v16', 'v17']
trials = ['trial1', 'trial2', 'trial3', 'trial4', 'trial5']
exp_path = data_path / 'experiments' / 'random'
try:
    exp_path.mkdir(parents=True)
except FileExistsError:
    pass

# Set some optimized config parameters
NUM_PARALLEL_EXEC_UNITS = 4
tf.config.threading.set_intra_op_parallelism_threads(NUM_PARALLEL_EXEC_UNITS)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.set_soft_device_placement(True)
# tf.config.experimental.set_visible_devices(NUM_PARALLEL_EXEC_UNITS, 'CPU')
os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

# ======================================================================================================================


def preprocessing_random_clouds(data_path, img, pctl, trial):
    """
    Preprocessing but gets cloud image from random cloud file directories
    """
    test = True
    img_path = data_path / 'images' / img
    stack_path = img_path / 'stack' / 'stack.tif'

    # load cloudmasks
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
    data_ind = np.where(~np.isnan(data[:, :, 1]))

    # Reshape into a 2D array, where rows = pixels and cols = features
    data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
    shape = data_vector.shape

    # Remove NaNs
    data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]

    data_mean = data_vector[:, 0:shape[1] - 2].mean(0)
    data_std = data_vector[:, 0:shape[1] - 2].std(0)

    # Normalize data - only the non-binary variables
    data_vector[:, 0:shape[1] - 2] = (data_vector[:, 0:shape[1] - 2] - data_mean) / data_std

    return data, data_vector, data_ind, feat_keep

# ======================================================================================================================
# Getting image-wide mean, variance, entropy of each variable in each trial

# Image with high variance
trials = ['trial1', 'trial2', 'trial3', 'trial4', 'trial5']
pctls = [10, 30, 50, 70, 90]
means_train = []
variances_train = []
entropies_train = []

feat_list = ['GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest', 'hand', 'other_landcover',
             'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

for img in img_list:
    print('Getting means for', img)
    for trial in trials:
        print(trial)
        for pctl in pctls:
            print(pctl)
            data_train, data_vector_train, data_ind_train, feat_keep = \
            preprocessing_random_clouds(data_path, img, pctl, trial)
            perm_index = feat_keep.index('GSW_perm')
            p = softmax(data_vector_train, axis=1)
            for feat in feat_keep:
                index = feat_keep.index(feat)
                means_train.append(np.mean(data_vector_train[:, index], axis=0))
                variances_train.append(np.var(data_vector_train[:, index], axis=0))
                entropies_train.append(entropy(p[:, index]))

np.savetxt(data_path / 'experiments' / 'means_train.csv', means_train, delimiter=",")
np.savetxt(data_path / 'experiments' / 'variances_train.csv', variances_train, delimiter=",")
np.savetxt(data_path / 'experiments' / 'entropies_train.csv', entropies_train, delimiter=",")

means_test = []
variances_test = []
entropies_test = []

for img in img_list:
    print('Getting means for', img)
    for trial in trials:
        print(trial)
        for pctl in pctls:
            print(pctl)
            data_test, data_vector_test, data_ind_test, feat_keep = \
            preprocessing_random_clouds(data_path, img, pctl, trial)
            perm_index = feat_list_new.index('GSW_perm')
            p = softmax(data_vector_test, axis=1)
            for feat in feat_keep:
                index = feat_keep.index(feat)
                means_test.append(np.mean(data_vector_test[:, index], axis=0))
                variances_test.append(np.var(data_vector_test[:, index], axis=0))
                entropies_test.append(entropy(p[:, index]))

np.savetxt(data_path / 'experiments' / 'means_test.csv', means_test, delimiter=",")
np.savetxt(data_path / 'experiments' / 'variances_test.csv', variances_test, delimiter=",")
np.savetxt(data_path / 'experiments' / 'entropies_test.csv', entropies_test, delimiter=",")

# ======================================================================================================================
from numpy import genfromtxt
means_train = genfromtxt(data_path / 'experiments' / 'means_train.csv', delimiter=',')
variances_train = genfromtxt(data_path / 'experiments' / 'variances_train.csv', delimiter=',')
entropies_train = genfromtxt(data_path / 'experiments' / 'entropies_train.csv', delimiter=',')
means_test = genfromtxt(data_path / 'experiments' / 'means_test.csv', delimiter=',')
variances_test = genfromtxt(data_path / 'experiments' / 'variances_test.csv', delimiter=',')
entropies_test = genfromtxt(data_path / 'experiments' / 'entropies_test.csv', delimiter=',')

# Make arrays
data = pd.DataFrame({'image': np.repeat(img_list, len(pctls)*len(trials)*len(feat_list)),
                     'cloud_cover': np.repeat(pctls, len(img_list)*len(trials)*len(feat_list)),
                     'trial': np.tile(trials, len(pctls)*len(img_list)*len(feat_list)),
                     'feature': np.tile(feat_list, len(pctls)*len(img_list)*len(trials)),
                     'mean_train': means_train,
                     'variance_train': variances_train,
                     'entropy_train': entropies_train,
                     'mean_test': means_test,
                     'variance_test': variances_test,
                     'entropy_test': entropies_test})

# Scatterplots of trial metrics vs. mean/var differences between train/test
# Different scatterplot for each feature
exp_path = data_path / 'experiments' / 'random'
trial_metrics = pd.read_csv(exp_path / 'trials_metrics.csv')

merge = pd.merge(data, trial_metrics, on=['image', 'trial', 'cloud_cover'])
merge['mean_diff'] = merge.mean_train - merge.mean_test
merge['variance_diff'] = merge.variance_train - merge.variance_test
merge['entropy_diff'] = merge.entropy_train - merge.entropy_test

g = sns.FacetGrid(merge, col='feature', col_wrap=4)
g.map(sns.scatterplot, 'mean_diff', 'recall')

g = sns.FacetGrid(merge, col='feature', col_wrap=4)
g.map(sns.scatterplot, 'variance_diff', 'recall')

g = sns.FacetGrid(merge, col='feature', col_wrap=4)
g.map(sns.scatterplot, 'entropy_diff', 'recall')

# Not much of a difference - perhaps they should be standardized though?

# Scatterplots of trial metric variances vs. mean/var differences between train/test
# Variances are calculated for each img+pctl
# Mean/var differences are squared and rooted then averaged for each img+pctl
merge['mean_diff'] = np.sqrt(np.square(merge['mean_diff']))
merge['variance_diff'] = np.sqrt(np.square(merge['variance_diff']))
merge['entropy_diff'] = np.sqrt(np.square(merge['variance_diff']))

merge.groupby(['image', 'cloud_cover'])

trial_metrics.groupby(['image', 'cloud_cover']).var().groupby('image').mean()
trial_metrics.groupby(['image', 'cloud_cover']).var().reset_index()


# ======================================================================================================================
# Measuring dissimilarity between random cloud trial images using PCA and Mahalanobis distance
#
#
# img = img_list[-1]
# pctl = [50]
# # Get image data
# data_train, data_vector_train, data_ind_train, feat_keep = preprocessing(data_path, img, 0, test=False, normalize=True)
# X = data_vector_train[:, :15]
# y = data_vector_train[:, 14]
#
# from sklearn.decomposition import PCA
#
# pca = PCA().fit(X)
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Variance (%)')  # for each component
# plt.show()
#
# pca = PCA(n_components=8).fit_transform(X)
# pc_df = pd.DataFrame(data=pca, columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8'])


# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlabel('Principal Component 1', fontsize=15)
# ax.set_ylabel('Principal Component 2', fontsize=15)
#
# sns.scatterplot(x=pc_df['pc1'], y=pc_df['pc2'])

# Get Mahalanobis distance between two random cloud test sets with high variance
# img_list[13] = '4444_LC08_044034_20170222_1' had high accuracy variance for 30%

# ------------------------------------------------------

# # Get cloud mask
# trial = 'trial1'
# clouds_dir = data_path / 'clouds' / 'random' / trial
# clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
# clouds[np.isnan(data_train[:, :, 0])] = np.nan
# cloudmask1 = clouds > np.nanpercentile(clouds, pctl)
# # Mask data with clouds as -999999
# data_train1 = data_train.copy()
# data_train1[cloudmask1] = -999999
# # Reshape to vector
# data_vector1 = data_train1.reshape([data_train1.shape[0] * data_train1.shape[1], data_train1.shape[2]])
# # Remove NaNs
# data_vector1 = data_vector1[~np.isnan(data_vector1).any(axis=1)]
# indices1 = np.argwhere(np.where(data_vector1[:, 1] == -999999, 1, 0))
# indices1 = indices1.reshape(indices1.shape[0], )
#
# # Get cloud mask
# trial = 'trial2'
# clouds_dir = data_path / 'clouds' / 'random' / trial
# clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
# clouds[np.isnan(data_train[:, :, 0])] = np.nan
# cloudmask2 = clouds > np.nanpercentile(clouds, pctl)
# # Mask data with clouds as -999999
# data_train2 = data_train.copy()
# data_train2[cloudmask2] = -999999
# # Reshape to vector
# data_vector2 = data_train2.reshape([data_train2.shape[0] * data_train2.shape[1], data_train2.shape[2]])
# # Remove NaNs
# data_vector2 = data_vector2[~np.isnan(data_vector2).any(axis=1)]
# indices2 = np.argwhere(np.where(data_vector2[:, 1] == -999999, 1, 0))
#
# # Get respective PCs
# pc_trial1 = pc_df.iloc[indices1]
# pc_trial2 = pc_df.iloc[indices2]
#
# from scipy.spatial.distance import mahalanobis
# from numpy import linalg
#
# np.cov([[1, 0, 0], [0, 1, 0]])
# linalg.inv(np.cov([[1, 0, 0], [0, 1, 0]]))

# ======================================================================================================================
# # Comparing variable distribution using Kullback-Leibler divergence
# import scipy.stats as ss
#
# feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'GSW_perm', 'aspect', 'curve', 'developed', 'elevation',
#                  'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'flooded']
#
# # First, compare standardized distributions
# data_test1, data_vector_test1, data_ind_test1, feat_keep_test1 = preprocessing_random_clouds(data_path, img, pctl,
#                                                                                              'trial1')
# data_test2, data_vector_test2, data_ind_test2, feat_keep_test2 = preprocessing_random_clouds(data_path, img, pctl,
#                                                                                              'trial2')
#
# twi_index = feat_list_new.index('twi')
# twi1 = data_vector_test1[twi_index]
# twi2 = data_vector_test2[twi_index]
# kde1 = ss.gaussian_kde(twi1)
# kde2 = ss.gaussian_kde(twi2)
#
# k1 = np.linspace(np.min(twi1), np.max(twi1), 1000)
# k2 = np.linspace(np.min(twi2), np.max(twi2), 1000)
#
# gv1 = kde1(k1)
# gv2 = kde2(k2)
# e = ss.entropy(gv1, gv2)
#
# plt.figure()
# plt.plot(k1, gv1)
#
# plt.figure()
# plt.plot(k2, gv2)



