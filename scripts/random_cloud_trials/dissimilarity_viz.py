import __init__
import pandas as pd
import numpy as np
import seaborn as sns
import rasterio
from zipfile import ZipFile
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

feat_list = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

batch = 'RCTs'
trials = ['trial1', 'trial2', 'trial3', 'trial4', 'trial5', 'trial6', 'trial7', 'trial8', 'trial9', 'trial10']
res_path = data_path / batch / 'results'
plot_path = data_path / batch / 'plots'
try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass
try:
    res_path.mkdir(parents=True)
except FileExistsError:
    pass
try:
    plot_path.mkdir()
except FileExistsError:
    pass

# ======================================================================================================================

# Create dataframe of all trial metrics

# df = pd.DataFrame(columns=['index', 'cloud_cover', 'accuracy', 'precision', 'recall', 'f1', 'index',
                           # 'pixels', 'flood_pixels', 'dry_pixels', 'trial', 'image', 'flood_dry_ratio'])
# for m, trial in enumerate(trials):
    # stack_list = [data_path / 'images' / img / 'stack' / 'stack.tif' for img in img_list]
    # pixel_counts = []
    # flood_counts = []
    # dry_counts = []
    # imgs = []

    # Get pixel metadata for each image + cloud cover %
    # for j, stack in enumerate(stack_list):
        # print('Getting pixel count of', img_list[j])
        # with rasterio.open(stack, 'r') as ds:
            # img = ds.read(ds.count)
            # img[img == -999999] = np.nan
            # img[np.isneginf(img)] = np.nan
            # clouds_dir = data_path / 'clouds' / 'random' / trial
            # clouds = np.load(clouds_dir / '{0}'.format(img_list[j] + '_clouds.npy'))
            # for k, pctl in enumerate(pctls):
                # cloud_mask = clouds.copy()
                # cloud_mask = cloud_mask < np.percentile(cloud_mask, pctl)
                # img_pixels = np.count_nonzero(~np.isnan(img))
                # img[cloud_mask] = np.nan
                # pixel_count = np.count_nonzero(~np.isnan(img))
                # pixel_counts.append(pixel_count)
                # flood_count = np.sum(img[~np.isnan(img)])
                # flood_counts.append(flood_count)
                # dry_count = pixel_count - flood_count
                # dry_counts.append(dry_count)
                # imgs.append(img_list[j])

    # metadata = np.column_stack([pixel_counts, flood_counts, dry_counts, np.repeat(trial, len(pixel_counts))])
    # metadata = pd.DataFrame(metadata, columns=['pixels', 'flood_pixels', 'dry_pixels', 'trial'])
    # metadata['pixels'] = metadata['pixels'].astype('float32').astype('int32')
    # metadata['flood_pixels'] = metadata['flood_pixels'].astype('float32').astype('int32')
    # metadata['dry_pixels'] = metadata['dry_pixels'].astype('float32').astype('int32')
    # metadata['trial'] = metadata['trial'].astype('str')
    # imgs_df = pd.DataFrame(imgs, columns=['image'])
    # metadata = pd.concat([metadata, imgs_df], axis=1)

    # print('Fetching performance metrics')
    # metrics_path = data_path / batch / trial / 'metrics' / 'testing'
    # file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
    # metrics = pd.concat(pd.read_csv(file) for file in file_list)
    # data = pd.concat([metrics.reset_index(), metadata.reset_index()], axis=1)
    # data['flood_dry_ratio'] = data['flood_pixels'] / data['dry_pixels']
    # df = df.append(data)

# df = df.drop('index', axis=1)
# df.to_csv(res_path / 'trials_metrics.csv', index=False)

# ======================================================================================================================
# Figures

df = pd.read_csv(res_path / 'trials_metrics.csv')

# Boxplot of metrics with whiskers

image_numbers = np.tile(np.repeat(range(len(img_list)), len(pctls))+1, len(trials))
plot_path_boxplot_whiskers = plot_path / 'boxplot_whiskers'
try:
    plot_path_boxplot_whiskers.mkdir(parents=True)
except FileExistsError:
    pass

# Recall
recalls = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['recall'])
plt.figure(figsize=(13, 6))
ax = sns.boxplot(x=image_numbers, y='value', data=recalls, palette='colorblind',
            hue='cloud_cover', linewidth=1, fliersize=2.5)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:9], labels[0:9], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='Images', ylabel='Recall')
for i, artist in enumerate(ax.artists):
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    for j in range(i*6,i*6+6):
        line = ax.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)
plt.savefig(plot_path_boxplot_whiskers / 'recall.png', bbox_inches='tight')

# Precision
precisions = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['precision'])
plt.figure(figsize=(13, 6))
ax = sns.boxplot(x=image_numbers, y='value', data=precisions, palette='colorblind',
            hue='cloud_cover', linewidth=1, fliersize=0.3)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:9], labels[0:9], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='Images', ylabel='Precision')
for i, artist in enumerate(ax.artists):
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    for j in range(i*6, i*6+6):
        line = ax.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)
plt.savefig(plot_path_boxplot_whiskers / 'precision.png', bbox_inches='tight')

# Accuracy
accuracies = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['accuracy'])
plt.figure(figsize=(13, 6))
ax = sns.boxplot(x=image_numbers, y='value', data=accuracies, palette='colorblind',
            hue='cloud_cover', linewidth=1, fliersize=0.3)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:9], labels[0:9], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='Images', ylabel='Accuracy')
for i, artist in enumerate(ax.artists):
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    for j in range(i*6,i*6+6):
        line = ax.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)
plt.savefig(plot_path_boxplot_whiskers / 'accuracy.png', bbox_inches='tight')

# F1
f1s = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['f1'])
plt.figure(figsize=(13, 6))
ax = sns.boxplot(x=image_numbers, y='value', data=f1s, palette='colorblind',
            hue='cloud_cover', linewidth=1, fliersize=0.3)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:9], labels[0:9], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='Images', ylabel='F1 Score')
for i, artist in enumerate(ax.artists):
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    for j in range(i*6,i*6+6):
        line = ax.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)
plt.savefig(plot_path_boxplot_whiskers / 'f1.png', bbox_inches='tight')

plt.close('all')
# ----------------------------------------------------------------------------------------------------------------------
# Boxplot of metrics with strip plot instead of whiskers

image_numbers = np.tile(np.repeat(range(len(img_list)), len(pctls))+1, len(trials))
plot_path_strip = plot_path / 'boxplot_stripplot'
try:
    plot_path_strip.mkdir(parents=True)
except FileExistsError:
    pass

# Recall
recalls = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['recall'])
plt.figure(figsize=(13, 6))
ax = sns.boxplot(x=image_numbers, y='value', data=recalls, palette='colorblind',
            hue='cloud_cover', linewidth=0, fliersize=0)
sns.stripplot(x=image_numbers, y='value', data=recalls, palette='colorblind',
              hue='cloud_cover', jitter=False, size=3, dodge=True)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:9], labels[0:9], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='Images', ylabel='Recall')
plt.savefig(plot_path_strip / 'recall.png', bbox_inches='tight')

# Precision
precisions = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['precision'])
plt.figure(figsize=(13, 6))
ax = sns.boxplot(x=image_numbers, y='value', data=precisions, palette='colorblind',
            hue='cloud_cover', linewidth=0, fliersize=0)
sns.stripplot(x=image_numbers, y='value', data=precisions, palette='colorblind',
              hue='cloud_cover', jitter=False, size=3, dodge=True)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:9], labels[0:9], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='Images', ylabel='Precision')
plt.savefig(plot_path_strip / 'precision.png', bbox_inches='tight')

# Accuracy
accuracies = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['accuracy'])
plt.figure(figsize=(13, 6))
ax = sns.boxplot(x=image_numbers, y='value', data=accuracies, palette='colorblind',
            hue='cloud_cover', linewidth=0, fliersize=0)
sns.stripplot(x=image_numbers, y='value', data=accuracies, palette='colorblind',
              hue='cloud_cover', jitter=False, size=3, dodge=True)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:9], labels[0:9], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='Images', ylabel='Accuracy')
plt.savefig(plot_path_strip / 'accuracy.png', bbox_inches='tight')

# F1
f1s = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['f1'])
plt.figure(figsize=(13, 6))
ax = sns.boxplot(x=image_numbers, y='value', data=f1s, palette='colorblind',
            hue='cloud_cover', linewidth=0, fliersize=0)
sns.stripplot(x=image_numbers, y='value', data=f1s, palette='colorblind',
              hue='cloud_cover', jitter=False, size=3, dodge=True)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:9], labels[0:9], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='Images', ylabel='F1 Score')
plt.savefig(plot_path_strip / 'f1.png', bbox_inches='tight')

plt.close('all')

# ----------------------------------------------------------------------------------------------------------------------
# Violin plots
image_numbers = np.tile(np.repeat(range(len(img_list)), len(pctls))+1, len(trials))
plot_path_violin = res_path / 'violinplot'
try:
    plot_path_violin.mkdir(parents=True)
except FileExistsError:
    pass

# Recall
recalls = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['recall'])
plt.figure(figsize=(13, 6))
ax = sns.violinplot(x=image_numbers, y='value', data=recalls, palette='colorblind',
            hue='cloud_cover', linewidth=0, fliersize=0)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:9], labels[0:9], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='Images', ylabel='Recall')
plt.savefig(plot_path_violin / 'recall.png', bbox_inches='tight')

# Precision
precisions = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['precision'])
plt.figure(figsize=(13, 6))
ax = sns.violinplot(x=image_numbers, y='value', data=precisions, palette='colorblind',
            hue='cloud_cover', linewidth=0, fliersize=0)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:9], labels[0:9], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='Images', ylabel='Precision')
plt.savefig(plot_path_violin / 'precision.png', bbox_inches='tight')

# Accuracy
accuracies = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['accuracy'])
plt.figure(figsize=(13, 6))
ax = sns.violinplot(x=image_numbers, y='value', data=accuracies, palette='colorblind',
            hue='cloud_cover', linewidth=0, fliersize=0)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:9], labels[0:9], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='Images', ylabel='Accuracy')
plt.savefig(plot_path_violin / 'accuracy.png', bbox_inches='tight')

# F1
f1s = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['f1'])
plt.figure(figsize=(13, 6))
ax = sns.violinplot(x=image_numbers, y='value', data=f1s, palette='colorblind',
            hue='cloud_cover', linewidth=0, fliersize=0, cut=0)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:9], labels[0:9], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='Images', ylabel='F1 Score')
plt.savefig(plot_path_violin / 'f1.png', bbox_inches='tight')

plt.close('all')

# ----------------------------------------------------------------------------------------------------------------------
# Mean of metric variances between trials
plot_path_vars = plot_path / 'variances'
try:
    plot_path_vars.mkdir(parents=True)
except FileExistsError:
    pass
df = pd.read_csv(res_path / 'trials_metrics.csv')
df.groupby(['image', 'cloud_cover']).var().groupby('image').mean()

# Group and reshape
var_group = df.groupby(['image', 'cloud_cover']).var().groupby('image').mean()
var_group.reset_index(level=0, inplace=True)
image_numbers = np.repeat(range(len(img_list)), 1) + 1
var_group = pd.concat([var_group, pd.DataFrame(image_numbers, columns=['image_numbers'])], axis=1)
var_long = pd.melt(var_group, id_vars=['image', 'image_numbers'], value_vars=['accuracy', 'recall', 'precision', 'f1'])

# Calculate mean values for each metric (i.e. mean mean variance)
acc_mean = var_group['accuracy'].mean()
recall_mean = var_group['recall'].mean()
precision_mean = var_group['precision'].mean()
f1_mean = var_group['f1'].mean()

# Add scatterplot
plt.figure(figsize=(10, 8))
ax = sns.scatterplot(x='image_numbers', y='value', hue='variable', data=var_long,
                     palette='colorblind', zorder=2, style='variable', s=50)

# Add mean lines to plot
palette = sns.color_palette('colorblind').as_hex()
ax.axhline(acc_mean, ls='--', zorder=1, color=palette[0])
ax.axhline(recall_mean, ls='--', zorder=1, color=palette[1])
ax.axhline(precision_mean, ls='--', zorder=1, color=palette[2])
ax.axhline(f1_mean, ls='--', zorder=1, color=palette[3])

plt.savefig(plot_path_vars / 'mean_variance.png', bbox_inches='tight')

# Separate mean variances
# plt.figure(figsize=(10, 8))
g = sns.FacetGrid(var_long, col='variable', hue='variable')
g.map(plt.scatter, "image_numbers", "value", alpha=.7, zorder=2)
for i, ax in enumerate(g.axes[0, :]):
    g.axes[0,i].set_xlabel(None)
g.axes[0, 0].set_xlabel('Image')
g.axes[0, 0].set_ylabel('Variance')


def plot_mean(data,**kwargs):
    m = data.mean()
    plt.axhline(m, **kwargs)

g.map(plot_mean, 'value', ls=":", c=".5", zorder=0)

g.fig.tight_layout(w_pad=1)
g.add_legend()

plt.savefig(plot_path_vars / 'mean_variance_separate.png', bbox_inches='tight')
# ----------------------------------------------------------------------------------------------------------------------
# Variances between cloud covers by individual metrics
df = pd.read_csv(res_path / 'trials_metrics.csv')
image_numbers = np.tile(np.repeat(range(len(img_list)), len(pctls))+1, 5)
df = pd.concat([df, pd.DataFrame(image_numbers, columns=['image_numbers'])], axis=1)
var_group = df.groupby(['image', 'image_numbers', 'cloud_cover']).var()
var_group.reset_index(inplace=True)
var_long = pd.melt(var_group, id_vars=['image_numbers', 'cloud_cover'],
                   value_vars=['accuracy', 'recall', 'precision', 'f1'])
plt.figure(figsize=(10, 8))
g = sns.FacetGrid(var_long, col='variable', hue='variable')
g.map(plt.scatter, "image_numbers", "value", alpha=.7)
g.add_legend()

plt.savefig(plot_path_vars / 'variance_separate.png', bbox_inches='tight')

# ======================================================================================================================
from numpy import genfromtxt

means_train = genfromtxt(data_path / 'experiments' / 'means_train.csv', delimiter=',')
variances_train = genfromtxt(data_path / 'experiments' / 'variances_train.csv', delimiter=',')
entropies_train = genfromtxt(data_path / 'experiments' / 'entropies_train.csv', delimiter=',')
means_test = genfromtxt(data_path / 'experiments' / 'means_test.csv', delimiter=',')
variances_test = genfromtxt(data_path / 'experiments' / 'variances_test.csv', delimiter=',')
entropies_test = genfromtxt(data_path / 'experiments' / 'entropies_test.csv', delimiter=',')

# Some features were removed from data because the std was 0 in either test or train,
# so need to get total number of feats
# Make arrays
data = pd.DataFrame({'image': np.repeat(img_list, len(pctls) * len(trials) * len(feat_list)),
                     'cloud_cover': np.repeat(pctls, len(img_list) * len(trials) * len(feat_list)),
                     'trial': np.tile(trials, len(pctls) * len(img_list) * len(feat_list)),
                     'feature': np.tile(feat_list, len(pctls) * len(img_list) * len(trials)),
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

g.savefig(plot_path_vars / 'metrics_vars.png', bbox_inches='tight')

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
#
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlabel('Principal Component 1', fontsize=15)
# ax.set_ylabel('Principal Component 2', fontsize=15)
#
# sns.scatterplot(x=pc_df['pc1'], y=pc_df['pc2'])
#
# # Get Mahalanobis distance between two random cloud test sets with high variance
# img_list[13] = '4444_LC08_044034_20170222_1' # had high accuracy variance for 30%

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
