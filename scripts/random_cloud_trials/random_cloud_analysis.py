import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import sys

sys.path.append('../../')
from CPR.configs import data_path

# Version numbers
print('Python Version:', sys.version)

# ======================================================================================================================
# Performance metrics vs. image metadata (dry/flood pixels, image size)
pctls = [10, 20, 50, 70, 90]

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

batches = ['v13', 'v14', 'v15', 'v16', 'v17']
trials = ['trial1', 'trial2', 'trial3', 'trial4', 'trial5']
uncertainty = False
exp_path = data_path / 'experiments' / 'random'

# ======================================================================================================================
# Create dataframe of all trial metrics

# df = pd.DataFrame(columns=['index', 'cloud_cover', 'accuracy', 'precision', 'recall', 'f1', 'index',
#                            'pixels', 'flood_pixels', 'dry_pixels', 'trial', 'image', 'flood_dry_ratio'])
# for m, trial in enumerate(trials):
#     stack_list = [data_path / 'images' / img / 'stack' / 'stack.tif' for img in img_list]
#     pixel_counts = []
#     flood_counts = []
#     dry_counts = []
#     imgs = []
#
#     # Get pixel metadata for each image + cloud cover %
#     for j, stack in enumerate(stack_list):
#         print('Getting pixel count of', img_list[j])
#         with rasterio.open(stack, 'r') as ds:
#             img = ds.read(ds.count)
#             img[img == -999999] = np.nan
#             img[np.isneginf(img)] = np.nan
#             clouds_dir = data_path / 'clouds' / 'random' / trial
#             clouds = np.load(clouds_dir / '{0}'.format(img_list[j] + '_clouds.npy'))
#             for k, pctl in enumerate(pctls):
#                 cloud_mask = clouds.copy()
#                 cloud_mask = cloud_mask < np.percentile(cloud_mask, pctl)
#                 img_pixels = np.count_nonzero(~np.isnan(img))
#                 img[cloud_mask] = np.nan
#                 pixel_count = np.count_nonzero(~np.isnan(img))
#                 pixel_counts.append(pixel_count)
#                 flood_count = np.sum(img[~np.isnan(img)])
#                 flood_counts.append(flood_count)
#                 dry_count = pixel_count - flood_count
#                 dry_counts.append(dry_count)
#                 imgs.append(img_list[j])
#
#     metadata = np.column_stack([pixel_counts, flood_counts, dry_counts, np.repeat(trial, len(pixel_counts))])
#     metadata = pd.DataFrame(metadata, columns=['pixels', 'flood_pixels', 'dry_pixels', 'trial'])
#     metadata['pixels'] = metadata['pixels'].astype('float32').astype('int32')
#     metadata['flood_pixels'] = metadata['flood_pixels'].astype('float32').astype('int32')
#     metadata['dry_pixels'] = metadata['dry_pixels'].astype('float32').astype('int32')
#     metadata['trial'] = metadata['trial'].astype('str')
#     imgs_df = pd.DataFrame(imgs, columns=['image'])
#     metadata = pd.concat([metadata, imgs_df], axis=1)
#
#     print('Fetching performance metrics')
#     metrics_path = data_path / batches[m] / 'metrics' / 'testing_nn'
#     plot_path = data_path / batches[m] / 'plots'
#
#     file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
#     metrics = pd.concat(pd.read_csv(file) for file in file_list)
#     data = pd.concat([metrics.reset_index(), metadata.reset_index()], axis=1)
#     data['flood_dry_ratio'] = data['flood_pixels'] / data['dry_pixels']
#     df = df.append(data)
#
# # Save dataframe of all random trial metrics
# try:
#     exp_path.mkdir(parents=True)
# except FileExistsError:
#     pass
# df = df.drop('index', axis=1)
# df.to_csv(exp_path / 'trials_metrics.csv', index=False)

# ======================================================================================================================
# Figures
df = pd.read_csv(exp_path / 'trials_metrics.csv')

# ----------------------------------------------------------------------------------------------------------------------
# Boxplot of metrics with whiskers

image_numbers = np.tile(np.repeat(range(22), len(pctls))+1, 5)
plot_path = exp_path / 'boxplot_whiskers'
try:
    plot_path.mkdir(parents=True)
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
plt.savefig(plot_path / 'recall.png', bbox_inches='tight')

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
plt.savefig(plot_path / 'precision.png', bbox_inches='tight')

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
plt.savefig(plot_path / 'accuracy.png', bbox_inches='tight')

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
plt.savefig(plot_path / 'f1.png', bbox_inches='tight')

plt.close('all')
# ----------------------------------------------------------------------------------------------------------------------
# Boxplot of metrics with strip plot instead of whiskers

image_numbers = np.tile(np.repeat(range(22), len(pctls))+1, 5)
plot_path = exp_path / 'boxplot_stripplot'
try:
    plot_path.mkdir(parents=True)
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
plt.savefig(plot_path / 'recall.png', bbox_inches='tight')

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
plt.savefig(plot_path / 'precision.png', bbox_inches='tight')

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
plt.savefig(plot_path / 'accuracy.png', bbox_inches='tight')

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
plt.savefig(plot_path / 'f1.png', bbox_inches='tight')

plt.close('all')

# ----------------------------------------------------------------------------------------------------------------------
# Violin plots
image_numbers = np.tile(np.repeat(range(22), len(pctls))+1, 5)
plot_path = exp_path / 'violinplot'
try:
    plot_path.mkdir(parents=True)
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
plt.savefig(plot_path / 'recall.png', bbox_inches='tight')

# Precision
precisions = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['precision'])
plt.figure(figsize=(13, 6))
ax = sns.violinplot(x=image_numbers, y='value', data=precisions, palette='colorblind',
            hue='cloud_cover', linewidth=0, fliersize=0)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:9], labels[0:9], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='Images', ylabel='Precision')
plt.savefig(plot_path / 'precision.png', bbox_inches='tight')

# Accuracy
accuracies = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['accuracy'])
plt.figure(figsize=(13, 6))
ax = sns.violinplot(x=image_numbers, y='value', data=accuracies, palette='colorblind',
            hue='cloud_cover', linewidth=0, fliersize=0)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:9], labels[0:9], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='Images', ylabel='Accuracy')
plt.savefig(plot_path / 'accuracy.png', bbox_inches='tight')

# F1
f1s = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['f1'])
plt.figure(figsize=(13, 6))
ax = sns.violinplot(x=image_numbers, y='value', data=f1s, palette='colorblind',
            hue='cloud_cover', linewidth=0, fliersize=0, cut=0)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:9], labels[0:9], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='Images', ylabel='F1 Score')
plt.savefig(plot_path / 'f1.png', bbox_inches='tight')

plt.close('all')

# ----------------------------------------------------------------------------------------------------------------------
# Mean of metric variances between trials
plot_path = exp_path / 'variances'
try:
    plot_path.mkdir(parents=True)
except FileExistsError:
    pass
df = pd.read_csv(exp_path / 'trials_metrics.csv')
df.groupby(['image', 'cloud_cover']).var().groupby('image').mean()

# Group and reshape
var_group = df.groupby(['image', 'cloud_cover']).var().groupby('image').mean()
var_group.reset_index(level=0, inplace=True)
image_numbers = np.repeat(range(22), 1) + 1
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

plt.savefig(plot_path / 'mean_variance.png', bbox_inches='tight')

# Separate mean variances
plt.figure(figsize=(10, 8))
g = sns.FacetGrid(var_long, col='variable', hue='variable')
g.map(plt.scatter, "image_numbers", "value", alpha=.7, zorder=2)
g.axes.set_xlabel('Image')

def plot_mean(data,**kwargs):
    m = data.mean()
    plt.axhline(m, **kwargs)

g.map(plot_mean, 'value', ls=":", c=".5", zorder=0)

g.fig.tight_layout(w_pad=1)
g.add_legend()

plt.savefig(plot_path / 'mean_variance_separate.png', bbox_inches='tight')
# ----------------------------------------------------------------------------------------------------------------------
# Variances between cloud covers by individual metrics
df = pd.read_csv(exp_path / 'trials_metrics.csv')
image_numbers = np.tile(np.repeat(range(22), len(pctls))+1, 5)
df = pd.concat([df, pd.DataFrame(image_numbers, columns=['image_numbers'])], axis=1)
var_group = df.groupby(['image', 'image_numbers', 'cloud_cover']).var()
var_group.reset_index(inplace=True)
var_long = pd.melt(var_group, id_vars=['image_numbers', 'cloud_cover'],
                   value_vars=['accuracy', 'recall', 'precision', 'f1'])
plt.figure(figsize=(10, 8))
g = sns.FacetGrid(var_long, col='variable', hue='variable')
g.map(plt.scatter, "image_numbers", "value", alpha=.7)
g.add_legend()

plt.savefig(plot_path / 'variance_separate.png', bbox_inches='tight')

# ======================================================================================================================


