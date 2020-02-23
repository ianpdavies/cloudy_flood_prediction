import __init__
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import rasterio
import pathlib

sys.path.append('../../')
from CPR.configs import data_path

# Version numbers
print('Python Version:', sys.version)

# ======================================================================================================================
# Performance metrics vs. image metadata (dry/flood pixels, image size)
pctls = [10, 30, 50, 70, 90]

img_list = os.listdir(data_path / 'images')
img_list.remove('4115_LC08_021033_20131227_test')

batch = 'RCTs'
trials = ['trial1', 'trial2', 'trial3', 'trial4', 'trial5', 'trial6', 'trial7', 'trial8', 'trial9', 'trial10']
exp_path = data_path / batch

try:
    exp_path.mkdir(parents=True)
except FileExistsError:
    pass

# ======================================================================================================================
# Create dataframe of all trial metrics
#
# clouds = []
# clouds_trial = []
#
# metrics = []
# metrics_trial = []
# for m, trial in enumerate(trials):
#     for img in img_list:
#         # Check for clouds
#         clouds_dir = data_path / 'clouds' / 'random' / trial
#         if not pathlib.Path(clouds_dir / '{0}'.format(img + '_clouds.npy')).exists():
#             clouds.append(img)
#             clouds_trial.append(trial)
#         # Check for metrics
#         metrics_path = data_path / batch / trial / 'metrics' / 'testing'
#         if not (metrics_path / img / 'metrics.csv').exists():
#             metrics.append(img)
#             metrics_trial.append(trial)
#
# results = pd.DataFrame({'clouds': clouds,
#                         'clouds_trial': clouds_trial})
# results.to_csv(data_path / batch / 'missing_clouds.csv')
#
# results = pd.DataFrame({'metrics': metrics,
#                         'metrics_trial': metrics_trial})
# results.to_csv(data_path / batch / 'missing_metrics.csv')
#
# df = pd.DataFrame(columns=['index', 'cloud_cover', 'accuracy', 'precision', 'recall', 'f1', 'index',
#                            'pixels', 'flood_pixels', 'dry_pixels', 'trial', 'image', 'flood_dry_ratio'])
# for m, trial in enumerate(trials):
#     print(trial)
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
#     metrics_path = data_path / batch / trial / 'metrics' / 'testing'
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
# df.to_csv(exp_path / 'results' / 'trials_metrics.csv', index=False)

# ======================================================================================================================
# Figures
df = pd.read_csv(exp_path / 'results' / 'trials_metrics.csv')

# ======================================================================================================================
# Boxplot of metrics with whiskers

image_numbers = np.tile(np.repeat(range(len(img_list)), len(pctls)) + 1, len(trials))
plot_path = exp_path / 'plots' / 'boxplot_whiskers'
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
    for j in range(i * 6, i * 6 + 6):
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
    for j in range(i * 6, i * 6 + 6):
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
    for j in range(i * 6, i * 6 + 6):
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
    for j in range(i * 6, i * 6 + 6):
        line = ax.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)
plt.savefig(plot_path / 'f1.png', bbox_inches='tight')

plt.close('all')
# ======================================================================================================================
# Boxplot of metrics with strip plot instead of whiskers

image_numbers = np.tile(np.repeat(range(len(img_list)), len(pctls)) + 1, len(trials))
plot_path = exp_path / 'plots' / 'boxplot_stripplot'
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
# ax.legend(labels=pctl_labels, loc='lower left', bbox_to_anchor=(0, 1),
#           ncol=5, borderaxespad=0, frameon=False, prop={'size': 7})
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

# ======================================================================================================================
# Violin plots
image_numbers = np.tile(np.repeat(range(len(img_list)), len(pctls)) + 1, len(trials))
plot_path = exp_path / 'plots' / 'violinplot'
try:
    plot_path.mkdir(parents=True)
except FileExistsError:
    pass

from matplotlib.patches import Patch

pctl_labels = ['Cloud cover', '10%', '30%', '50%', '70%', '90%']
dark2_colors = sns.color_palette("Dark2", 8)
color_inds = [0, 1, 2, 5, 7]
colors = []
for i in color_inds:
    colors.append(dark2_colors[i])

current_palette = sns.color_palette(colors)

legend_colors = colors.copy()
legend_colors.insert(0, (1.0, 1.0, 1.0))

legend_patches = [Patch(color=icolor, label=label)
                  for icolor, label in zip(legend_colors, pctl_labels)]

# Recall
recalls = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['recall'])
plt.figure(figsize=(13, 6))
ax = sns.violinplot(x=image_numbers, y='value', data=recalls, palette=current_palette,
                    hue='cloud_cover', linewidth=0, fliersize=0)
plt.legend(handles=legend_patches, labels=pctl_labels, loc='lower left', bbox_to_anchor=(-.04, 1), borderaxespad=0.,
           ncol=6, frameon=False)
ax.set(xlabel='Images', ylabel='Recall')
plt.tight_layout()
plt.savefig(plot_path / 'recall.png', bbox_inches='tight')

# Precision
precisions = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['precision'])
plt.figure(figsize=(13, 6))
ax = sns.violinplot(x=image_numbers, y='value', data=precisions, palette='colorblind',
                    hue='cloud_cover', linewidth=0, fliersize=0)
plt.legend(handles=legend_patches, labels=pctl_labels, loc='lower left', bbox_to_anchor=(-.04, 1), borderaxespad=0.,
           ncol=6, frameon=False)
ax.set(xlabel='Images', ylabel='Precision')
plt.savefig(plot_path / 'precision.png', bbox_inches='tight')

# Accuracy
accuracies = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['accuracy'])
plt.figure(figsize=(13, 6))
ax = sns.violinplot(x=image_numbers, y='value', data=accuracies, palette='colorblind',
                    hue='cloud_cover', linewidth=0, fliersize=0)
plt.legend(handles=legend_patches, labels=pctl_labels, loc='lower left', bbox_to_anchor=(-.04, 1), borderaxespad=0.,
           ncol=6, frameon=False)
ax.set(xlabel='Images', ylabel='Accuracy')
plt.savefig(plot_path / 'accuracy.png', bbox_inches='tight')

# F1
f1s = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['f1'])
plt.figure(figsize=(13, 6))
ax = sns.violinplot(x=image_numbers, y='value', data=f1s, palette='colorblind',
                    hue='cloud_cover', linewidth=0, fliersize=0, cut=0)
plt.legend(handles=legend_patches, labels=pctl_labels, loc='lower left', bbox_to_anchor=(-.04, 1), borderaxespad=0.,
           ncol=6, frameon=False)
ax.set(xlabel='Images', ylabel='F1 Score')
plt.savefig(plot_path / 'f1.png', bbox_inches='tight')

plt.close('all')


# ======================================================================================================================
# Violinplots in one big 4x1 matrix
accuracies = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['accuracy'])
f1s = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['f1'])
recalls = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['recall'])
precisions = pd.melt(df, id_vars=['image', 'cloud_cover'], value_vars=['precision'])

fig = plt.figure(figsize=(8, 11))
axes = [plt.subplot(4, 1, i + 1) for i in range(4)]
sns.violinplot(x=image_numbers, y='value', data=accuracies, palette=current_palette,
               hue='cloud_cover', linewidth=0, fliersize=0, ax=axes[0], zorder=2)
sns.violinplot(x=image_numbers, y='value', data=f1s, palette=current_palette,
               hue='cloud_cover', linewidth=0, fliersize=0, ax=axes[1], zorder=2)
sns.violinplot(x=image_numbers, y='value', data=recalls, palette=current_palette,
               hue='cloud_cover', linewidth=0, fliersize=0, ax=axes[2], zorder=2)
sns.violinplot(x=image_numbers, y='value', data=precisions, palette=current_palette,
               hue='cloud_cover', linewidth=0, fliersize=0, ax=axes[3], zorder=2)
# plt.legend(handles=legend_patches, labels=pctl_labels, loc='lower left', bbox_to_anchor=(-.04, 4.6), borderaxespad=0.,
#            ncol=6, frameon=False)

fancy_labels = ['Accuracy', 'F1', 'Recall', 'Precision']
for i in range(3):
    axes[i].get_legend().remove()
    axes[i].set_xticks([])

for i in range(4):
    axes[i].set_ylabel(fancy_labels[i], fontsize=10)
    axes[i].axhline(0.5, ls='--', zorder=1, color='grey', alpha=0.5)
    # axes[i].set_ylim([-0.5, 1.5])
    axes[i].set_ylim([0, 1])
    # axes[i].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axes[i].set_yticks([0.2, 0.4, 0.6, 0.8])

# axes[3].set_xticks([0, 30])
# axes[3].set_xticklabels([1, 31])
axes[3].set_xlabel('Images', fontsize=10)

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.02, top=0.95)
plt.legend(handles=legend_patches, labels=pctl_labels, loc='lower left', bbox_to_anchor=(-.04, 4.1), borderaxespad=0.,
           ncol=6, frameon=False)
plt.savefig(plot_path / 'violinplots.png', bbox_inches='tight')

# ======================================================================================================================
# Violin plots but all pctls are combined
dark2_colors = sns.color_palette("Dark2", 8)
color_inds = [0, 2, 5, 7]
colors = []
for j in range(4):
    for i in color_inds:
        colors.append(dark2_colors[i])

accuracies = pd.melt(df, id_vars=['image'], value_vars=['accuracy'])
f1s = pd.melt(df, id_vars=['image'], value_vars=['f1'])
recalls = pd.melt(df, id_vars=['image'], value_vars=['recall'])
precisions = pd.melt(df, id_vars=['image'], value_vars=['precision'])

fig = plt.figure(figsize=(8, 11))
axes = [plt.subplot(4, 1, i + 1) for i in range(4)]
sns.violinplot(x=image_numbers, y='value', data=accuracies, linewidth=0, fliersize=0, ax=axes[0], color=colors[0], zorder=2)
sns.violinplot(x=image_numbers, y='value', data=f1s, linewidth=0, fliersize=0, ax=axes[1], color=colors[1], zorder=2)
sns.violinplot(x=image_numbers, y='value', data=recalls, linewidth=0, fliersize=0, ax=axes[2], color=colors[2], zorder=2)
sns.violinplot(x=image_numbers, y='value', data=precisions,  linewidth=0, fliersize=0, ax=axes[3], color=colors[3], zorder=2)

fancy_labels = ['Accuracy', 'F1', 'Recall', 'Precision']
for i in range(3):
    axes[i].set_xticks([])

for i in range(4):
    axes[i].set_ylabel(fancy_labels[i], fontsize=10)
    # axes[i].set_ylim([0, 1])

axes[0].set_ylim([0, 1])
axes[3].set_xlabel('Images', fontsize=10)

plt.tight_layout()
plt.savefig(plot_path / 'violinplots_noCC.png', bbox_inches='tight')

# ======================================================================================================================
# Mean of metric variances between trials
# i.e. for an image and cloud cover, find variance between all trials, then average for each image

plot_path = exp_path / 'plots' / 'variances'
try:
    plot_path.mkdir(parents=True)
except FileExistsError:
    pass
df = pd.read_csv(exp_path / 'results' / 'trials_metrics.csv')
df.groupby(['image', 'cloud_cover']).var().groupby('image').mean()

metric_names_fancy = ['Accuracy', 'F1', 'Recall', 'Precision']
dark2_colors = sns.color_palette("Dark2", 8)
color_inds = [2, 5, 7, 0]
colors = []
for j in range(4):
    for i in color_inds:
        colors.append(dark2_colors[i])
sns.set_palette(colors)

# Group and reshape
var_group = df.groupby(['image', 'cloud_cover']).var().groupby('image').mean()
var_group.reset_index(level=0, inplace=True)
image_numbers = np.repeat(range(len(img_list)), 1) + 1
var_group = pd.concat([var_group, pd.DataFrame(image_numbers, columns=['image_numbers'])], axis=1)
var_long = pd.melt(var_group, id_vars=['image', 'image_numbers'], value_vars=['accuracy', 'f1', 'recall', 'precision'])

# Calculate mean values for each metric (i.e. mean mean variance)
acc_mean = var_group['accuracy'].mean()
recall_mean = var_group['recall'].mean()
precision_mean = var_group['precision'].mean()
f1_mean = var_group['f1'].mean()

# Add scatterplot
plt.figure(figsize=(10, 8))
ax = sns.scatterplot(x='image_numbers', y='value', hue='variable',
                     data=var_long, zorder=2, style='variable', s=50)

# Add mean lines to plot
palette = sns.color_palette().as_hex()
ax.axhline(acc_mean, ls='--', zorder=1, color=palette[0])
ax.axhline(recall_mean, ls='--', zorder=1, color=palette[1])
ax.axhline(precision_mean, ls='--', zorder=1, color=palette[2])
ax.axhline(f1_mean, ls='--', zorder=1, color=palette[3])
ax.set_xlabel('Image')
ax.set_ylabel('Mean variance')

plt.savefig(plot_path / 'mean_variance.png', bbox_inches='tight')

# ======================================================================================================================
# Separate mean variances
df = pd.read_csv(exp_path / 'results' / 'trials_metrics.csv')
df.groupby(['image', 'cloud_cover']).var().groupby('image').mean()
# Group and reshape
var_group = df.groupby(['image', 'cloud_cover']).var().groupby('image').mean()
var_group.reset_index(level=0, inplace=True)
image_numbers = np.repeat(range(len(img_list)), 1) + 1
var_group = pd.concat([var_group, pd.DataFrame(image_numbers, columns=['image_numbers'])], axis=1)
var_long = pd.melt(var_group, id_vars=['image', 'image_numbers'], value_vars=['accuracy', 'f1', 'recall', 'precision'])

g = sns.FacetGrid(var_long, col='variable', hue='variable')
g.map(plt.scatter, "image_numbers", "value", alpha=.8, zorder=2)


def plot_mean(data, **kwargs):
    m = data.mean()
    plt.axhline(m, **kwargs)


g.map(plot_mean, 'value', ls=":", c=".5", zorder=0, alpha=.9)

for i, ax in enumerate(g.axes.flat):
    g.set_xlabels('')
    # ax.set_xticks([])
    g.axes[0, i].set_title(metric_names_fancy[i], fontsize=10)
g.axes[0, 0].set_xlabel('Images')
g.axes[0, 0].set_ylabel('Mean variance')
g.fig.tight_layout(w_pad=1)

plt.savefig(plot_path / 'mean_variance_separate.png', bbox_inches='tight')

# ------------------------------------------------------------------------
# Variances between cloud covers by individual metrics
df = pd.read_csv(exp_path / 'results' / 'trials_metrics.csv')
image_numbers = np.tile(np.repeat(range(len(img_list)), len(pctls)) + 1, len(trials))
df = pd.concat([df, pd.DataFrame(image_numbers, columns=['image_numbers'])], axis=1)
var_group = df.groupby(['image', 'image_numbers', 'cloud_cover']).var()
var_group.reset_index(inplace=True)
var_long = pd.melt(var_group, id_vars=['image_numbers', 'cloud_cover'],
                   value_vars=['accuracy', 'f1', 'recall', 'precision'])
# plt.figure(figsize=(10, 8))
g = sns.FacetGrid(var_long, col='variable', hue='variable')
g.map(plt.scatter, "image_numbers", "value", alpha=.5)


def plot_mean(data, **kwargs):
    m = data.mean()
    plt.axhline(m, **kwargs)


g.map(plot_mean, 'value', ls=":", c=".5", zorder=0, alpha=.9)

for i, ax in enumerate(g.axes.flat):
    g.set_xlabels('')
    g.axes[0, i].set_title(metric_names_fancy[i], fontsize=10)
g.axes[0, 0].set_xlabel('Images')
g.axes[0, 0].set_ylabel('Variance')
g.fig.tight_layout(w_pad=1)

plt.savefig(plot_path / 'variance_separate.png', bbox_inches='tight')

# ======================================================================================================================
from numpy import genfromtxt

feat_list = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

first = genfromtxt(exp_path / 'results' / 'means_test1.csv', delimiter=',')
second = genfromtxt(exp_path / 'results' / 'means_test2.csv', delimiter=',')
second = second[second.shape[0]-12800:]
merged = np.concatenate((first, second), axis=0)
with open(exp_path / 'results' / 'means_test.csv', 'w') as f:
    np.savetxt(f, merged, delimiter=',')

first = genfromtxt(exp_path / 'results' / 'variances_test1.csv', delimiter=',')
second = genfromtxt(exp_path / 'results' / 'variances_test2.csv', delimiter=',')
second = second[second.shape[0]-12800:]
merged = np.concatenate((first, second), axis=0)
with open(exp_path / 'results' / 'variances_test.csv', 'w') as f:
    np.savetxt(f, merged, delimiter=',')

first = genfromtxt(exp_path / 'results' / 'entropies_test1.csv', delimiter=',')
second = genfromtxt(exp_path / 'results' / 'entropies_test2.csv', delimiter=',')
second = second[second.shape[0]-12800:]
merged = np.concatenate((first, second), axis=0)
with open(exp_path / 'results' / 'entropies_test.csv', 'w') as f:
    np.savetxt(f, merged, delimiter=',')

count = 0
feat_list_new = feat_list
for img in img_list[15:]:
    for trial in trials:
        for pctl in pctls:
            for i, feat in enumerate(feat_list_new):
                count += 1
print(count)



means_train = genfromtxt(exp_path / 'results' / 'means_train.csv', delimiter=',')
variances_train = genfromtxt(exp_path / 'results' / 'variances_train.csv', delimiter=',')
entropies_train = genfromtxt(exp_path / 'results' / 'entropies_train.csv', delimiter=',')
means_test = genfromtxt(exp_path / 'results' / 'means_test.csv', delimiter=',')
variances_test = genfromtxt(exp_path / 'results' / 'variances_test.csv', delimiter=',')
entropies_test = genfromtxt(exp_path / 'results' / 'entropies_test.csv', delimiter=',')

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
trial_metrics = pd.read_csv(exp_path / 'results' / 'trials_metrics.csv')

merge = pd.merge(data, trial_metrics, on=['image', 'trial', 'cloud_cover'])
merge['mean_diff'] = merge.mean_train - merge.mean_test
merge['variance_diff'] = merge.variance_train - merge.variance_test
merge['entropy_diff'] = merge.entropy_train - merge.entropy_test
merge['mean_diff2'] = np.sqrt(np.square(merge['mean_diff']))
merge['variance_diff2'] = np.sqrt(np.square(merge['variance_diff']))
merge['entropy_diff2'] = np.sqrt(np.square(merge['entropy_diff']))

merge = merge[merge['feature'] != 'GSW_perm']
merge = merge[merge['feature'] != 'flooded']
merge_subset = merge[merge['cloud_cover'] == 50]

g = sns.FacetGrid(merge_subset, col='feature', col_wrap=4, sharex=False, sharey=False)
g.map(sns.scatterplot, 'mean_diff', 'recall')

g = sns.FacetGrid(merge_subset, col='feature', col_wrap=4, sharex=False, sharey=False)
g.map(sns.scatterplot, 'variance_diff', 'recall')

g = sns.FacetGrid(merge_subset, col='feature', col_wrap=4, sharex=False, sharey=False)
g.map(sns.scatterplot, 'entropy_diff', 'recall')

g.savefig(exp_path / 'plots' / 'metrics_vars.png', bbox_inches='tight')

# MAKE SUBPLOTS AND ADD REG LINE
feat_list_fancy = ['Max SW extent', 'Dist from max SW', 'Elevation', 'Slope', 'HAND', 'SPI', 'TWI', 'Aspect',
                   'Curve', 'Developed', 'Forested', 'Other LULC', 'Planted',   'Wetlands']

feat_list_plot = ['GSW_maxExtent', 'GSW_distExtent', 'elevation', 'slope', 'hand', 'spi', 'twi',
                  'aspect', 'curve', 'developed', 'forest', 'other_landcover', 'planted',  'wetlands']

metric = 'precision'
stats = ['mean_diff2', 'variance_diff2', 'entropy_diff2']
for stat in stats:
    fig = plt.figure(figsize=(7.5, 8))
    axes = [plt.subplot(4, 4, i + 1) for i in range(14)]
    for i, ax in enumerate(axes):
        merge_subset_feat = merge_subset[merge_subset['feature'] == feat_list_plot[i]]
        x = merge_subset_feat[metric]
        y = merge_subset_feat[stat]
        ax.scatter(y, x, alpha=0.3, s=13)
        ax.set_title(feat_list_fancy[i], fontsize=10)
        ax.get_yaxis().set_visible(False)
        # sns.regplot(y, x, ax=ax)
    for i in range(0, 14, 4):
        axes[i].set_yticks([0, 0.5, 1.0])
    plt.suptitle(stat)
    axes[0].get_yaxis().set_visible(True)
    axes[0].set_ylabel(metric.capitalize())
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

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