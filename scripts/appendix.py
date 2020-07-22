import __init__
from CPR.utils import preprocessing
import os
import matplotlib
import matplotlib.pyplot as plt
from CPR.configs import data_path
import rasterio
from matplotlib import gridspec
from rasterio.windows import Window
import numpy as np
from matplotlib.patches import Patch
from PIL import Image
from rasterio.windows import Window
import seaborn as sns
import pandas as pd

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

figure_path = data_path / 'figures'
try:
    figure_path.mkdir(parents=True)
except FileExistsError:
    pass

# ======================================================================================================================
# Median performance metrics of LR, RF, and NN in 5x2 matrix

data_path = data_path
figure_path = data_path / 'figures'
try:
    figure_path.mkdir(parents=True)
except FileExistsError:
    pass

img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]
metric_names = ['accuracy', 'f1', 'recall', 'precision', 'auc']
metric_names_fancy = ['Accuracy', 'F1', 'Recall', 'Precision', 'AUC']
batches = ['LR_perm_noGSW_1', 'LR_perm_noGSW_3']
plot_name = 'median_perm_water_trials_noGSW.png'
batches_fancy = ['(A)', '(B)']

dark2_colors = sns.color_palette("Dark2", 8)
color_inds = [0, 2, 5, 7, 1]
colors = []
for j in range(5):
    for i in color_inds:
        colors.append(dark2_colors[i])

fig = plt.figure(figsize=(5, 6))
axes = [plt.subplot(5, 2, i + 1) for i in range(10)]

j = 0
batches = np.tile(batches, 5)
metric_names = np.tile(metric_names, 2)
metric_names_fancy = np.tile(metric_names_fancy, 2)
medians = []
for i, ax in enumerate(axes):
    batch = batches[i]
    metrics_path = data_path / batch / 'metrics' / 'testing'
    file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
    if i in [0, 2, 4, 6, 8]:
        metric = metric_names[j]
        j += 1
    for file in file_list:
        df_concat = pd.concat(pd.read_csv(file) for file in file_list)
        median_metrics = df_concat.groupby('cloud_cover').median().reset_index()
        metrics = pd.read_csv(file)
        metrics.plot(x='cloud_cover', y=metric, ylim=(0, 1), ax=ax, color=colors[j], lw=1, alpha=0.3)
    median_metrics.plot(x='cloud_cover', y=metric, ylim=(0, 1), ax=ax, style='.-', color=colors[j], lw=2,
                        alpha=0.9)
    medians.append(df_concat[metric].median())
    ax.get_legend().remove()
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('')
    ax.set_xticks([])

j = 0
for i in range(0, 10, 2):
    axes[i].set_ylabel(metric_names_fancy[j], fontsize=9)
    axes[i].get_yaxis().set_visible(True)
    axes[i].yaxis.set_tick_params(labelsize=8)
    axes[i].set_ylim(0, 1)
    axes[i].set_yticks([0.25, 0.5, 0.75])
    j += 1

for i in range(3):
    axes[i].set_title(batches_fancy[i], fontsize=9)

axes[12].set_xlabel('Cloud Cover')
plt.gcf().subplots_adjust(bottom=0.2)
axes[9].set_xticks([10, 30, 50, 70, 90])
axes[9].xaxis.set_tick_params(labelsize=7)
plt.ylabel(metric.capitalize())

medians = np.round(medians, 3)
for i, ax in enumerate(axes):
    ax.text(0.855, 0.1, str(medians[i]), horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, fontsize=8)

plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.2)
plt.savefig(figure_path / '{}'.format(plot_name), dpi=300)

# ======================================================================================================================
# Mean performance metrics of perm water trials
plot_name = 'mean_perm_water_trials_noGSW.png'
fig = plt.figure(figsize=(5, 6))
axes = [plt.subplot(5, 2, i + 1) for i in range(10)]

j = 0
batches = np.tile(batches, 5)
metric_names = np.tile(metric_names, 2)
metric_names_fancy = np.tile(metric_names_fancy, 2)
means = []
for i, ax in enumerate(axes):
    batch = batches[i]
    metrics_path = data_path / batch / 'metrics' / 'testing'
    file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
    if i in [0, 2, 4, 6, 8]:
        metric = metric_names[j]
        j += 1
    for file in file_list:
        metrics_path = data_path / batch / 'metrics' / 'testing'
        file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
        df_concat = pd.concat(pd.read_csv(file) for file in file_list)
        mean_metrics = df_concat.groupby('cloud_cover').mean().reset_index()
        metrics = pd.read_csv(file)
        metrics.plot(x='cloud_cover', y=metric, ylim=(0, 1), ax=ax, color=colors[j], lw=1, alpha=0.3)
    mean_metrics.plot(x='cloud_cover', y=metric, ylim=(0, 1), ax=ax, style='.-', color=colors[j], lw=2,
                      alpha=0.9)
    means.append(df_concat[metric].mean())
    ax.get_legend().remove()
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('')
    ax.set_xticks([])

j = 0
for i in range(0, 10, 2):
    axes[i].set_ylabel(metric_names_fancy[j], fontsize=9)
    axes[i].get_yaxis().set_visible(True)
    axes[i].yaxis.set_tick_params(labelsize=8)
    axes[i].set_ylim(0, 1)
    axes[i].set_yticks([0.25, 0.5, 0.75])
    j += 1

for i in range(3):
    axes[i].set_title(batches_fancy[i], fontsize=9)

axes[12].set_xlabel('Cloud Cover')
plt.gcf().subplots_adjust(bottom=0.2)
axes[12].set_xticks([10, 30, 50, 70, 90])
axes[9].xaxis.set_tick_params(labelsize=8)
plt.ylabel(metric.capitalize())

means = np.round(means, 3)
for i, ax in enumerate(axes):
    ax.text(0.855, 0.1, str(means[i]), horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, fontsize=8)

plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.2)
plt.savefig(figure_path / '{}'.format(plot_name), dpi=300)

plt.close('all)')


# ======================================================================================================================
# # LR sample at only 30% CC
# import seaborn as sns
#
# plot_path = data_path / 'figures'
# metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
# metric_names_fancy = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
#
# dark2_colors = sns.color_palette("Dark2", 8)
# color_inds = [0, 1, 2, 5, 7]
# colors = []
# for i in color_inds:
#     colors.append(dark2_colors[i])
#
# try:
#     plot_path.mkdir(parents=True)
# except FileExistsError:
#     pass
#
# batches = ['LR_sample_1', 'LR_sample_2', 'LR_sample_3']
# metrics1 = []
# metrics2 = []
# metrics3 = []
# batch_metrics = []
# batch_names = pd.DataFrame(data=['(A)', '(B)', '(C)'], columns=['batch'])
# # Get mean values
# for batch in batches:
#     metrics_path = data_path / batch / 'metrics' / 'testing'
#     file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
#     df_concat = pd.concat(pd.read_csv(file) for file in file_list)
#     median_metrics = df_concat.groupby('cloud_cover').mean().reset_index()
#     batch_metrics.append(median_metrics)
#
#
# batch_metrics = pd.concat([batch_metrics[0], batch_metrics[1], batch_metrics[2]], axis=0)
# print(np.round(batch_metrics, 3))
#
# fig = plt.figure(figsize=(7, 2.5))
# axes = [plt.subplot(1, 4, i + 1) for i in range(4)]
# for i, ax in enumerate(axes):
#     metric = metric_names[i]
#     if i is 0:
#         for file in file_list:
#             metrics = pd.read_csv(file)
#             metrics.plot(x='cloud_cover', y=metric, ylim=(0, 1), ax=ax, color=colors[i], lw=1, alpha=0.3)
#         median_metrics.plot(x='cloud_cover', y=metric, ylim=(0, 1), ax=ax, style='.-', color=colors[i], lw=2,
#                             alpha=0.9)
#         ax.set_title(metric_names_fancy[i], fontsize=10)
#         ax.get_legend().remove()
#         ax.set_xlabel('Cloud Cover')
#         ax.set_xticks([10, 30, 50, 70, 90])
#     else:
#         for file in file_list:
#             metrics = pd.read_csv(file)
#             metrics.plot(x='cloud_cover', y=metric, ylim=(0, 1), ax=ax, color=colors[i], lw=1, alpha=0.3)
#         median_metrics.plot(x='cloud_cover', y=metric, ylim=(0, 1), ax=ax, style='.-', color=colors[i], lw=2,
#                             alpha=0.9)
#         ax.set_title(metric_names_fancy[i], fontsize=10)
#         ax.get_legend().remove()
#         ax.get_yaxis().set_visible(False)
#         ax.set_xlabel('')
#         ax.set_xticks([])
# plt.subplots_adjust(wspace=0.05, hspace=0.05)
# plt.gcf().subplots_adjust(bottom=0.2)
# plt.ylabel(metric.capitalize())
# plt.savefig(plot_path / '{}'.format('median_highlights.png'), dpi=300)

# ======================================================================================================================
# Plot RCT cloud masks
img = '4050_LC08_023036_20130429_1'
clouds_dir = data_path / 'clouds' / 'random'


trials = [1, 2, 3, 4, 5]
cloudmasks = []
pctl = 30
for trial in trials:
    clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds' + str(trial) + '.npy'))
    cloudmask = np.less(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))
    cloudmasks.append(cloudmask)

import matplotlib.pyplot as plt
axes = [plt.subplot(3, 2, i + 1) for i in range(5)]
for i, ax in enumerate(axes):
    ax.imshow(cloudmasks[i], cmap='Blues_r')
    ax.axis('off')
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)

import rasterio
img = '4514_LC08_027033_20170826_1'
stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'
with rasterio.open(stack_path, 'r') as ds:
    data = ds.read(ds.count-1)
    data[data==-999999] = np.nan
    data[np.isneginf(data)] = np.nan
    plt.imshow(data)