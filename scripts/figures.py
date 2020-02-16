from CPR.utils import preprocessing
import matplotlib
import matplotlib.pyplot as plt
from CPR.configs import data_path
import rasterio
from matplotlib import gridspec
from rasterio.windows import Window
import numpy as np

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

figure_path = data_path / 'figures'
try:
    figure_path.mkdir(parents=True)
except FileExistsError:
    pass
# ======================================================================================================================
# Displaying all feature layers of an image
# Need to figure out how to reproject them to make them not tilted. Below is a solution where I just clip them instead,
# but it would be nice to show more of the image
img = '4050_LC08_023036_20130429_1'
batch = 'v2'
pctl = 50
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

# Read only a portion of the image
window = Window.from_slices((950, 1250), (1400, 1900))
with rasterio.open(stack_path, 'r') as src:
    w = src.read(window=window)
    w[w == -999999] = np.nan
    w[np.isneginf(w)] = np.nan

feat_list_fancy = ['Max SW extent', 'Dist from max SW', 'Aspect', 'Curve', 'Developed', 'Elevation', 'Forested',
                 'HAND', 'Other LULC', 'Planted', 'Slope', 'SPI', 'TWI', 'Wetlands', 'Permanent water', 'Flooded']

titles = feat_list_fancy
plt.figure(figsize=(6, 4))
axes = [plt.subplot(4, 4, i + 1) for i in range(16)]
for i, ax in enumerate(axes):
    ax.imshow(w[i])
    ax.set_title(titles[i], fontdict={'fontsize': 10, 'fontname': 'Helvetica'})
    ax.axis('off')
plt.tight_layout()
plt.savefig(figure_path / 'features.png', dpi=300)

# ======================================================================================================================
# Varying cloud cover image

pctls = [10, 30, 50, 70, 90]
cloud_masks = []
clouds_dir = data_path / 'clouds'
clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
for i, pctl in enumerate(pctls):
    cloud_masks.append(clouds > np.percentile(clouds, pctl))
titles = ['10%', '30%', '50%', '70%', '90%']

plt.figure(figsize=(13, 8))
gs1 = gridspec.GridSpec(1, 5)
gs1.update(wspace=0.1, hspace=0.25)  # set the spacing between axes.

blues_reversed = matplotlib.cm.get_cmap('Blues_r')

for i, gs in enumerate(gs1):
    ax = plt.subplot(gs1[i])
    ax.imshow(cloud_masks[i], cmap=blues_reversed)
    ax.set_title(titles[i], fontdict={'fontsize': 15})
    ax.axis('off')

# ======================================================================================================================
# Visualizing correlation matrices
import seaborn as sns
import pandas as pd

feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

col_names = ['Extent', 'Distance', 'Aspect', 'Curve', 'Develop', 'Elev', 'Forest', 'HAND', 'Other', 'Crop', 'Slope', 'SPI', 'TWI',
             'Wetland', 'Flood']

feat_list_fancy = ['Max SW extent', 'Dist from max SW', 'Aspect', 'Curve', 'Developed', 'Elevation', 'Forested',
                 'HAND', 'Other LULC', 'Planted', 'Slope', 'SPI', 'TWI', 'Wetlands', 'Flooded']

perm_index = feat_list_new.index('GSW_perm')

plt.figure()
cor_arr = np.load(data_path / 'corr_matrix_permzero.npy')
cor_arr = np.delete(cor_arr, perm_index, axis=1)
cor_arr = np.delete(cor_arr, perm_index, axis=0)
cor_df = pd.DataFrame(data=cor_arr, columns=col_names)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.zeros_like(cor_arr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
g = sns.heatmap(cor_df, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=col_names, yticklabels=col_names)
g.set_yticklabels(g.get_yticklabels(), rotation=0)
bottom, top = g.get_ylim()
g.set_ylim(bottom + 1, top - 1)
plt.tight_layout()
plt.savefig(figure_path / 'corr_matrix.png', dpi=300)

# ======================================================================================================================
# Median performance metrics of LR, RF, and NN in 4x3 matrix
import os
import pandas as pd
from CPR.configs import data_path
import seaborn as sns
import numpy as np

data_path = data_path
figure_path = data_path / 'figures'
try:
    figure_path.mkdir(parents=True)
except FileExistsError:
    pass

img_list = os.listdir(data_path / 'images')
img_list.remove('4115_LC08_021033_20131227_test')
metric_names = ['accuracy', 'precision', 'recall', 'f1']
metric_names_fancy = ['Accuracy', 'Precision', 'Recall', 'F1']
batches = ['LR', 'RF', 'NN']
batches_fancy = ['Logistic Regression', 'Random Forest', 'Neural Network']

dark2_colors = sns.color_palette("Dark2", 8)
color_inds = [0, 2, 5, 7]
colors = []
for j in range(4):
    for i in color_inds:
        colors.append(dark2_colors[i])

fig = plt.figure(figsize=(7.5, 9))
axes = [plt.subplot(4, 3, i + 1) for i in range(12)]

j = 0
batches = np.tile(batches, 4)
metric_names = np.tile(metric_names, 3)
metric_names_fancy = np.tile(metric_names_fancy, 3)
medians = []
for i, ax in enumerate(axes):
    batch = batches[i]
    if i in [0, 3, 6, 9]:
        metric = metric_names[j]
        j += 1
    for file in file_list:
        metrics_path = data_path / batch / 'metrics' / 'testing'
        file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
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
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

for i in range(0, 12, 3):
    axes[i].set_ylabel(metric_names_fancy[i], fontsize=10)
    axes[i].get_yaxis().set_visible(True)
    axes[i].yaxis.set_tick_params(labelsize=9)
    axes[i].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

for i in range(3):
    axes[i].set_title(batches_fancy[i], fontsize=10)

axes[9].set_xlabel('Cloud Cover')
plt.gcf().subplots_adjust(bottom=0.2)
axes[9].set_xticks([10, 30, 50, 70, 90])
axes[9].xaxis.set_tick_params(labelsize=9)
plt.ylabel(metric.capitalize())

medians = np.round(medians, 3)
for i, ax in enumerate(axes):
    ax.text(0.855, 0.1, str(medians[i]), horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, fontsize=9)


plt.savefig(figure_path / '{}'.format('median_highlights_all.png'), dpi=300)

# ======================================================================================================================
# Mean performance metrics of LR, RF, and NN in 4x3 matrix
fig = plt.figure(figsize=(7.5, 9))
axes = [plt.subplot(4, 3, i + 1) for i in range(12)]

j = 0
batches = np.tile(batches, 4)
metric_names = np.tile(metric_names, 3)
metric_names_fancy = np.tile(metric_names_fancy, 3)
means = []
for i, ax in enumerate(axes):
    batch = batches[i]
    if i in [0, 3, 6, 9]:
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
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

for i in range(0, 12, 3):
    axes[i].set_ylabel(metric_names_fancy[i], fontsize=10)
    axes[i].get_yaxis().set_visible(True)
    axes[i].yaxis.set_tick_params(labelsize=9)
    axes[i].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

for i in range(3):
    axes[i].set_title(batches_fancy[i], fontsize=10)

axes[9].set_xlabel('Cloud Cover')
plt.gcf().subplots_adjust(bottom=0.2)
axes[9].set_xticks([10, 30, 50, 70, 90])
axes[9].xaxis.set_tick_params(labelsize=9)
plt.ylabel(metric.capitalize())

means = np.round(means, 3)
for i, ax in enumerate(axes):
    ax.text(0.855, 0.1, str(means[i]), horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, fontsize=9)

plt.savefig(figure_path / '{}'.format('mean_highlights_all.png'), dpi=300)

