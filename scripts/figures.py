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
    w[15, ((w[14, :, :] == 1) & (w[15, :, :] == 1))] = 0
    w = w[1:, :, :]

feat_list_fancy = ['Dist from Seasonal', 'Aspect', 'Curve', 'Developed', 'Elevation', 'Forested',
                   'HAND', 'Other LULC', 'Planted', 'Slope', 'SPI', 'TWI', 'Wetlands', 'Permanent water', 'Flooded']

titles = feat_list_fancy
plt.figure(figsize=(6, 4))
axes = [plt.subplot(4, 4, i + 1) for i in range(15)]
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
# Visualizing correlation matrices, perm water = 0

feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

col_names = ['Distance', 'Aspect', 'Curve', 'Develop', 'Elev', 'Forest', 'HAND', 'Other', 'Crop', 'Slope',
             'SPI', 'TWI',
             'Wetland', 'Flood']

feat_list_fancy = ['Dist perm', 'Aspect', 'Curve', 'Developed', 'Elevation', 'Forested',
                   'HAND', 'Other LULC', 'Planted', 'Slope', 'SPI', 'TWI', 'Wetlands', 'Flooded']

perm_index = feat_list_new.index('GSW_perm')
gsw_index = feat_list_new.index('GSW_maxExtent')

plt.figure()
cor_arr = np.load(data_path / 'corr_matrix_permzero.npy')
cor_arr = np.delete(cor_arr, perm_index, axis=1)
cor_arr = np.delete(cor_arr, perm_index, axis=0)
cor_arr = np.delete(cor_arr, gsw_index, axis=1)
cor_arr = np.delete(cor_arr, gsw_index, axis=0)
cor_df = pd.DataFrame(data=cor_arr, columns=col_names)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.zeros_like(cor_arr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
g = sns.heatmap(cor_df, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                # xticklabels=col_names, yticklabels=col_names)
                xticklabels=col_names, yticklabels=col_names, annot=True, annot_kws={"size": 5})
g.set_yticklabels(g.get_yticklabels(), rotation=0)
bottom, top = g.get_ylim()
g.set_ylim(bottom + 1, top - 1)
plt.tight_layout()
plt.savefig(figure_path / 'corr_matrix_annot.png', dpi=300)

# Correlation matrix with perm water included in flood feature, and also as a column

feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

col_names = ['Max Extent', 'Distance', 'Aspect', 'Curve', 'Develop', 'Elev', 'Forest', 'HAND', 'Other', 'Crop', 'Slope',
             'SPI', 'TWI',
             'Wetland', 'Perm Water', 'Flood']

feat_list_fancy = ['Max SW extent', 'Dist max SW', 'Aspect', 'Curve', 'Developed', 'Elevation', 'Forested',
                   'HAND', 'Other LULC', 'Planted', 'Slope', 'SPI', 'TWI', 'Wetlands', 'Perm Water', 'Flooded']

perm_index = feat_list_new.index('GSW_perm')

plt.figure()
cor_arr = np.load(data_path / 'corr_matrix.npy')
cor_df = pd.DataFrame(data=cor_arr, columns=col_names)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.zeros_like(cor_arr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
g = sns.heatmap(cor_df, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=col_names, yticklabels=col_names, annot=True, annot_kws={"size": 5})
g.set_yticklabels(g.get_yticklabels(), rotation=0)
bottom, top = g.get_ylim()
g.set_ylim(bottom + 1, top - 1)
plt.tight_layout()

plt.savefig(figure_path / 'corr_matrix_allwater_annot.png', dpi=300)
# ======================================================================================================================
# Median performance metrics of LR, RF, and NN in 5x3 matrix

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
batches = ['LR_noGSW', 'RF_noGSW', 'NN_noGSW']
plot_name = 'median_highlights_noGSW.png'
batches_fancy = ['Logistic Regression', 'Random Forest', 'Neural Network']

dark2_colors = sns.color_palette("Dark2", 8)
color_inds = [0, 2, 5, 7, 1]
colors = []
for j in range(5):
    for i in color_inds:
        colors.append(dark2_colors[i])

fig = plt.figure(figsize=(6.5, 6))
axes = [plt.subplot(5, 3, i + 1) for i in range(15)]

j = 0
batches = np.tile(batches, 5)
metric_names = np.tile(metric_names, 3)
metric_names_fancy = np.tile(metric_names_fancy, 3)
medians = []
for i, ax in enumerate(axes):
    batch = batches[i]
    metrics_path = data_path / batch / 'metrics' / 'testing'
    file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
    if i in [0, 3, 6, 9, 12]:
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
for i in range(0, 15, 3):
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
# Mean performance metrics of LR, RF, and NN in 5x3 matrix
plot_name = 'mean_highlights_noGSW.png'
fig = plt.figure(figsize=(6.5, 6))
axes = [plt.subplot(5, 3, i + 1) for i in range(15)]

j = 0
batches = np.tile(batches, 5)
metric_names = np.tile(metric_names, 3)
metric_names_fancy = np.tile(metric_names_fancy, 3)
means = []
for i, ax in enumerate(axes):
    batch = batches[i]
    metrics_path = data_path / batch / 'metrics' / 'testing'
    file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
    if i in [0, 3, 6, 9, 12]:
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
for i in range(0, 15, 3):
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
# Plotting BNN and LR uncertainty histograms side by side

set1_colors = sns.color_palette("Set1", 8)
color_inds = [1, 2, 0]
colors = []
for j in range(3):
    for i in color_inds:
        colors.append(set1_colors[i])

class_labels = ['True Positive', 'False Positive', 'False Negative']
legend_patches = [Patch(color=icolor, label=label)
                  for icolor, label in zip(colors, class_labels)]

# Load uncertainties
bnn_uncertainty = pd.read_csv(data_path / 'BNN_kwon_noGSW' / 'metrics' / 'uncertainty_binned_noGSW.csv')
lr_uncertainty = pd.read_csv(data_path / 'LR_noGSW' / 'metrics' / 'uncertainty_binned_noGSW.csv')
fig = plt.figure(figsize=(8, 4))
axes = [plt.subplot(1, 2, i + 1) for i in range(2)]
binned_dfs = [bnn_uncertainty, lr_uncertainty]
for i, ax in enumerate(axes):
    df_group = binned_dfs[i]
    w = np.min(np.diff(df_group['bins_float'])) / 3
    ax.bar(df_group['bins_float'] - w, df_group['tp'], width=w, color=colors[0], align='center')
    ax.bar(df_group['bins_float'], df_group['fp'], width=w, color=colors[1], align='center')
    ax.bar(df_group['bins_float'] + w, df_group['fn'], width=w, color=colors[2], align='center')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True, useOffset=True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.autoscale(tight=True)
    ax.yaxis.offsetText.set_fontsize(SMALL_SIZE)
    ax.xaxis.offsetText.set_fontsize(SMALL_SIZE)

axes[0].legend(labels=class_labels, handles=legend_patches, loc='lower left', bbox_to_anchor=(0.1, 1),
               ncol=5, borderaxespad=0, frameon=False, prop={'size': 7})
axes[0].set_xlabel('Aleatoric + Epistemic Uncertainty')
axes[1].set_xlabel('Confidence Interval')
axes[1].get_yaxis().set_visible(False)
axes[0].set_ylabel('Prediction Type / Total')
plt.tight_layout()
plt.subplots_adjust(wspace=0.05)
plt.savefig(figure_path / 'uncertainty_histograms.png', dpi=300)

# ======================================================================================================================
# MODEL MAP COMPARISON GLOBAL PLOT PARAMETERS

colors = []
white = colors.append(matplotlib.colors.to_rgb(matplotlib.colors.cnames['white']))
blue = colors.append(matplotlib.colors.to_rgb(matplotlib.colors.cnames['blue']))
red = colors.append(matplotlib.colors.to_rgb(matplotlib.colors.cnames['red']))
yellow = colors.append(matplotlib.colors.to_rgb(matplotlib.colors.cnames['yellow']))
grey = colors.append((174 / 255, 236 / 255, 238 / 255, 255 / 255))
black = colors.append(matplotlib.colors.to_rgb(matplotlib.colors.cnames['black']))

class_labels = ['TP', 'FP', 'FN', 'Cloud Border', 'Visible Flood', 'QA Mask']

legend_patches = [Patch(facecolor=icolor, label=label, edgecolor='black')
                  for icolor, label in zip(colors, class_labels)]

# ------------------------------------------------------------------------------
# Comparing model predictions with maps
img = '4444_LC08_044033_20170222_4'
pctl = 30
y_top = 1700
x_left = 1570
y_bottom = 2100
x_right = 2300

batches = ['LR_allwater', 'LR_allwater', 'LR_allwater']
fancy_batches = ['Logistic Regression', 'Random Forest', 'Neural Network']

fig = plt.figure(figsize=(8, 4))
axes = [plt.subplot(1, 3, i + 1) for i in range(3)]
for i, batch in enumerate(batches):
    falsemap_file = data_path / batch / 'plots' / img / '{}'.format('false_map_border' + str(pctl) + '.png')
    falsemap_img = Image.open(falsemap_file)
    axes[i].imshow(falsemap_img.crop((x_left, y_top, x_right, y_bottom)))
    axes[i].axis('off')

axes[0].legend(labels=class_labels, handles=legend_patches, loc='lower left', bbox_to_anchor=(0, 1),
               ncol=6, borderaxespad=0.5, frameon=False, prop={'size': 7})

plt.subplots_adjust(wspace=0.05)

# ------------------------------------------------------------------------------
# Comparing model predictions with maps
img = '4468_LC08_024036_20170501_2'
pctl = 70
y_top = 750
x_left = 650
y_bottom = 1150
x_right = 1380

batches = ['LR_allwater', 'LR_allwater', 'LR_allwater']
fancy_batches = ['Logistic Regression', 'Random Forest', 'Neural Network']

fig = plt.figure(figsize=(8, 4))
axes = [plt.subplot(1, 3, i + 1) for i in range(3)]
for i, batch in enumerate(batches):
    falsemap_file = data_path / batch / 'plots' / img / '{}'.format('false_map_border' + str(pctl) + '.png')
    falsemap_img = Image.open(falsemap_file)
    axes[i].imshow(falsemap_img.crop((x_left, y_top, x_right, y_bottom)))
    axes[i].axis('off')

axes[0].legend(labels=class_labels, handles=legend_patches, loc='lower left', bbox_to_anchor=(0, 1),
               ncol=6, borderaxespad=0.5, frameon=False, prop={'size': 7})

plt.subplots_adjust(wspace=0.05)

# ------------------------------------------------------------------------------
# Comparing model predictions with maps
img = '4115_LC08_021033_20131227_1'
pctl = 90
y_top = 1070
x_left = 3300
y_bottom = 1470
x_right = 4030

batches = ['LR_allwater', 'LR_allwater', 'LR_allwater']
fancy_batches = ['Logistic Regression', 'Random Forest', 'Neural Network']

fig = plt.figure(figsize=(8, 4))
axes = [plt.subplot(1, 3, i + 1) for i in range(3)]
for i, batch in enumerate(batches):
    falsemap_file = data_path / batch / 'plots' / img / '{}'.format('false_map_border' + str(pctl) + '.png')
    falsemap_img = Image.open(falsemap_file)
    axes[i].imshow(falsemap_img.crop((x_left, y_top, x_right, y_bottom)))
    axes[i].axis('off')

axes[0].legend(labels=class_labels, handles=legend_patches, loc='lower left', bbox_to_anchor=(0, 1),
               ncol=6, borderaxespad=0.5, frameon=False, prop={'size': 7})

plt.subplots_adjust(wspace=0.05)

# ------------------------------------------------------------------------------
# Comparing model predictions with maps
img = '4444_LC08_045032_20170301_1'
pctl = 90
y_top = 2850
x_left = 1630
y_bottom = 3250
x_right = 2360

batches = ['LR_allwater', 'LR_allwater', 'LR_allwater']
fancy_batches = ['Logistic Regression', 'Random Forest', 'Neural Network']

fig = plt.figure(figsize=(8, 4))
axes = [plt.subplot(1, 3, i + 1) for i in range(3)]
for i, batch in enumerate(batches):
    falsemap_file = data_path / batch / 'plots' / img / '{}'.format('false_map_border' + str(pctl) + '.png')
    falsemap_img = Image.open(falsemap_file)
    axes[i].imshow(falsemap_img.crop((x_left, y_top, x_right, y_bottom)))
    axes[i].axis('off')

axes[0].legend(labels=class_labels, handles=legend_patches, loc='lower left', bbox_to_anchor=(0, 1),
               ncol=6, borderaxespad=0.5, frameon=False, prop={'size': 7})

plt.subplots_adjust(wspace=0.05)

# All together
y_tops = np.repeat([1700, 750, 970, 2850], 3)
x_lefts = np.repeat([1570, 650, 3100, 1630], 3)
y_bottoms = np.repeat([2100, 1150, 1370, 3250], 3)
x_rights = np.repeat([2300, 1380, 3830, 2360], 3)
pctls = np.repeat([90, 70, 90, 90], 3)
img_list = np.repeat(['4444_LC08_044033_20170222_4', '4468_LC08_024036_20170501_2',
                      '4115_LC08_021033_20131227_1', '4444_LC08_045032_20170301_1'], 3)

batches = np.tile(['LR_allwater', 'RF_allwater', 'NN_allwater'], 4)
batches_fancy = ['Logistic Regression', 'Random Forest', 'Neural Network']

fig = plt.figure(figsize=(7.5, 6))
axes = [plt.subplot(4, 3, i + 1) for i in range(12)]

for i, ax in enumerate(axes):
    falsemap_file = data_path / batches[i] / 'plots' / img_list[i] / '{}'.format(
        'false_map_border' + str(pctls[i]) + '.png')
    falsemap_img = Image.open(falsemap_file)
    axes[i].imshow(falsemap_img.crop((x_lefts[i], y_tops[i], x_rights[i], y_bottoms[i])))
    batch = batches[i]
    # axes[i].axis('off')
    axes[i].get_yaxis().set_visible(False)
    axes[i].get_xaxis().set_visible(False)
    axes[i].axis('off')

# for i in range(3):
#     axes[i].set_title(batches_fancy[i], fontsize=10)

axes[0].legend(labels=class_labels, handles=legend_patches, loc='lower left', bbox_to_anchor=(0, 1),
               ncol=6, borderaxespad=0.5, frameon=False, prop={'size': 7})

# plt.gcf().subplots_adjust(bottom=0.2)

plt.tight_layout()
plt.subplots_adjust(wspace=0.03, hspace=0.1)  # top=0.98, bottom=0.15)
plt.savefig(figure_path / '{}'.format('comparison_maps.png'), dpi=300)

# plt.close('all)')


# ------------------------------------------------------------------------------
# Comparing RCT images
img = '4444_LC08_044033_20170222_4'
pctl = 90
y_top = 1700
x_left = 1650
y_bottom = 2100
x_right = 2300

batch = 'RCTs'
trials = ['trial2', 'trial3', 'trial4']

fig = plt.figure(figsize=(8, 2.5))
axes = [plt.subplot(1, 3, i + 1) for i in range(3)]
for i, trial in enumerate(trials):
    falsemap_file = data_path / batch / trial / 'plots' / img / '{}'.format('false_map_border' + str(pctl) + '.png')
    falsemap_img = Image.open(falsemap_file)
    axes[i].imshow(falsemap_img.crop((x_left, y_top, x_right, y_bottom)))
    axes[i].axis('off')

axes[0].legend(labels=class_labels, handles=legend_patches, loc='lower left', bbox_to_anchor=(0, 1),
               ncol=6, borderaxespad=0.5, frameon=False, prop={'size': 7})

plt.tight_layout()
plt.subplots_adjust(wspace=0.05)
plt.savefig(figure_path / '{}'.format('comparison_maps_RCTs.png'), dpi=400)

# ------------------------------------------------------------------------------
# plot imagses
plt.figure()
plt.imshow(Image.open(data_path / 'LR_allwater' / 'plots' / '4444_LC08_043034_20170303_1' / 'map_fpfn_90.png'))
plt.figure()
plt.imshow(Image.open(data_path / 'LR_allwater' / 'plots' / '4444_LC08_043034_20170303_1' / 'map_uncertainty_90.png'))

# Comparing uncertainty maps
y_tops = np.repeat([690], 2)
x_lefts = np.repeat([810], 2)
y_bottoms = np.repeat([900], 2)
x_rights = np.repeat([1450], 2)
pctls = np.repeat([90], 2)
img_list = np.repeat(['4444_LC08_043034_20170303_1'], 2)

batches = ['LR_allwater', 'BNN_kwon']
batches_fancy = ['Logistic Regression', 'Bayesian Neural Network']

fpfn_colors = ['white', 'saddlebrown', 'limegreen', 'red', 'blue']
fpfn_labels = ['Training Data', 'TN', 'FP', 'FN', 'TP']

legend_patches = [Patch(facecolor=icolor, label=label, edgecolor='black')
                  for icolor, label in zip(fpfn_colors, fpfn_labels)]

fig = plt.figure(figsize=(7.5, 6))
axes = [plt.subplot(2, 2, i + 1) for i in range(4)]

for i, ax in enumerate(axes[:2]):
    falsemap_file = data_path / batches[i] / 'plots' / img_list[i] / '{}'.format(
        'map_fpfn_' + str(pctls[i]) + '.png')
    falsemap_img = Image.open(falsemap_file)
    ax.imshow(falsemap_img.crop((x_lefts[i], y_tops[i], x_rights[i], y_bottoms[i])))
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.axis('off')

y_tops = np.repeat([605], 2)
x_lefts = np.repeat([460], 2)
y_bottoms = np.repeat([830], 2)
x_rights = np.repeat([1040], 2)

for i, ax in enumerate(axes[2:]):
    falsemap_file = data_path / batches[i] / 'plots' / img_list[i] / '{}'.format(
        'map_uncertainty_' + str(pctls[i]) + '.png')
    falsemap_img = Image.open(falsemap_file)
    ax.imshow(falsemap_img.crop((x_lefts[i], y_tops[i], x_rights[i], y_bottoms[i])))
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.axis('off')

# for i in range(3):
#     axes[i].set_title(batches_fancy[i], fontsize=10)

import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

axes[0].legend(labels=fpfn_labels, handles=legend_patches, loc='lower left', bbox_to_anchor=(0, 1),
               ncol=5, borderaxespad=0.5, frameon=False, prop={'size': 11})

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = matplotlib.cm.plasma
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1.2))
plt.colorbar(sm, orientation='horizontal', aspect=50)

axins = inset_axes(ax2,
                   width="5%",  # width = 5% of parent_bbox width
                   height="50%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=ax2.transAxes,
                   borderpad=0,
                   )
fig.colorbar(sm, cax=axins)
plt.tight_layout()
plt.subplots_adjust(wspace=0.03, hspace=0.1)  # top=0.98, bottom=0.15)
plt.savefig(figure_path / '{}'.format('comparison_maps.png'), dpi=300)

# plt.close('all)')


# ===================================================================================================================

metrics_path = data_path / 'LR_allwater' / 'metrics' / 'testing'
file_list = [metrics_path / img / 'metrics.csv' for img in img_list[:10]]
df_concat = pd.concat(pd.read_csv(file) for file in file_list)
mean_metrics = df_concat.mean().reset_index()
print(mean_metrics)

file_list = [metrics_path / img / 'metrics.csv' for img in img_list[10:]]
df_concat = pd.concat(pd.read_csv(file) for file in file_list)
mean_metrics = df_concat.mean().reset_index()
print(mean_metrics)

# ======================================================================================================================
# Create KDEs for presentation

# Normal distrib kde
from scipy.stats import norm

mu = 998.8
sigma = 73.10
x1 = 900
x2 = 1100

z1 = (x1 - mu) / sigma
z2 = (x2 - mu) / sigma

x = np.arange(z1, z2, 0.001)  # range of x in spec
x_all = np.arange(-10, 10, 0.001)  # entire range of x, both in and out of spec
# mean = 0, stddev = 1, since Z-transform was calculated
y = norm.pdf(x, 0, 1)
y2 = norm.pdf(x_all, 0, 1)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x_all, y2)

ax.set_xlim([-4, 4])
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
[s.set_visible(False) for s in ax.spines.values()]
[t.set_visible(False) for t in ax.get_xticklines()]
[t.set_visible(False) for t in ax.get_yticklines()]
ax.fill_between(x_all, y2, 0, alpha=0.3)
plt.savefig(figure_path / 'normal_curve.png', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x_all, y2)
ax.set_xlim([-4, 4])
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
[s.set_visible(False) for s in ax.spines.values()]
[t.set_visible(False) for t in ax.get_xticklines()]
[t.set_visible(False) for t in ax.get_yticklines()]
# px = np.arange(-1.5, -1, 0.01)
px = np.arange(-0.15, 0.15, 0.01)
ax.fill_between(x_all, y2, 0, alpha=0.3)
ax.fill_between(px, norm.pdf(px), 0, alpha=0.55, color='red')
plt.savefig(figure_path / 'normal_curve_subset.png', bbox_inches='tight')


import rasterio
with rasterio.open('C:/Users/ipdavies/Downloads/4594_LC08_022034_20180404_1 (3)/4594_LC08_022034_20180404_1.twi.tif', 'r') as f:
    data = f.read()
    data[data==-999999] = np.nan
    plt.imshow(data[0])
    print(data.shape)