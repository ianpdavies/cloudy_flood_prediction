from CPR.utils import preprocessing
import matplotlib.pyplot as plt
from CPR.configs import data_path
import rasterio
from rasterio.windows import Window
import numpy as np

# ======================================================================================================================

# Displaying all feature layers of an image
# Need to figure out how to reproject them to make them not tilted. Below is a solution where I just clip them instead,
# but it would be nice to show more of the image
img = '4514_LC08_027033_20170826_1'
batch = 'v2'
pctl = 50
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

# Read only a portion of the image
window = Window.from_slices((500, 1400), (270, 1300))
with rasterio.open(stack_path, 'r') as src:
    w = src.read(window=window)
    w[w == -999999] = np.nan
    w[np.isneginf(w)] = np.nan

titles = feat_list_new
fig, axes = plt.subplots(4, 4, figsize=(30, 20))
i = 0
for j, col in enumerate(axes):
    for k, row in enumerate(axes):
        ax = axes[j][k]
        ax.imshow(w[i])
        ax.set_title(titles[i], fontdict={'fontsize': 10})
        ax.axis('off')
        i += 1

titles = ['Max SW extent', 'Dist from max SW', 'Permanent water', 'Aspect', 'Curve', 'Developed', 'Elevation',
                 'Forested', 'HAND', 'Other LULC', 'Planted', 'Slope', 'SPI', 'TWI', 'Wetlands', 'Flooded']
import matplotlib.gridspec as gridspec
plt.figure(figsize=(15, 15))
gs1 = gridspec.GridSpec(4, 4)
gs1.update(wspace=0.1, hspace=0.25) # set the spacing between axes.

i = 0
for j, col in enumerate(axes):
    for k, row in enumerate(axes):
        ax = plt.subplot(gs1[i])
        ax.imshow(w[i])
        ax.set_title(titles[i], fontdict={'fontsize': 10})
        ax.axis('off')
        i += 1

ax.get_figure()

# ======================================================================================================================
# Varying cloud cover image
import matplotlib

pctls = [10, 30, 50, 70, 90]
cloud_masks = []
clouds_dir = data_path / 'clouds'
clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
for i, pctl in enumerate(pctls):
    cloud_masks.append(clouds > np.percentile(clouds, pctl))
titles = ['10%', '30%', '50%', '70%', '90%']

plt.figure(figsize=(13, 8))
gs1 = gridspec.GridSpec(1, 5)
gs1.update(wspace=0.1, hspace=0.25) # set the spacing between axes.

blues_reversed = matplotlib.cm.get_cmap('Blues_r')

for i, gs in enumerate(gs1):
    ax = plt.subplot(gs1[i])
    ax.imshow(cloud_masks[i], cmap=blues_reversed)
    ax.set_title(titles[i], fontdict={'fontsize': 15})
    ax.axis('off')

# ======================================================================================================================
# Correlation matrix
import seaborn as sns
import pandas as pd

data_test, data_vector_train, data_ind_train, feat_keep = preprocessing(data_path, img, pctl, feat_list_new, test=True)

df = pd.DataFrame(data=data_vector_train, columns=feat_list_new)
corr_matrix = df.corr()
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 15))
heatmap = sns.heatmap(corr_matrix,
                      mask=mask,
                      square=True,
                      linewidths=.5,
                      cmap='coolwarm',
                      cbar_kws={'shrink': .4,
                                'ticks': [-1, -.5, 0, 0.5, 1]},
                      vmin=-1,
                      vmax=1,
                      annot=True,
                      annot_kws={'size': 12})
# add the column names as labels
ax.set_yticklabels(corr_matrix.columns, rotation=0)
ax.set_xticklabels(corr_matrix.columns)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

# ======================================================================================================================
def highlight_plot():
    print('Making highlight plots')
    plt.ioff()
    metrics_path = data_path / batch / 'metrics' / 'testing'
    plot_path = data_path / batch / 'plots'
    try:
        plot_path.mkdir(parents=True)
    except FileExistsError:
        pass

    colors = sns.color_palette("colorblind", 4)

    metrics = ['accuracy', 'recall', 'precision', 'f1']
    file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
    df_concat = pd.concat(pd.read_csv(file) for file in file_list)
    mean_metrics = df_concat.groupby('cloud_cover').mean().reset_index()

    for i, metric in enumerate(metrics):
        plt.figure(figsize=(7, 5), dpi=300)
        for file in file_list:
            metrics = pd.read_csv(file)
            plt.plot(metrics['cloud_cover'], metrics[metric], color=colors[i], linewidth=1, alpha=0.3)
        plt.plot(mean_metrics['cloud_cover'], mean_metrics[metric], color=colors[i], linewidth=3, alpha=0.9)
        plt.ylim(0, 1)
        plt.xlabel('Cloud Cover', fontsize=13)
        plt.ylabel(metric.capitalize(), fontsize=13)
        plt.savefig(plot_path / '{}'.format(metric + '_highlight.png'))

highlight_plot()


