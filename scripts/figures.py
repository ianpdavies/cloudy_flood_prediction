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
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'GSW_perm', 'aspect', 'curve', 'developed', 'elevation',
                 'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'flooded']

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

data_test, data_vector_train, data_ind_train, feat_keep = preprocessing(data_path, img, pctl, gaps=False,
                                                                        normalize=False)

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
