from CPR.utils import preprocessing
import matplotlib
import matplotlib.pyplot as plt
from CPR.configs import data_path
import rasterio
from rasterio.windows import Window
import numpy as np

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
gs1.update(wspace=0.1, hspace=0.25)  # set the spacing between axes.

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
# Visualizing correlation matrices
import seaborn as sns
import matplotlib.pyplot as plt
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
g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=8)
bottom, top = g.get_ylim()
g.set_ylim(bottom + 1, top - 1)
plt.tight_layout()
plt.savefig(figure_path / 'corr_matrix.png', dpi=300)