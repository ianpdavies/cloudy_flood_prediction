from CPR.utils import preprocessing
import matplotlib.pyplot as plt
from CPR.configs import data_path
import rasterio
from rasterio.windows import Window
import numpy as np

# Displaying all feature layers of an image
# Need to figure out how to reproject them to make them not tilted. Below is a solution where I just clip them instead,
# but it would be nice to show more of the image
img = '4444_LC08_044033_20170222_3'
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