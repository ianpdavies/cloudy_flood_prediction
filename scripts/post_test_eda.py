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

# pctls = [10, 20, 30, 40, 50, 60, 70, 80, 90]
pctls = [50]
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

batch = 'v2'
uncertainty = False

if uncertainty:
    metrics_path = data_path / batch / 'metrics' / 'training_nn_mcd'
    plot_path = data_path / batch / 'plots' / 'nn_mcd'
else:
    metrics_path = data_path / batch / 'metrics' / 'training_nn'
    plot_path = data_path / batch / 'plots' / 'nn'

stack_list = [data_path / 'images' / img / 'stack' / 'stack.tif' for img in img_list]
pixel_counts = []
flood_counts = []
dry_counts = []
imgs = []

for j, stack in enumerate(stack_list):
    print('Getting pixel count of', img_list[j])
    with rasterio.open(stack, 'r') as ds:
        img = ds.read(ds.count)
        img[img == -999999] = np.nan
        img[np.isneginf(img)] = np.nan
        cloud_mask_dir = data_path / 'clouds'
        cloud_mask = np.load(cloud_mask_dir / '{0}'.format(img_list[j] + '_clouds.npy'))
        for k, pctl in enumerate(pctls):
            cloud_mask = cloud_mask < np.percentile(cloud_mask, pctl)
            img_pixels = np.count_nonzero(~np.isnan(img))
            img[cloud_mask] = np.nan
            pixel_count = np.count_nonzero(~np.isnan(img))
            pixel_counts.append(pixel_count)
            flood_count = np.sum(img[~np.isnan(img)])
            flood_counts.append(flood_count)
            dry_count = pixel_count - flood_count
            dry_counts.append(dry_count)
            imgs.append(img_list[j])

metadata = np.column_stack([pixel_counts, flood_counts, dry_counts])
metadata = pd.DataFrame(metadata, columns=['pixels', 'flood_pixels', 'dry_pixels'])
imgs_df = pd.DataFrame(imgs, columns=['image'])
metadata = pd.concat([metadata, imgs_df], axis=1)

print('Fetching performance metrics')
if uncertainty:
    metrics_path = data_path / batch / 'metrics' / 'testing_nn_mcd'
    plot_path = data_path / batch / 'plots' / 'nn_mcd'
else:
    metrics_path = data_path / batch / 'metrics' / 'testing_nn'
    plot_path = data_path / batch / 'plots' / 'nn'
file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
metrics = pd.concat(pd.read_csv(file) for file in file_list)
data = pd.concat([metrics.reset_index(), metadata.reset_index()], axis=1)
data['flood_dry_ratio'] = data['flood_pixels'] / data['dry_pixels']

# Performance metrics vs. flood pixel counts, colored by image
data_long = pd.melt(data, id_vars=['image',  'pixels', 'flood_pixels', 'dry_pixels', 'flood_dry_ratio'],
                    value_vars=['accuracy', 'precision', 'recall', 'f1'])
sns.scatterplot(x='flood_pixels', y='value', hue='image', data=data_long)
plt.figure()
sns.scatterplot(x='flood_dry_ratio', y='value', hue='image', data=data_long)

# Without accuracy
plt.figure()
data_long = pd.melt(data, id_vars=['image',  'pixels', 'flood_pixels', 'dry_pixels', 'flood_dry_ratio'],
                    value_vars=['precision', 'recall', 'f1'])
sns.scatterplot(x='flood_pixels', y='value', hue='image', data=data_long)

# print('Creating and saving plots')
# cover_times = times_sizes.plot.scatter(x='cloud_cover', y='training_time')
# cover_times_fig = cover_times.get_figure()
# cover_times_fig.savefig(plot_path / 'cloud_cover_times.png')
#
# pixel_times = times_sizes.plot.scatter(x='pixels', y='training_time')
# pixel_times_fig = pixel_times.get_figure()
# pixel_times_fig.savefig(plot_path / 'size_times.png')


# ======================================================================================================================


# ======================================================================================================================
# Performance metrics vs. feature values


