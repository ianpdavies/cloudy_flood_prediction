# This script is for examining images, features, and predictions to visually identify patterns
import __init__
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import rasterio
import rasterio.plot
from PIL import Image
from rasterio.windows import Window
import os
import pandas as pd
from matplotlib import gridspec

sys.path.append('../')
from CPR.configs import data_path

feat_list_new = ['GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

feat_list_fancy = ['Dist from Perm', 'Aspect', 'Curve', 'Developed', 'Elevation', 'Forested',
                 'HAND', 'Other LULC', 'Planted', 'Slope', 'SPI', 'TWI', 'Wetlands', 'Permanent water', 'Flooded']


feat_list_all = ['developed', 'forest', 'planted', 'wetlands', 'openspace', 'hydgrpA',
                'hydgrpAD', 'hydgrpB', 'hydgrpBD', 'hydgrpC', 'hydgrpCD', 'hydgrpD',
                'GSWDistSeasonal', 'aspect', 'curve', 'elevation', 'hand', 'slope',
                'spi', 'twi', 'sti', 'precip', 'GSWPerm', 'flooded']

img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4101_LC08_027038_20131103_2',
           '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]

pctls = [10, 30, 50, 70, 90]
myDpi = 300
plt.subplots_adjust(top=0.98, bottom=0.055, left=0.024, right=0.976, hspace=0.2, wspace=0.2)

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

# ======================================================================================================================
# View a given feature in all images.
# Requires clicking image to close window and continue with script
# Closing image window instead of clicking image will cause program to crash.

feat = 'curve'
img_list = [img_list[0]]

for i, img in enumerate(img_list):
    print('Image {}/{}, ({})'.format(i + 1, len(img_list), img))
    stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'
    with rasterio.open(stack_path, 'r', crs='EPSG:4326') as ds:
        image = ds.read(feat_list_new.index(feat)+1)
        image[image == -999999] = np.nan
        image[np.isneginf(image)] = np.nan
        bounds = rasterio.plot.plotting_extent(ds)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, extent=bounds)
        plt.waitforbuttonpress()
        plt.close()

# ======================================================================================================================
# Plot all features for one image
img = '4050_LC08_023036_20130429_1'
for i, feat in enumerate(feat_list_all):
    print(feat)
    stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'
    with rasterio.open(stack_path, 'r', crs='EPSG:4326') as ds:
        image = ds.read(i+1)
        image[image == -999999] = np.nan
        image[np.isneginf(image)] = np.nan
        bounds = rasterio.plot.plotting_extent(ds)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, extent=bounds)
        plt.waitforbuttonpress()
        plt.close()

# ======================================================================================================================
# Plot flooding on top of RGB, all images
flood_index = feat_list_all.index('flooded')
perm_index = feat_list_all.index('GSWPerm')
for img in img_list:
    stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'
    band_combo_dir = data_path / 'band_combos'
    rgb_file = band_combo_dir / '{}'.format(img + '_rgb_img' + '.png')
    rgb_img = Image.open(rgb_file)

    with rasterio.open(stack_path, 'r') as src:
        flood = src.read(src.count - 1)
        perm = src.read(src.count - 2)
        bounds = rasterio.plot.plotting_extent(src)
        flood[flood == -999999] = np.nan
        flood[flood == 0] = np.nan
        flood[perm == 1] = 0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb_img, extent=bounds)
    ax.imshow(flood, cmap='autumn', extent=bounds)

    plt.waitforbuttonpress()
    while True:
        if plt.waitforbuttonpress():
            break
    plt.close()
# ======================================================================================================================
# Plot entire RGB image to get bounding box
img = '4444_LC08_043034_20170303_1'
pctl = None
# batch = 'v30'
feat = 'curve'

stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'
band_combo_dir = data_path / 'band_combos'
rgb_file = band_combo_dir / '{}'.format(img + '_rgb_img' + '.png')
rgb_img = Image.open(rgb_file)

plt.imshow(rgb_img)

with rasterio.open(stack_path, 'r') as src:
    flood = src.read(16)
    flood[flood == -999999] = np.nan
    flood[flood == 0] = np.nan

plt.imshow(flood, cmap='autumn')

with rasterio.open(stack_path, 'r') as src:
    perm = src.read(15)
    perm[perm == -999999] = np.nan
    perm[perm == 0] = np.nan

plt.imshow(perm, cmap='autumn')


# Overlay flooding
with rasterio.open('C:/Users/ipdavies/Downloads/drive-download-20200702T191806Z-001/4050_LC08_023036_20130429_1_curve.tif', 'r') as ds:
    img = ds.read(1)
    img[img == -999999] = np.nan
    img[img == -0] = 0
    plt.imshow(img>100)
    # rasterio.plot.plotting_extent(ds)
    # fig, ax = plt.subplots(figsize=(8, 8))
    # rasterio.plot.show(ds, ax=ax, with_bounds=True)

# ======================================================================================================================
# View all features in bounding box
y_top = 950
x_left = 1250
y_bottom = 650
x_right = 900

# Read only a portion of the image
window = Window.from_slices((y_top, y_bottom), (x_left, x_right))

# Read only a portion of the image
window = Window.from_slices((y_top, y_bottom), (x_left, x_right))
# window = Window.from_slices((950, 1250), (1400, 1900))
with rasterio.open(stack_path, 'r') as src:
    w = src.read(window=window)
    w[w == -999999] = np.nan
    w[np.isneginf(w)] = np.nan
    w[15, ((w[14,:,:]==1) & (w[15,:,:] ==1))] = 0
    w = w[1:, :, :]


titles = feat_list_fancy
plt.figure(figsize=(6, 4))
axes = [plt.subplot(4, 4, i + 1) for i in range(15)]
for i, ax in enumerate(axes):
    ax.imshow(w[i])
    ax.set_title(titles[i], fontdict={'fontsize': 8, 'fontname': 'Helvetica'})
    ax.axis('off')
# plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.1)


with rasterio.open(stack_path, 'r') as ds:
    data = ds.read()
    data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
    data[data == -999999] = np.nan
    data[np.isneginf(data)] = np.nan

plt.figure(figsize=(6, 4))
plt.imshow(data[:, :, feat_list_new.index(feat)])
for i, ax in enumerate(axes):
    ax.imshow()
    ax.set_title(feat_list_fancy[i], fontdict={'fontsize': 10, 'fontname': 'Helvetica'})
    ax.axis('off')
plt.tight_layout()


# ======================================================================================================================
# Find images that performed worst/best


batches = ['LR_allwater', 'RF_allwater', 'NN_allwater']
img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]

metrics = ['accuracy', 'f1', 'recall', 'precision']
metrics_fancy = ['Accuracy', 'F1', 'Recall', 'Precision']

dark2_colors = sns.color_palette("Dark2", 8)
color_inds = [0, 2, 5, 7]
colors = []
for j in range(4):
    for i in color_inds:
        colors.append(dark2_colors[i])

for batch in batches:
    metrics_path = data_path / batch / 'metrics' / 'testing'
    file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
    df_concat = pd.concat(pd.read_csv(file) for file in file_list)
    image_numbers = np.repeat(range(0, len(img_list)), len(pctls))
    df_concat['image_numbers'] = image_numbers

    plt.figure(figsize=(7.5, 9))
    axes = [plt.subplot(4, 1, i + 1) for i in range(4)]
    grouped = df_concat.groupby('image_numbers').mean().reset_index()
    for i, ax in enumerate(axes):
        # ax.scatter(df_concat['image_numbers'], df_concat[metrics[i]], s=12)
        ax.scatter(grouped['image_numbers'], grouped[metrics[i]], s=12)
        ax.set_ylabel(metrics_fancy[i])
        ax.set_ylim([0, 1])
        ax.axhline(0.5, ls='--', zorder=1, color='grey', alpha=0.5)
        ax.set_xticks(range(31))
    axes[0].set_title(batch, fontsize=BIGGER_SIZE)
    plt.tight_layout()
    plt.savefig(data_path / batch / 'plots' / 'image_mean_metrics.png', dpi=myDpi)

metrics = ['accuracy', 'f1', 'recall', 'precision']
metrics_fancy = ['Accuracy', 'F1', 'Recall', 'Precision']

legend_patches = [Patch(color=icolor, label=label)
                  for icolor, label in zip(colors, batches)]

plt.figure(figsize=(9, 9))
axes = [plt.subplot(4, 1, i + 1) for i in range(4)]

# Compare all batches image by image
for k, batch in enumerate(batches):
    metrics_path = data_path / batch / 'metrics' / 'testing'
    file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
    df_concat = pd.concat(pd.read_csv(file) for file in file_list)
    image_numbers = np.repeat(range(0, len(img_list)), len(pctls))
    df_concat['image_numbers'] = image_numbers
    grouped = df_concat.groupby('image_numbers').mean().reset_index()
    for i, ax in enumerate(axes):
        # ax.scatter(df_concat['image_numbers'], df_concat[metrics[i]], s=12)
        ax.scatter(grouped['image_numbers'], grouped[metrics[i]], s=12, color=colors[k])
        ax.set_ylabel(metrics_fancy[i])
        ax.set_ylim([0, 1])
        ax.axhline(0.5, ls='--', zorder=1, color='grey', alpha=0.5)
        ax.set_xticks(range(31))
    axes[0].legend(labels=batches, handles=legend_patches, loc='lower left', bbox_to_anchor=(0.1, 1),
          ncol=3, borderaxespad=0, frameon=False, prop={'size': 7})
    plt.tight_layout()

# ======================================================================================================================
# Plot RGB + FP/FN + cloud borders

for img in img_list:
    img_path = data_path / 'images' / img
    stack_path = img_path / 'stack' / 'stack.tif'
    plot_path = data_path / batch / 'plots' / img

    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
        data[data == -999999] = np.nan
        data[np.isneginf(data)] = np.nan

    # Get flooded image (remove perm water) --------------------------------------
    flood_index = feat_list_new.index('flooded')
    perm_index = feat_list_new.index('GSW_perm')
    indices = np.where((data[:, :, flood_index] == 1) & (data[:, :, perm_index] == 1))
    rows, cols = zip(indices)
    true_flood = data[:, :, flood_index]
    true_flood[rows, cols] = 0
    # Now convert to a gray color image
    true_flood_rgb = np.zeros((true_flood.shape[0], true_flood.shape[1], 4), 'uint8')
    true_flood_rgb[:, :, 0] = true_flood * 174
    true_flood_rgb[:, :, 1] = true_flood * 236
    true_flood_rgb[:, :, 2] = true_flood * 238
    true_flood_rgb[:, :, 3] = true_flood * 255
    # Make non-flood pixels transparent
    indices = np.where((true_flood_rgb[:, :, 0] == 0) & (true_flood_rgb[:, :, 1] == 0) &
                       (true_flood_rgb[:, :, 2] == 0) & (true_flood_rgb[:, :, 3] == 0))
    true_flood_rgb[indices] = 0
    true_flood_rgb = Image.fromarray(true_flood_rgb, mode='RGBA')

    for pctl in pctls:
        # Get RGB image --------------------------------------
        rgb_file = plot_path / '{}'.format('rgb_img' + '.png')
        rgb_img = Image.open(rgb_file)

        # Get FP/FN image --------------------------------------
        comparison_img_file = plot_path / '{}'.format('false_map' + str(pctl) + '.png')
        flood_overlay = Image.open(comparison_img_file)
        flood_overlay_arr = np.array(flood_overlay)
        indices = np.where((flood_overlay_arr[:, :, 0] == 0) & (flood_overlay_arr[:, :, 1] == 0) &
                           (flood_overlay_arr[:, :, 2] == 0) & (flood_overlay_arr[:, :, 3] == 255))
        flood_overlay_arr[indices] = 0
        flood_overlay = Image.fromarray(flood_overlay_arr, mode='RGBA')

        # Create cloud border image --------------------------------------
        clouds_dir = data_path / 'clouds'
        clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
        clouds[np.isnan(data[:, :, 0])] = np.nan
        cloudmask = np.less(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))

        from scipy.ndimage import binary_dilation, binary_erosion

        cloudmask_binary = cloudmask.astype('int')
        cloudmask_border = binary_erosion(cloudmask_binary, iterations=2)
        cloudmask_border = binary_dilation(cloudmask_binary, iterations=3)
        cloudmask_border = (cloudmask_border - cloudmask_binary)
        # Convert border to yellow image
        border = np.zeros((cloudmask_border.shape[0], cloudmask_border.shape[1], 4), 'uint8')
        border[:, :, 0] = cloudmask_border * 255
        border[:, :, 1] = cloudmask_border * 255
        border[:, :, 2] = cloudmask_border * 0
        border[:, :, 3] = cloudmask_border * 255
        # Make non-border pixels transparent
        indices = np.where((border[:, :, 0] == 0) & (border[:, :, 1] == 0) &
                           (border[:, :, 2] == 0) & (border[:, :, 3] == 0))
        border[indices] = 0
        border_rgb = Image.fromarray(border, mode='RGBA')

        # Plot all layers together --------------------------------------e
        rgb_img.paste(true_flood_rgb, (0, 0), true_flood_rgb)
        rgb_img.paste(flood_overlay, (0, 0), flood_overlay)
        rgb_img.paste(border_rgb, (0, 0), border_rgb)
        rgb_img.save(plot_path / '{}'.format('false_map_border' + str(pctl) + '.png'), dpi=(300, 300))
