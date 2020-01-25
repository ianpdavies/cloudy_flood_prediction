# This script is for examining images, features, and predictions to visually identify patterns
import numpy as np
import sys
import matplotlib.pyplot as plt
import rasterio
from PIL import Image
sys.path.append('../')
from CPR.configs import data_path

batch = 'v30'

feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

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

pctls = [10, 30, 50, 70, 90]
myDpi = 300
plt.subplots_adjust(top=0.98, bottom=0.055, left=0.024, right=0.976, hspace=0.2, wspace=0.2)

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

    for pctl in pctls:
        # Get RGB image --------------------------------------
        rgb_file = plot_path / '{}'.format('rgb_img' + '.png')
        rgb_img = Image.open(rgb_file)

        # Get FP/FN image --------------------------------------
        comparison_img_file = plot_path / '{}'.format('false_map' + str(pctl) + '.png')
        flood_overlay = Image.open(comparison_img_file)

        # Create cloud border image --------------------------------------
        clouds_dir = data_path / 'clouds'
        clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
        clouds[np.isnan(data[:, :, 0])] = np.nan
        cloudmask = np.less(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))
        np.sum(cloudmask) / (cloudmask.shape[0] * cloudmask.shape[1])
        data[cloudmask] = -999999
        data[data == -999999] = np.nan

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
        indices = np.where((border[:, :, 0] == 0) & (border[:, :, 1] == 0) & (border[:, :, 2] == 0) & (border[:, :, 3] == 0))
        rows, cols = zip(indices)
        border[rows, cols, 0] = 0
        border[rows, cols, 1] = 0
        border[rows, cols, 2] = 0
        border[rows, cols, 3] = 0
        border_rgb = Image.fromarray(border, mode='RGBA')

        # Get flooded image (remove perm water) --------------------------------------
        flood_index = feat_list_new.index('flooded')
        perm_index = feat_list_new.index('GSW_perm')
        indices = np.where((data[:, :, flood_index] == 1) & (data[:, :, perm_index] ==1))
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
        indices = np.where((true_flood_rgb[:, :, 0] == 0) & (true_flood_rgb[:, :, 1] == 0) & (true_flood_rgb[:, :, 2] == 0) & (true_flood_rgb[:, :, 3] == 0))
        rows, cols = zip(indices)
        true_flood_rgb[rows, cols, 0] = 0
        true_flood_rgb[rows, cols, 1] = 0
        true_flood_rgb[rows, cols, 2] = 0
        true_flood_rgb[rows, cols, 3] = 0
        true_flood_rgb = Image.fromarray(true_flood_rgb, mode='RGBA')

        # Plot all layers together --------------------------------------
        # Convert black pixels to transparent in comparison image so it can overlay RGB
        flood_datas = flood_overlay.getdata()
        new_flood_datas = []
        for item in flood_datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                new_flood_datas.append((255, 255, 255, 0))
            else:
                new_flood_datas.append(item)
        flood_overlay.putdata(new_flood_datas)

        # Superimpose comparison image and RGB image, then save and close
        rgb_img.paste(true_flood_rgb, (0, 0), true_flood_rgb)
        rgb_img.paste(flood_overlay, (0, 0), flood_overlay)
        rgb_img.paste(border_rgb, (0, 0), border_rgb)
        rgb_img.save(plot_path / '{}'.format('false_map_border' + str(pctl) + '.png'), dpi=(300, 300))
