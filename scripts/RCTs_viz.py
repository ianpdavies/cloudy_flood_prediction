import __init__

import os
from zipfile import ZipFile
from results_viz import VizFuncs
import sys
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from CPR.utils import preprocessing
from PIL import Image
import h5py
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import numpy.ma as ma
import pandas as pd
import seaborn as sns

sys.path.append('../../')
from CPR.configs import data_path

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

# Version numbers
print('Python Version:', sys.version)

# ==================================================================================
# Parameters
pctls = [10, 30, 50, 70, 90]

batch = 'RCTs'

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

# Plotting some of the images with the highest variability across trials
img_list = ['4477_LC08_022033_20170519_1',
            '4337_LC08_026038_20160325_1',
            '4594_LC08_022034_20180404_1',
            '4469_LC08_015036_20170502_1',
            '4468_LC08_022035_20170503_1',
            '4101_LC08_027039_20131103_1']

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'batch': batch,
              'feat_list_new': feat_list_new}


# ======================================================================================================================

def metric_plots(data_path, trial):
    """
    Creates plot of performance metrics vs. cloud cover for a single image
    """
    plt.ioff()
    for i, img in enumerate(img_list):
        print('Making metric plots for {}'.format(img))
        metrics_path = data_path / batch / trial / 'metrics' / 'testing' / img
        plot_path = data_path / batch / trial / 'plots' / img

        try:
            plot_path.mkdir(parents=True)
        except FileExistsError:
            pass

        metrics = pd.read_csv(metrics_path / 'metrics.csv')

        if len(pctls) > 1:
            fig, ax = plt.subplots(figsize=(5, 3))
            metrics_plot = metrics.plot(x='cloud_cover', y=['accuracy', 'precision', 'recall', 'f1'],
                                        ylim=(0, 1), ax=ax, style='.-', colormap='Dark2')
            ax.legend(['Accuracy', 'Precision', 'Recall', 'F1'], loc='lower left', bbox_to_anchor=(0.0, 1.01),
                      ncol=4, borderaxespad=0, frameon=False)
            plt.xticks([10, 30, 50, 70, 90])
            plt.xlabel('Cloud Cover')
            plt.tight_layout()
            metrics_fig = metrics_plot.get_figure()
            metrics_fig.savefig(plot_path / 'metrics_plot.png', dpi=300)
        else:
            metrics_plot = sns.scatterplot(data=pd.melt(metrics, id_vars='cloud_cover'), x='cloud_cover', y='value',
                                           hue='variable')
            metrics_plot.set(ylim=(0, 1))

        metrics_fig = metrics_plot.get_figure()
        metrics_fig.savefig(plot_path / 'metrics_plot.png', dpi=300)

        plt.close('all')


def false_map(data_path, trial, probs, save=True):
    """
    Creates map of FP/FNs overlaid on RGB image
    save : bool
    If true, saves RGB FP/FN overlay image. If false, just saves FP/FN overlay
    """
    plt.ioff()
    data_path = data_path
    for i, img in enumerate(img_list):
        print('Creating false map for {}'.format(img))
        plot_path = data_path / batch / trial / 'plots' / img
        band_combo_dir = data_path / 'band_combos'
        bin_file = data_path / batch / trial / 'predictions' / img / 'predictions.h5'
        stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

        try:
            plot_path.mkdir(parents=True)
        except FileExistsError:
            pass

        # Get RGB image
        rgb_file = band_combo_dir / '{}'.format(img + '_rgb_img' + '.png')
        rgb_img = Image.open(rgb_file)

        for j, pctl in enumerate(pctls):
            print('Fetching flood predictions for', str(pctl) + '{}'.format('%'))
            # Read predictions
            with h5py.File(bin_file, 'r') as f:
                if probs:
                    prediction_probs = f[str(pctl)]
                    prediction_probs = np.array(prediction_probs)  # Copy h5 dataset to array
                    predictions = np.argmax(prediction_probs, axis=1)
                else:
                    predictions = f[str(pctl)]
                    predictions = np.array(predictions)  # Copy h5 dataset to array

            data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl,
                                                                                  feat_list_new, test=True)

            shape = data_test.shape[:2]

            # Add predicted values to cloud-covered pixel positions
            prediction_img = np.zeros(shape)
            prediction_img[:] = np.nan
            rows, cols = zip(data_ind_test)
            prediction_img[rows, cols] = predictions

            # Remove perm water from predictions and actual
            perm_index = feat_keep.index('GSW_perm')
            flood_index = feat_keep.index('flooded')
            data_vector_test[data_vector_test[:, perm_index] == 1, flood_index] = 0
            data_shape = data_vector_test.shape
            perm_feat = data_test[:, :, perm_index]
            prediction_img[((prediction_img == 1) & (perm_feat == 1))] = 0

            # Add actual flood values to cloud-covered pixel positions
            flooded_img = np.zeros(shape)
            flooded_img[:] = np.nan
            flooded_img[rows, cols] = data_vector_test[:, data_shape[1] - 1]

            # Visualizing FNs/FPs
            ones = np.ones(shape=shape)
            red_actual = np.where(ones, flooded_img, 0.5)  # Actual
            blue_preds = np.where(ones, prediction_img, 0.5)  # Predictions
            green_combo = np.minimum(red_actual, blue_preds)
            alphas = np.ones(shape)

            # Convert black pixels to transparent in fpfn image so it can overlay RGB
            fpfn_img = np.dstack((red_actual, green_combo, blue_preds, alphas)) * 255
            fpfn_overlay_file = plot_path / '{}'.format('false_map' + str(pctl) + '.png')
            indices = np.where((np.isnan(fpfn_img[:, :, 0])) & np.isnan(fpfn_img[:, :, 1])
                               & np.isnan(fpfn_img[:, :, 2]) & (fpfn_img[:, :, 3] == 255))
            fpfn_img[indices] = [255, 255, 255, 0]
            fpfn_overlay = Image.fromarray(np.uint8(fpfn_img), mode='RGBA')
            fpfn_overlay.save(fpfn_overlay_file, dpi=(300, 300))

            # Superimpose comparison image and RGB image, then save and close
            if save:
                rgb_img.paste(fpfn_overlay, (0, 0), fpfn_overlay)
                print('Saving overlay image for', str(pctl) + '{}'.format('%'))
                rgb_img.save(plot_path / '{}'.format('false_map_overlay' + str(pctl) + '.png'), dpi=(300, 300))
            plt.close('all')


def false_map_borders(data_path, trial):
    plt.ioff()
    for img in img_list:
        img_path = data_path / 'images' / img
        stack_path = img_path / 'stack' / 'stack.tif'
        plot_path = data_path / batch / trial / 'plots' / img
        band_combo_dir = data_path / 'band_combos'

        try:
            plot_path.mkdir(parents=True)
        except FileExistsError:
            pass

        with rasterio.open(str(stack_path), 'r') as ds:
            data = ds.read()
            data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
            data[data == -999999] = np.nan
            data[np.isneginf(data)] = np.nan

        # Get flood image, remove perm water --------------------------------------
        flood_index = feat_list_new.index('flooded')
        perm_index = feat_list_new.index('GSW_perm')
        perm_img = data[:, :, perm_index]
        true_flood = data[:, :, flood_index]
        true_flood[((true_flood == 1) & (perm_img == 1))] = 0

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
            rgb_file = band_combo_dir / '{}'.format(img + '_rgb_img' + '.png')
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
            cloudmask_border = binary_dilation(cloudmask_binary, iterations=3)
            cloudmask_border = (cloudmask_border - cloudmask_binary)
            # Convert border to yellow
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
            plt.close('all')


def fpfn_map(probs, data_path, trial):
    plt.ioff()
    my_dpi = 300
    # Get predictions and variances
    for img in img_list:
        print('Creating FN/FP map for {}'.format(img))
        plot_path = data_path / batch / trial / 'plots' / img
        preds_bin_file = data_path / batch / trial / 'predictions' / img / 'predictions.h5'
        stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

        try:
            plot_path.mkdir(parents=True)
        except FileExistsError:
            pass

        # Reshape variance values back into image band
        with rasterio.open(stack_path, 'r') as ds:
            shape = ds.read(1).shape  # Shape of full original image

        for pctl in pctls:
            data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl,
                                                                                  feat_list_new, test=True)

            print('Fetching flood predictions for', str(pctl) + '{}'.format('%'))
            with h5py.File(preds_bin_file, 'r') as f:
                predictions = f[str(pctl)]
                if probs:
                    predictions = np.argmax(np.array(predictions), axis=1)  # Copy h5 dataset to array
                if not probs:
                    predictions = np.array(predictions)

            prediction_img = np.zeros(shape)
            prediction_img[:] = np.nan
            rows, cols = zip(data_ind_test)
            prediction_img[rows, cols] = predictions

            perm_index = feat_keep.index('GSW_perm')
            flood_index = feat_keep.index('flooded')
            floods = data_test[:, :, flood_index]
            perm_water = (data_test[:, :, perm_index] == 1)
            tp = np.logical_and(prediction_img == 1, floods == 1).astype('int')
            tn = np.logical_and(prediction_img == 0, floods == 0).astype('int')
            fp = np.logical_and(prediction_img == 1, floods == 0).astype('int')
            fn = np.logical_and(prediction_img == 0, floods == 1).astype('int')

            # Mask out clouds, etc.
            tp = ma.masked_array(tp, mask=np.isnan(prediction_img))
            fp = ma.masked_array(fp, mask=np.isnan(prediction_img))
            fn = ma.masked_array(fn, mask=np.isnan(prediction_img))

            true_false = fp + (fn * 2) + (tp * 3)
            true_false[perm_water] = -1
            colors = ['darkgray', 'saddlebrown', 'limegreen', 'red', 'blue']
            class_labels = ['Permanent Water', 'True Negatives', 'False Floods', 'Missed Floods', 'True Floods']
            legend_patches = [Patch(color=icolor, label=label)
                              for icolor, label in zip(colors, class_labels)]
            cmap = ListedColormap(colors)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.imshow(true_false, cmap=cmap)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.legend(labels=class_labels, handles=legend_patches, loc='lower left', bbox_to_anchor=(0, 1),
                      ncol=5, borderaxespad=0, frameon=False, prop={'size': 7})
            plt.tight_layout()
            plt.savefig(plot_path / '{}'.format('map_fpfn_' + str(pctl) + '.png'), dpi=my_dpi, pad_inches=0.0)

            plt.close('all')


# ======================================================================================================================

cloud_dir = data_path / 'clouds'
try:
    (cloud_dir / 'random').mkdir()
except FileExistsError:
    pass

trial_nums = [1, 2, 3, 4]
for trial_num in trial_nums:
    trial = 'trial' + str(trial_num)
    print('RUNNING', trial, '################################################################')
    zip_dir = str(cloud_dir / 'random' / '{}'.format(trial + '.zip'))
    if not os.path.isdir(str(cloud_dir / 'random' / trial)):
        with ZipFile(zip_dir, 'r') as dst:
            dst.extractall(str(cloud_dir))
    metric_plots(data_path, trial)
    false_map(data_path, trial, probs=True, save=False)
    false_map_borders(data_path, trial)
    fpfn_map(data_path, trial, probs=True)
    for img in img_list:
        cloud_img = cloud_dir / '{}'.format(img + '_clouds.npy')
        os.remove(str(cloud_img))


