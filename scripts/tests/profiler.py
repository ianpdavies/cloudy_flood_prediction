# Profiler for getting line-by-line run times of functions

import __init__
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../')
from CPR.configs import data_path
from line_profiler import LineProfiler

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)

# ==================================================================================
import rasterio
from PIL import Image, ImageEnhance
import h5py
from CPR.utils import preprocessing, tif_stacker
import matplotlib.pyplot as plt
from CPR.configs import data_path
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

img_list = ['4444_LC08_044033_20170222_2']
batch = 'v39'
pctls = [10, 30, 50, 70, 90]
pctls = [90]


def false_map(probs, data_path, save=True):
    """
    Creates map of FP/FNs overlaid on RGB image
    save : bool
    If true, saves RGB FP/FN overlay image. If false, just saves FP/FN overlay
    """
    plt.ioff()
    for i, img in enumerate(img_list):
        print('Creating FN/FP map for {}'.format(img))
        plot_path = data_path / batch / 'plots' / img
        bin_file = data_path / batch / 'predictions' / img / 'predictions.h5'

        stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

        # Get RGB image
        print('Stacking RGB image')
        band_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        tif_stacker(data_path, img, band_list, features=False, overwrite=False)
        spectra_stack_path = data_path / 'images' / img / 'stack' / 'spectra_stack.tif'

        # Function to normalize the grid values
        def normalize(array):
            """Normalizes numpy arrays into scale 0.0 - 1.0"""
            array_min, array_max = np.nanmin(array), np.nanmax(array)
            return ((array - array_min) / (array_max - array_min))

        print('Processing RGB image')
        with rasterio.open(spectra_stack_path, 'r') as f:
            red, green, blue = f.read(4), f.read(3), f.read(2)
            red[red == -999999] = np.nan
            green[green == -999999] = np.nan
            blue[blue == -999999] = np.nan
            redn = normalize(red)
            greenn = normalize(green)
            bluen = normalize(blue)
            rgb = np.dstack((redn, greenn, bluen))

        # Convert to PIL image, enhance, and save
        rgb_img = Image.fromarray((rgb * 255).astype(np.uint8()))
        rgb_img = ImageEnhance.Contrast(rgb_img).enhance(1.5)
        rgb_img = ImageEnhance.Sharpness(rgb_img).enhance(2)
        rgb_img = ImageEnhance.Brightness(rgb_img).enhance(2)

        print('Saving RGB image')
        rgb_file = plot_path / '{}'.format('rgb_img' + '.png')
        rgb_img.save(rgb_file, dpi=(300, 300))

        # Reshape predicted values back into image band
        with rasterio.open(stack_path, 'r') as ds:
            shape = ds.read(1).shape  # Shape of full original image

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

            # Add predicted values to cloud-covered pixel positions
            prediction_img = np.zeros(shape)
            prediction_img[:] = np.nan
            rows, cols = zip(data_ind_test)
            prediction_img[rows, cols] = predictions

            # Remove perm water from predictions and actual
            perm_index = feat_keep.index('GSW_perm')
            flood_index = feat_keep.index('flooded')
            data_vector_test[
                data_vector_test[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
            data_shape = data_vector_test.shape
            with rasterio.open(stack_path, 'r') as ds:
                perm_feat = ds.read(perm_index + 1)
                prediction_img[perm_feat == 1] = 0

            # Add actual flood values to cloud-covered pixel positions
            flooded_img = np.zeros(shape)
            flooded_img[:] = np.nan
            flooded_img[rows, cols] = data_vector_test[:, data_shape[1] - 1]

            # Visualizing FNs/FPs
            ones = np.ones(shape=shape)
            red_actual = np.where(ones, flooded_img, 0.5)  # Actual
            blue_preds = np.where(ones, prediction_img, 0.5)  # Predictions
            green_combo = np.minimum(red_actual, blue_preds)
            alphas = np.ones(shape) * 255

            # Convert black pixels to transparent in fpfn image so it can overlay RGB
            fpfn_img = np.dstack((red_actual, green_combo, blue_preds, alphas))
            fpfn_overlay_file = plot_path / '{}'.format('false_map' + str(pctl) + '.png')
            indices = np.where(
                (fpfn_img[:, :, 0] == 0) & (fpfn_img[:, :, 1] == 0) & (fpfn_img[:, :, 2] == 0) & (
                        fpfn_img[:, :, 3] == 255))
            fpfn_img[indices] = 0
            fpfn_overlay = Image.fromarray(fpfn_img, mode='RGBA')
            fpfn_overlay.save(fpfn_overlay_file, dpi=(300, 300))

            # Superimpose comparison image and RGB image, then save and close
            if save:
                rgb_img.paste(fpfn_overlay, (0, 0), fpfn_overlay)
                print('Saving overlay image for', str(pctl) + '{}'.format('%'))
                rgb_img.save(plot_path / '{}'.format('false_map_overlay' + str(pctl) + '.png'), dpi=(300, 300))
            plt.close('all')


lp = LineProfiler()
lp_wrapper = lp(false_map)
lp_wrapper(probs=False, data_path=data_path, save=True)
lp.print_stats()
