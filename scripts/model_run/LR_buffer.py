# Logistic Regression
# Training only on pixels within buffer of all detected water (not just floods)

import __init__
import tensorflow as tf
import os
import multiprocessing
import time
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import h5py
from scipy.ndimage import binary_dilation
from CPR.utils import tif_stacker, cloud_generator, preprocessing, timer
import pandas as pd
import sys
import rasterio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from CPR.utils import preprocessing, tif_stacker
from PIL import Image, ImageEnhance
import seaborn as sns

sys.path.append('../')
from CPR.configs import data_path

# Version numbers
print('Python Version:', sys.version)

# ==================================================================================
# Parameters
batch = 'LR_buffer_noGSW'
pctls = [30]
buffer_iters = [5, 10, 20, 30, 40]

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# Get all images in image directory
img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_distSeasonal', 'aspect', 'curve', 'developed', 'elevation',
                 'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm',
                 'flooded']
viz_params = {'img_list': img_list,
              'pctls': pctls,
              'buffer_iters': buffer_iters,
              'data_path': data_path,
              'batch': batch,
              'feat_list_new': feat_list_new,
              'buffer_iters': buffer_iters}


# ======================================================================================================================


def log_reg_training_buffer(img_list, pctls, feat_list_new, data_path, batch, buffer_iters, buffer_flood_only):
    for img in img_list:
        print(img + ': stacking tif, generating clouds')
        times = []
        tif_stacker(data_path, img, feat_list_new, features=True, overwrite=True)
        cloud_generator(img, data_path, overwrite=False)

        for pctl in pctls:
            print('Preprocessing')
            data_train_full, data_vector_train, data_ind_train, feat_keep = preprocessing(data_path, img, pctl,
                                                                                          feat_list_new,
                                                                                          test=False)
            for buffer_iter in buffer_iters:
                perm_index = feat_keep.index('GSW_perm')
                flood_index = feat_keep.index('flooded')
                data_train = data_train_full.copy()
                if buffer_flood_only:
                    data_train[data_train[:, :, perm_index] == 1, flood_index] = 0
                    mask = data_train[:, :, flood_index]
                    buffer_mask = np.invert(binary_dilation(mask, iterations=buffer_iter))
                else:
                    mask = data_train[:, :, flood_index]
                    buffer_mask = np.invert(binary_dilation(mask, iterations=buffer_iter))
                    data_train[data_train[:, :, perm_index] == 1, flood_index] = 0
                data_train[buffer_mask] = np.nan

                data_vector_train = data_train.reshape([data_train.shape[0] * data_train.shape[1], data_train.shape[2]])
                data_vector_train = data_vector_train[~np.isnan(data_vector_train).any(axis=1)]
                data_vector_train = np.delete(data_vector_train, perm_index, axis=1)  # Remove perm water column
                shape = data_vector_train.shape
                X_train, y_train = data_vector_train[:, 0:shape[1] - 1], data_vector_train[:, shape[1] - 1]

                model_path = data_path / batch / 'models' / img
                metrics_path = data_path / batch / 'metrics' / 'training' / img / '{}'.format(
                    img + '_clouds_' + str(pctl))

                if not model_path.exists():
                    model_path.mkdir(parents=True)
                if not metrics_path.exists():
                    metrics_path.mkdir(parents=True)

                model_path = model_path / '{}'.format(img + '_clouds_' + str(pctl) + 'buff' + str(buffer_iter) + '.sav')

                # Save data flooding image to check that buffering is working correctly
                # imwrite(model_path.parents[0] / '{}'.format('buff' + str(buffer_iter) + '.jpg'), data_train[:, :, 6])

                print('Training')
                start_time = time.time()
                logreg = LogisticRegression(n_jobs=-1, solver='sag')
                logreg.fit(X_train, y_train)
                end_time = time.time()
                times.append(timer(start_time, end_time, False))
                joblib.dump(logreg, model_path)

        metrics_path = metrics_path.parent
        times = [float(i) for i in times]
        times = np.column_stack([np.repeat(pctls, len(buffer_iters)), np.tile(buffer_iters, len(pctls)), times])
        times_df = pd.DataFrame(times, columns=['cloud_cover', 'buffer_iters', 'training_time'])
        times_df.to_csv(metrics_path / 'training_times.csv', index=False)


def prediction_buffer(img_list, pctls, feat_list_new, data_path, batch, remove_perm, buffer_iters):
    for img in img_list:
        times = []
        accuracy, precision, recall, f1, roc_auc = [], [], [], [], []
        preds_path = data_path / batch / 'predictions' / img
        bin_file = preds_path / 'predictions.h5'
        metrics_path = data_path / batch / 'metrics' / 'testing' / img

        try:
            metrics_path.mkdir(parents=True)
        except FileExistsError:
            print('Metrics directory already exists')

        for pctl in pctls:
            print('Preprocessing', img, pctl, '% cloud cover')
            data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, feat_list_new,
                                                                                  test=True)
            perm_index = feat_keep.index('GSW_perm')
            flood_index = feat_keep.index('flooded')
            data_vector_test[
                data_vector_test[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water

            data_vector_test = np.delete(data_vector_test, perm_index, axis=1)  # Remove GSW_perm column
            data_shape = data_vector_test.shape
            X_test, y_test = data_vector_test[:, 0:data_shape[1] - 1], data_vector_test[:, data_shape[1] - 1]

            for buffer_iter in buffer_iters:
                print('Predicting for {} at {}% cloud cover'.format(img, pctl))
                start_time = time.time()
                model_path = data_path / batch / 'models' / img / '{}'.format(img + '_clouds_' + str(pctl) +
                                                                              'buff' + str(buffer_iter) + '.sav')
                trained_model = joblib.load(model_path)
                pred_probs = trained_model.predict_proba(X_test)
                preds = np.argmax(pred_probs, axis=1)

                try:
                    preds_path.mkdir(parents=True)
                except FileExistsError:
                    pass

                with h5py.File(bin_file, 'a') as f:
                    pred_name = str(pctl) + '_buff_' + str(buffer_iter)
                    if pred_name in f:
                        print('Deleting earlier mean predictions')
                        del f[pred_name]
                    f.create_dataset(pred_name, data=pred_probs)

                times.append(timer(start_time, time.time(), False))  # Elapsed time for MC simulations

                print('Evaluating predictions')
                accuracy.append(accuracy_score(y_test, preds))
                precision.append(precision_score(y_test, preds))
                recall.append(recall_score(y_test, preds))
                f1.append(f1_score(y_test, preds))
                roc_auc.append(roc_auc_score(y_test, pred_probs[:, 1]))

            del preds, X_test, y_test, trained_model, data_test, data_vector_test, data_ind_test

        metrics = pd.DataFrame(np.column_stack([np.repeat(pctls, len(buffer_iters)),
                                                np.tile(buffer_iters, len(pctls)), accuracy, precision, recall, f1, roc_auc]),
                               columns=['cloud_cover', 'buffer_iters', 'accuracy', 'precision', 'recall', 'f1', 'auc'])
        metrics.to_csv(metrics_path / 'metrics.csv', index=False)
        times = [float(i) for i in times]  # Convert time objects to float, otherwise valMetrics will be non-numeric
        times_df = pd.DataFrame(
            np.column_stack([np.repeat(pctls, len(buffer_iters)), np.tile(buffer_iters, len(pctls)), times]),
            columns=['cloud_cover', 'buffer_iters', 'testing_time'])
        times_df.to_csv(metrics_path / 'testing_times.csv', index=False)


class VizFuncsBuffer:

    def __init__(self, atts):
        self.img_list = None
        self.pctls = None
        self.buffer_iters = None
        self.data_path = None
        self.uncertainty = False
        self.batch = None
        self.feat_list_new = None
        for k, v in atts.items():
            setattr(self, k, v)

    def metric_plots(self):
        """
        Creates plot of performance metrics vs. cloud cover for a single image
        """
        plt.ioff()
        for i, img in enumerate(self.img_list):
            print('Making metric plots for {}'.format(img))
            metrics_path = data_path / self.batch / 'metrics' / 'testing' / img
            plot_path = data_path / self.batch / 'plots' / img

            try:
                plot_path.mkdir(parents=True)
            except FileExistsError:
                pass

            metrics = pd.read_csv(metrics_path / 'metrics.csv')
            metrics.drop(columns='cloud_cover', inplace=True)
            metrics_plot = metrics.plot(x='buffer_iters', y=['recall', 'precision', 'f1', 'accuracy'], ylim=(0, 1))

            metrics_fig = metrics_plot.get_figure()
            metrics_fig.savefig(plot_path / 'metrics_plot.png', dpi=300)

            plt.close('all')

    def time_plot(self):
        """
        Creates plot of training time vs. cloud cover
        """
        plt.ioff()
        for i, img in enumerate(self.img_list):
            print('Making time plots for {}'.format(img))
            metrics_path = data_path / self.batch / 'metrics' / 'training' / img
            plot_path = data_path / self.batch / 'plots' / img

            try:
                plot_path.mkdir(parents=True)
            except FileExistsError:
                pass

            times = pd.read_csv(metrics_path / 'training_times.csv')
            if len(self.pctls) > 1:
                time_plot = times.plot(x='buffer_iters', y=['training_time'])
            else:
                time_plot = times.plot.scatter(x='buffer_iters', y=['training_time'])

            time_plot = time_plot.get_figure()
            time_plot.savefig(plot_path / 'training_times.png', dpi=300)
            plt.close('all')

    def cir_image(self):
        """
        Creates CIR image
        """
        plt.ioff()
        data_path = self.data_path
        for i, img in enumerate(self.img_list):
            print('Creating FN/FP map for {}'.format(img))
            plot_path = data_path / self.batch / 'plots' / img
            bin_file = data_path / self.batch / 'predictions' / img / 'predictions.h5'

            stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

            # Get RGB image
            print('Stacking image')
            band_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
            tif_stacker(data_path, img, band_list, features=False, overwrite=False)
            spectra_stack_path = data_path / 'images' / img / 'stack' / 'spectra_stack.tif'

            # Function to normalize the grid values
            def normalize(array):
                """Normalizes numpy arrays into scale 0.0 - 1.0"""
                array_min, array_max = np.nanmin(array), np.nanmax(array)
                return ((array - array_min) / (array_max - array_min))

            print('Processing CIR image')
            with rasterio.open(spectra_stack_path, 'r') as f:
                nir, red, green = f.read(5), f.read(4), f.read(3)
                nir[nir == -999999] = np.nan
                red[red == -999999] = np.nan
                green[green == -999999] = np.nan
                nirn = normalize(nir)
                redn = normalize(red)
                greenn = normalize(green)
                cir = np.dstack((nirn, redn, greenn))

            # Convert to PIL image, enhance, and save
            cir_img = Image.fromarray((cir * 255).astype(np.uint8()))
            cir_img = ImageEnhance.Contrast(cir_img).enhance(1.5)
            cir_img = ImageEnhance.Sharpness(cir_img).enhance(2)
            cir_img = ImageEnhance.Brightness(cir_img).enhance(2)

            print('Saving CIR image')
            cir_file = plot_path / '{}'.format('cir_img' + '.png')
            cir_img.save(cir_file, dpi=(300, 300))

    def false_map(self, probs, save=True):
        """
        Creates map of FP/FNs overlaid on RGB image
        save : bool
        If true, saves RGB FP/FN overlay image. If false, just saves FP/FN overlay
        """
        plt.ioff()
        data_path = self.data_path
        for i, img in enumerate(self.img_list):
            print('Creating FN/FP map for {}'.format(img))
            plot_path = data_path / self.batch / 'plots' / img
            bin_file = data_path / self.batch / 'predictions' / img / 'predictions.h5'

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

            pctl = self.pctls[0]
            for buffer in self.buffer_iters:
                print('Fetching flood predictions for', str(buffer) + '{}'.format('%'))
                # Read predictions
                with h5py.File(bin_file, 'r') as f:
                    pred_name = str(pctl) + '_buff_' + str(buffer)
                    if probs:
                        prediction_probs = f[(pred_name)]
                        prediction_probs = np.array(prediction_probs)  # Copy h5 dataset to array
                        predictions = np.argmax(prediction_probs, axis=1)
                    else:
                        predictions = f[pred_name]
                        predictions = np.array(predictions)  # Copy h5 dataset to array

                data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl,
                                                                                      self.feat_list_new, test=True)

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
                fpfn_overlay_file = plot_path / '{}'.format('false_map' + str(buffer) + '.png')
                indices = np.where((np.isnan(fpfn_img[:, :, 0])) & np.isnan(fpfn_img[:, :, 1])
                                   & np.isnan(fpfn_img[:, :, 2]) & (fpfn_img[:, :, 3] == 255))
                fpfn_img[indices] = [255, 255, 255, 0]
                fpfn_overlay = Image.fromarray(np.uint8(fpfn_img), mode='RGBA')
                fpfn_overlay.save(fpfn_overlay_file, dpi=(300, 300))

                # Superimpose comparison image and RGB image, then save and close
                if save:
                    rgb_img.paste(fpfn_overlay, (0, 0), fpfn_overlay)
                    print('Saving overlay image for', str(buffer) + '{}'.format('%'))
                    rgb_img.save(plot_path / '{}'.format('false_map_overlay' + str(buffer) + '.png'), dpi=(300, 300))
                plt.close('all')

    def false_map_borders(self):
        """
        Creates map of FP/FNs overlaid on RGB image with cloud borders
        cir : bool
        If true, adds FP/FN overlay to CR
        """
        plt.ioff()
        pctl = self.pctls[0]
        for img in self.img_list:
            img_path = data_path / 'images' / img
            stack_path = img_path / 'stack' / 'stack.tif'
            plot_path = data_path / self.batch / 'plots' / img

            with rasterio.open(str(stack_path), 'r') as ds:
                data = ds.read()
                data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
                data[data == -999999] = np.nan
                data[np.isneginf(data)] = np.nan

            # Get flooded image (including perm water) --------------------------------------
            flood_index = self.feat_list_new.index('flooded')
            perm_index = self.feat_list_new.index('GSW_perm')
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

            for buffer in self.buffer_iters:
                # Get RGB image --------------------------------------
                rgb_file = plot_path / '{}'.format('rgb_img' + '.png')
                rgb_img = Image.open(rgb_file)

                # Get FP/FN image --------------------------------------
                comparison_img_file = plot_path / '{}'.format('false_map' + str(buffer) + '.png')
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
                rgb_img.save(plot_path / '{}'.format('false_map_border' + str(buffer) + '.png'), dpi=(300, 300))

    def false_map_borders_cir(self):
        """
        Creates map of FP/FNs overlaid on CIR image with cloud borders
        """
        plt.ioff()
        for img in self.img_list:
            img_path = data_path / 'images' / img
            stack_path = img_path / 'stack' / 'stack.tif'
            plot_path = data_path / self.batch / 'plots' / img

            with rasterio.open(str(stack_path), 'r') as ds:
                data = ds.read()
                data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
                data[data == -999999] = np.nan
                data[np.isneginf(data)] = np.nan

            # Get flooded image (remove perm water)
            flood_index = self.feat_list_new.index('flooded')
            perm_index = self.feat_list_new.index('GSW_perm')
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

            for pctl in self.pctls:
                # Get CIR image
                cir_file = plot_path / '{}'.format('cir_img' + '.png')
                cir_img = Image.open(cir_file)

                # Get FP/FN image
                comparison_img_file = plot_path / '{}'.format('false_map' + str(pctl) + '.png')
                flood_overlay = Image.open(comparison_img_file)
                flood_overlay_arr = np.array(flood_overlay)
                indices = np.where((flood_overlay_arr[:, :, 0] == 0) & (flood_overlay_arr[:, :, 1] == 0) &
                                   (flood_overlay_arr[:, :, 2] == 0) & (flood_overlay_arr[:, :, 3] == 255))
                flood_overlay_arr[indices] = 0
                # Change red to lime green
                red_indices = np.where((flood_overlay_arr[:, :, 0] == 255) & (flood_overlay_arr[:, :, 1] == 0) &
                                       (flood_overlay_arr[:, :, 2] == 0) & (flood_overlay_arr[:, :, 3] == 255))
                flood_overlay_arr[red_indices] = [0, 255, 64, 255]
                flood_overlay = Image.fromarray(flood_overlay_arr, mode='RGBA')

                # Create cloud border image
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

                # Plot all layers together
                cir_img.paste(true_flood_rgb, (0, 0), true_flood_rgb)
                cir_img.paste(flood_overlay, (0, 0), flood_overlay)
                cir_img.paste(border_rgb, (0, 0), border_rgb)
                cir_img.save(plot_path / '{}'.format('false_map_border_cir' + str(pctl) + '.png'), dpi=(300, 300))

    def metric_plots_multi(self):
        """
        Creates plot of average performance metrics of all images vs. cloud cover
        """
        plt.ioff()
        data_path = self.data_path
        metrics_path = data_path / self.batch / 'metrics' / 'testing'
        plot_path = data_path / self.batch / 'plots'

        try:
            plot_path.mkdir(parents=True)
        except FileExistsError:
            pass

        file_list = [metrics_path / img / 'metrics.csv' for img in self.img_list]
        df_concat = pd.concat(pd.read_csv(file) for file in file_list)
        df_concat.drop(columns='cloud_cover', inplace=True)

        # Median of metric values together in one plot
        median_plot = df_concat.groupby('buffer_iters').median().plot(ylim=(0, 1))
        metrics_fig = median_plot.get_figure()
        metrics_fig.savefig(plot_path / 'median_metrics.png', dpi=300)

        # Mean of metric values together in one plot
        mean_plot = df_concat.groupby('buffer_iters').mean().plot(ylim=(0, 1))
        metrics_fig = mean_plot.get_figure()
        metrics_fig.savefig(plot_path / 'mean_metrics.png', dpi=300)

        # Scatter of cloud_cover vs. metric for each metric, with all image metrics represented as a point
        colors = sns.color_palette("colorblind", 4)
        for j, val in enumerate(df_concat.columns[1:]):
            name = val + 's.png'
            all_metric = df_concat.plot.scatter(x='buffer_iters', y=val, ylim=(0, 1), color=colors[j], alpha=0.3)
            all_metric_fig = all_metric.get_figure()
            all_metric_fig.savefig(plot_path / name, dpi=300)

        plt.close('all')

    def median_highlight(self):
        plt.ioff()
        metrics_path = data_path / self.batch / 'metrics' / 'testing'
        plot_path = data_path / self.batch / 'plots'
        try:
            plot_path.mkdir(parents=True)
        except FileExistsError:
            pass

        colors = sns.color_palette("colorblind", 4)

        metrics = ['accuracy', 'recall', 'precision', 'f1']
        file_list = [metrics_path / img / 'metrics.csv' for img in self.img_list]
        df_concat = pd.concat(pd.read_csv(file) for file in file_list)
        df_concat.drop(columns='cloud_cover', inplace=True)
        median_metrics = df_concat.groupby('buffer_iters').median().reset_index()

        for i, metric in enumerate(metrics):
            plt.figure(figsize=(7, 5), dpi=300)
            for file in file_list:
                metrics = pd.read_csv(file)
                plt.plot(metrics['buffer_iters'], metrics[metric], color=colors[i], linewidth=1, alpha=0.3)
            plt.plot(median_metrics['buffer_iters'], median_metrics[metric], color=colors[i], linewidth=3, alpha=0.9)
            plt.ylim(0, 1)
            plt.xlabel('Buffer Iterations', fontsize=13)
            plt.ylabel(metric.capitalize(), fontsize=13)
            plt.savefig(plot_path / '{}'.format(metric + '_highlight.png'))

    def time_size(self):
        """
        Creates plot of training time vs. number of pixels for each image in a scatterplot
        """
        plt.ioff()
        data_path = self.data_path
        metrics_path = data_path / self.batch / 'metrics' / 'training'
        plot_path = data_path / self.batch / 'plots'

        stack_list = [data_path / 'images' / img / 'stack' / 'stack.tif' for img in self.img_list]
        pixel_counts = []
        for j, stack in enumerate(stack_list):
            print('Getting pixel count of', self.img_list[j])
            with rasterio.open(stack, 'r') as ds:
                img = ds.read(1)
                img[img == -999999] = np.nan
                img[np.isneginf(img)] = np.nan
                cloud_mask_dir = data_path / 'clouds'
                cloud_mask = np.load(cloud_mask_dir / '{0}'.format(self.img_list[j] + '_clouds.npy'))
                for k, pctl in enumerate(self.pctls):
                    cloud_mask = cloud_mask < np.percentile(cloud_mask, pctl)
                    img_pixels = np.count_nonzero(~np.isnan(img))
                    img[cloud_mask] = np.nan
                    pixel_count = np.count_nonzero(~np.isnan(img))
                    pixel_counts.append(pixel_count)

        # pixel_counts = np.tile(pixel_counts, len(self.pctls))
        times_sizes = np.column_stack([np.tile(self.pctls, len(self.img_list)),
                                       # np.repeat(img_list, len(pctls)),
                                       pixel_counts])
        # times_sizes[:, 1] = times_sizes[:, 1].astype(np.int) / times_sizes[:, 0].astype(np.int)

        print('Fetching training times')
        file_list = [metrics_path / img / 'training_times.csv' for img in self.img_list]
        times = pd.concat(pd.read_csv(file) for file in file_list)
        times_sizes = np.column_stack([times_sizes, np.array(times['training_time'])])
        times_sizes = pd.DataFrame(times_sizes, columns=['cloud_cover', 'pixels', 'training_time'])

        print('Creating and saving plots')
        cover_times = times_sizes.plot.scatter(x='cloud_cover', y='training_time')
        cover_times_fig = cover_times.get_figure()
        cover_times_fig.savefig(plot_path / 'cloud_cover_times.png', dpi=300)

        pixel_times = times_sizes.plot.scatter(x='pixels', y='training_time')
        pixel_times_fig = pixel_times.get_figure()
        pixel_times_fig.savefig(plot_path / 'size_times.png', dpi=300)

        plt.close('all')


# ======================================================================================================================
log_reg_training_buffer(img_list, pctls, feat_list_new, data_path, batch, buffer_iters, buffer_flood_only=True)
prediction_buffer(img_list, pctls, feat_list_new, data_path, batch, True, buffer_iters)
viz = VizFuncsBuffer(viz_params)
viz.metric_plots()
viz.cir_image()
viz.time_plot()
viz.false_map(probs=True, save=False)
viz.false_map_borders()
viz.metric_plots_multi()
viz.median_highlight()
