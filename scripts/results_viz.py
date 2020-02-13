import __init__
import pandas as pd
import sys
import rasterio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from CPR.utils import preprocessing, tif_stacker
from PIL import Image, ImageEnhance
import h5py
sys.path.append('../')
from CPR.configs import data_path
import seaborn as sns

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

# ==================================================================================


class VizFuncs:

    def __init__(self, atts):
        self.img_list = None
        self.pctls = None
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

            if len(self.pctls) > 1:
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

                fig, ax = plt.subplots(figsize=(5, 3))
                time_plot = times.plot(x='cloud_cover', y=['training_time'], ax=ax, style='.-', colormap='Dark2')
                ax.legend(['Training Time (minutes)'], loc='lower left', bbox_to_anchor=(0.0, 1.01), borderaxespad=0,
                          frameon=False)
                plt.xlabel('Cloud Cover')
                plt.tight_layout()
            else:
                time_plot = times.plot.scatter(x='cloud_cover', y=['training_time'])

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

            for j, pctl in enumerate(self.pctls):
                print('Fetching flood predictions for', str(pctl)+'{}'.format('%'))
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
                                                                                      self.feat_list_new, test=True)

                # Add predicted values to cloud-covered pixel positions
                prediction_img = np.zeros(shape)
                prediction_img[:] = np.nan
                rows, cols = zip(data_ind_test)
                prediction_img[rows, cols] = predictions

                # Remove perm water from predictions and actual
                perm_index = feat_keep.index('GSW_perm')
                flood_index = feat_keep.index('flooded')
                data_vector_test[data_vector_test[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
                data_shape = data_vector_test.shape
                with rasterio.open(stack_path, 'r') as ds:
                    perm_feat = ds.read(perm_index+1)
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
                    rgb_img.save(plot_path / '{}'.format('false_map_overlay' + str(pctl) + '.png'), dpi=(300,300))
                plt.close('all')

    def false_map_borders(self):
        """
        Creates map of FP/FNs overlaid on RGB image with cloud borders
        cir : bool
        If true, adds FP/FN overlay to CR
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


            # Get flood image, remove perm water --------------------------------------
            flood_index = self.feat_list_new.index('flooded')
            perm_index = self.feat_list_new.index('GSW_perm')
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

            for pctl in self.pctls:
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

        # Median of metric values together in one plot
        if len(self.pctls) > 1:
            fig, ax = plt.subplots(figsize=(5, 3))
            medians = df_concat.groupby('cloud_cover').median().reset_index(level=0)
            median_plot = medians.plot(x='cloud_cover', y=['accuracy', 'precision', 'recall', 'f1'], ylim=(0, 1), ax=ax,
                                       style='.-', colormap='Dark2')
            ax.legend(['Accuracy', 'Precision', 'Recall', 'F1'], loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=4,
                      borderaxespad=0, frameon=False)
            plt.xticks([10, 30, 50, 70, 90])
            plt.xlabel('Cloud Cover')
            plt.tight_layout()
        else:
            median_plot = sns.scatterplot(data=pd.melt(df_concat.groupby('cloud_cover').median().reset_index(),
                                                        id_vars='cloud_cover'),
                                           x='cloud_cover', y='value',
                                           hue='variable')
            median_plot.set(ylim=(0, 1))
        metrics_fig = median_plot.get_figure()
        metrics_fig.savefig(plot_path / 'median_metrics.png', dpi=300)

        # Mean of metric values together in one plot
        if len(self.pctls) > 1:
            fig, ax = plt.subplots(figsize=(5, 3))
            means = df_concat.groupby('cloud_cover').mean().reset_index(level=0)
            mean_plot = means.plot(x='cloud_cover', y=['accuracy', 'precision', 'recall', 'f1'], ylim=(0, 1), ax=ax,
                                       style='.-', colormap='Dark2')
            ax.legend(['Accuracy', 'Precision', 'Recall', 'F1'], loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=4,
                      borderaxespad=0, frameon=False)
            plt.xticks([10, 30, 50, 70, 90])
            plt.xlabel('Cloud Cover')
            plt.tight_layout()
        else:
            mean_plot = sns.scatterplot(data=pd.melt(df_concat.groupby('cloud_cover').mean().reset_index(),
                                                        id_vars='cloud_cover'),
                                           x='cloud_cover', y='value',
                                           hue='variable')
            mean_plot.set(ylim=(0, 1))
        metrics_fig = mean_plot.get_figure()
        metrics_fig.savefig(plot_path / 'mean_metrics.png', dpi=300)

        # Scatter of cloud_cover vs. metric for each metric, with all image metrics represented as a point
        colors = sns.color_palette("colorblind", 4)
        for j, val in enumerate(df_concat.columns[1:]):
            name = val + 's.png'
            all_metric = df_concat.plot.scatter(x='cloud_cover', y=val, ylim=(0, 1), color=colors[j], alpha=0.3)
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

        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        metric_names_fancy = ['Accuracy', 'Precision', 'Recall', 'F1']
        file_list = [metrics_path / img / 'metrics.csv' for img in self.img_list]
        df_concat = pd.concat(pd.read_csv(file) for file in file_list)
        median_metrics = df_concat.groupby('cloud_cover').median().reset_index()

        dark2_colors = sns.color_palette("Dark2", 8)
        color_inds = [0, 2, 5, 7]
        colors = []
        for i in color_inds:
            colors.append(dark2_colors[i])

        fig = plt.figure(figsize=(7, 2.5))
        axes = [plt.subplot(1, 4, i + 1) for i in range(4)]
        for i, ax in enumerate(axes):
            metric = metric_names[i]
            if i is 0:
                for file in file_list:
                    metrics = pd.read_csv(file)
                    metrics.plot(x='cloud_cover', y=metric, ylim=(0, 1), ax=ax, color=colors[i], lw=1, alpha=0.3)
                median_metrics.plot(x='cloud_cover', y=metric, ylim=(0, 1), ax=ax, style='.-', color=colors[i], lw=2,
                                    alpha=0.9)
                ax.set_title(metric_names_fancy[i], fontsize=10)
                ax.get_legend().remove()
                ax.set_xlabel('Cloud Cover')
                ax.set_xticks([10, 30, 50, 70, 90])
            else:
                for file in file_list:
                    metrics = pd.read_csv(file)
                    metrics.plot(x='cloud_cover', y=metric, ylim=(0, 1), ax=ax, color=colors[i], lw=1, alpha=0.3)
                median_metrics.plot(x='cloud_cover', y=metric, ylim=(0, 1), ax=ax, style='.-', color=colors[i], lw=2,
                                    alpha=0.9)
                ax.set_title(metric_names_fancy[i], fontsize=10)
                ax.get_legend().remove()
                ax.get_yaxis().set_visible(False)
                ax.set_xlabel('')
                ax.set_xticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.ylabel(metric.capitalize())
        plt.savefig(plot_path / '{}'.format('median_highlights.png'), dpi=300)

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

# # # Create histogram of pixel values
# # from rasterio.plot import show_hist
# # with rasterio.open(spectra_stack_path, 'r') as f:
# #     dat = f.read((4, 3, 2))
# #     dat[dat==-999999] = np.nan
# #     show_hist(
# #         dat, bins=50, lw=0.0, stacked=True, alpha=0.3,
# #         histtype='stepfilled', title="Histogram")

# ============================================================================

# # Get predictions and uncertainties
# probs = True
# for i, img in enumerate(img_list):
#     print('Creating FN/FP map for {}'.format(img))
#     plot_path = data_path / batch / 'plots' / img
#     stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'
#     preds_bin_file = data_path / batch / 'predictions' / img / 'predictions.h5'
#
#     # Reshape variance values back into image band
#     with rasterio.open(stack_path, 'r') as ds:
#         shape = ds.read(1).shape  # Shape of full original image
#
#     for j, pctl in enumerate(pctls):
#         print('Fetching prediction uncertainties for', str(pctl) + '{}'.format('%'))
#
#         data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, feat_list_new,
#                                                                               test=True)
#         if probs:
#             with h5py.File(preds_bin_file, 'r') as f:
#                 uncertainties = f[str(pctl)]
#                 uncertainties = np.array(uncertainties)
#                 predictions = np.argmax(uncertainties, axis=1)
#                 uncertainties = uncertainties[:, 1]
#         else:
#             vars_bin_file = data_path / batch / 'variances' / img / 'variances.h5'
#             with h5py.File(vars_bin_file, 'r') as f:
#                 uncertainties = f[str(pctl)]
#                 uncertainties = np.array(uncertainties)
#             with h5py.File(preds_bin_file, 'r') as f:
#                 predictions = f[str(pctl)]
#                 predictions = np.array(predictions)
#
#         prediction_img = np.zeros(shape)
#         prediction_img[:] = np.nan
#         rows, cols = zip(data_ind_test)
#         prediction_img[rows, cols] = predictions
#
#         unc_img = np.zeros(shape)
#         unc_img[:] = np.nan
#         rows, cols = zip(data_ind_test)
#         unc_img[rows, cols] = uncertainties
#
#         # ----------------------------------------------------------------
#         # Plotting
#         plt.ioff()
#         plot_path = data_path / batch / 'plots' / img
#         try:
#             plot_path.mkdir(parents=True)
#         except FileExistsError:
#             pass
#         # ----------------------------------------------------------------
#         # # Plot predicted floods
#         # colors = ['saddlebrown', 'blue']
#         # class_labels = ['Predicted No Flood', 'Predicted Flood']
#         # legend_patches = [Patch(color=icolor, label=label)
#         #                   for icolor, label in zip(colors, class_labels)]
#         # cmap = ListedColormap(colors)
#         #
#         # fig, ax = plt.subplots(figsize=(10, 10))
#         # ax.imshow(prediction_img, cmap=cmap)
#         # ax.legend(handles=legend_patches,
#         #           facecolor='white',
#         #           edgecolor='white')
#         # ax.get_xaxis().set_visible(False)
#         # ax.get_yaxis().set_visible(False)
#         #
#         # myFig = ax.get_figure()
#         # myFig.savefig(plot_path / 'predictions.png', dpi=myDpi)
#         # plt.close('all')
#
#         # ----------------------------------------------------------------
#         # Plot variance
#         # fig, ax = plt.subplots(figsize=(10, 10))
#         fig, ax = plt.subplots()
#         img = ax.imshow(unc_img, cmap='plasma')
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#         im_ratio = unc_img.shape[0] / unc_img.shape[1]
#         fig.colorbar(img, ax=ax, fraction=0.046*im_ratio, pad=0.04*im_ratio)
#         myFig = ax.get_figure()
#         myFig.savefig(plot_path / 'uncertainty.png', dpi=myDpi)
#         plt.close('all')
#
#         # ----------------------------------------------------------------
#         # Plot trues and falses
#         flood_index = feat_keep.index('flooded')
#         floods = data_test[:, :, flood_index]
#         tp = np.logical_and(prediction_img == 1, floods == 1).astype('int')
#         tn = np.logical_and(prediction_img == 0, floods == 0).astype('int')
#         fp = np.logical_and(prediction_img == 1, floods == 0).astype('int')
#         fn = np.logical_and(prediction_img == 0, floods == 1).astype('int')
#         falses = fp + fn
#         trues = tp + tn
#         # Mask out clouds, etc.
#         tp = ma.masked_array(tp, mask=np.isnan(prediction_img))
#         tn = ma.masked_array(tn, mask=np.isnan(prediction_img))
#         fp = ma.masked_array(fp, mask=np.isnan(prediction_img))
#         fn = ma.masked_array(fn, mask=np.isnan(prediction_img))
#         falses = ma.masked_array(falses, mask=np.isnan(prediction_img))
#         trues = ma.masked_array(trues, mask=np.isnan(prediction_img))
#
#         true_false = fp + (fn * 2) + (tp * 3)
#         colors = ['saddlebrown',
#                   'red',
#                   'limegreen',
#                   'blue']
#         class_labels = ['True Negatives',
#                         'False Floods',
#                         'Missed Floods',
#                         'True Floods']
#         legend_patches = [Patch(color=icolor, label=label)
#                           for icolor, label in zip(colors, class_labels)]
#         cmap = ListedColormap(colors)
#
#         # fig, ax = plt.subplots(figsize=(10, 10))
#         fig, ax = plt.subplots()
#         ax.imshow(true_false, cmap=cmap)
#         ax.legend(handles=legend_patches,
#                   facecolor='white',
#                   edgecolor='white')
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#
#         myFig = ax.get_figure()
#         myFig.savefig(plot_path / 'truefalse.png', dpi=myDpi)
#         plt.close('all')
#
#         # ======================================================================================================================
#         # Plot uncertainty and predictions but with perm water noted
#         perm_index = feat_keep.index('GSW_perm')
#         perm_water = (data_test[:, :, perm_index] == 1)
#         perm_water_mask = np.ones(shape)
#         perm_water_mask = ma.masked_array(perm_water_mask, mask=~perm_water)
#
#         # ----------------------------------------------------------------
#         # Plot predicted floods with perm water noted
#         colors = ['gray', 'saddlebrown', 'blue']
#         class_labels = ['Permanent Water', 'Predicted No Flood', 'Predicted Flood']
#         # Would be nice to add hatches over permanent water
#         legend_patches = [Patch(color=icolor, label=label)
#                           for icolor, label, in zip(colors, class_labels)]
#         cmap = ListedColormap(colors)
#         prediction_img_mask = prediction_img.copy()
#         prediction_img_mask[perm_water] = -1
#
#         fig, ax = plt.subplots(figsize=(10, 10))
#         ax.imshow(prediction_img_mask, cmap=cmap)
#         ax.legend(handles=legend_patches,
#                   facecolor='white',
#                   edgecolor='white')
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#
#         myFig = ax.get_figure()
#         myFig.savefig(plot_path / 'predictions_perm.png', dpi=myDpi)
#         plt.close('all')
#
#         # ----------------------------------------------------------------
#         # Plot uncertainty with perm water noted
#         colors = ['darkgray']
#         legend_patches = [Patch(color=colors[0], label='Permanent Water')]
#         cmap = ListedColormap(colors)
#
#         # fig, ax = plt.subplots(figsize=(10, 10))
#         fig, ax = plt.subplots()
#         ax.imshow(unc_img, cmap='plasma')
#         ax.imshow(perm_water_mask, cmap=cmap)
#         ax.legend(handles=legend_patches, facecolor='white', edgecolor='white')
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#         im_ratio = unc_img.shape[0] / unc_img.shape[1]
#         fig.colorbar(img, ax=ax, fraction=0.046*im_ratio, pad=0.04*im_ratio)
#         myFig = ax.get_figure()
#         myFig.savefig(plot_path / 'uncertainty_perm.png', dpi=myDpi)
#         plt.close('all')
#
#         # ----------------------------------------------------------------
#         # Plot trues and falses with perm noted
#         floods = data_test[:, :, flood_index]
#         tp = np.logical_and(prediction_img == 1, floods == 1).astype('int')
#         tn = np.logical_and(prediction_img == 0, floods == 0).astype('int')
#         fp = np.logical_and(prediction_img == 1, floods == 0).astype('int')
#         fn = np.logical_and(prediction_img == 0, floods == 1).astype('int')
#         falses = fp + fn
#         trues = tp + tn
#         # Mask out clouds, etc.
#         tp = ma.masked_array(tp, mask=np.isnan(prediction_img))
#         tn = ma.masked_array(tn, mask=np.isnan(prediction_img))
#         fp = ma.masked_array(fp, mask=np.isnan(prediction_img))
#         fn = ma.masked_array(fn, mask=np.isnan(prediction_img))
#         falses = ma.masked_array(falses, mask=np.isnan(prediction_img))
#         trues = ma.masked_array(trues, mask=np.isnan(prediction_img))
#
#         true_false = fp + (fn * 2) + (tp * 3)
#         true_false[perm_water] = -1
#         colors = ['darkgray',
#                   'saddlebrown',
#                   'red',
#                   'limegreen',
#                   'blue']
#         class_labels = ['Permanent Water',
#                         'True Negatives',
#                         'False Floods',
#                         'Missed Floods',
#                         'True Floods']
#         legend_patches = [Patch(color=icolor, label=label)
#                           for icolor, label in zip(colors, class_labels)]
#         cmap = ListedColormap(colors)
#
#         # fig, ax = plt.subplots(figsize=(10, 10))
#         fig, ax = plt.subplots()
#         ax.imshow(true_false, cmap=cmap)
#         ax.legend(handles=legend_patches,
#                   facecolor='white',
#                   edgecolor='white')
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#
#         myFig = ax.get_figure()
#         myFig.savefig(plot_path / 'truefalse_perm.png', dpi=myDpi)
#         plt.close('all')