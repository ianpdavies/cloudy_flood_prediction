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
        plt.ioff()
        for i, img in enumerate(self.img_list):
            print('Making metric plots for {}'.format(img))
            metrics_path = data_path / self.batch / 'metrics' / 'testing_nn' / img
            plot_path = data_path / self.batch / 'plots' / 'nn' / img

            try:
                plot_path.mkdir(parents=True)
            except FileExistsError:
                pass

            metrics = pd.read_csv(metrics_path / 'metrics.csv')
            if len(self.pctls) > 1:
                metrics_plot = metrics.plot(x='cloud_cover', y=['recall', 'precision', 'f1', 'accuracy'],
                                                   ylim=(0, 1))
            else:
                metrics_plot = sns.scatterplot(data=pd.melt(metrics, id_vars='cloud_cover'), x='cloud_cover', y='value',
                                     hue='variable')
                metrics_plot.set(ylim=(0, 1))

            metrics_fig = metrics_plot.get_figure()
            metrics_fig.savefig(plot_path / 'metrics_plot.png')

            # metrics = pd.read_csv(metrics_path / 'metrics_np.csv')
            # metrics_plot = metrics.plot(x='cloud_cover', y=['recall', 'precision', 'f1', 'accuracy'],
            #                                    ylim=(0, 1))
            # metrics_fig = metrics_plot.get_figure()
            # metrics_fig.savefig(plot_path / 'metrics_np_plot.png')

            plt.close('all')

    def time_plot(self):
        plt.ioff()
        for i, img in enumerate(self.img_list):
            print('Making time plots for {}'.format(img))
            metrics_path = data_path / self.batch / 'metrics' / 'training_nn' / img
            plot_path = data_path / self.batch / 'plots' / 'nn' / img

            try:
                plot_path.mkdir(parents=True)
            except FileExistsError:
                pass

            times = pd.read_csv(metrics_path / 'training_times.csv')
            time_plot = times.plot(x='cloud_cover', y=['training_time'])
            time_plot = time_plot.get_figure()
            time_plot.savefig(plot_path / 'training_times.png')

            plt.close('all')

    def false_map(self):
        plt.ioff()
        data_path = self.data_path
        for i, img in enumerate(self.img_list):
            print('Creating FN/FP map for {}'.format(img))
            plot_path = data_path / self.batch / 'plots' / 'nn' / img
            bin_file = data_path / self.batch / 'predictions' / 'nn' / img / 'predictions.h5'

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
            rgb_img.save(rgb_file)

            # Reshape predicted values back into image band
            with rasterio.open(stack_path, 'r') as ds:
                shape = ds.read(1).shape  # Shape of full original image

            for j, pctl in enumerate(self.pctls):
                print('Fetching flood predictions for', str(pctl)+'{}'.format('%'))
                # Read predictions
                with h5py.File(bin_file, 'r') as f:
                    predictions = f[str(pctl)]
                    predictions = np.array(predictions)  # Copy h5 dataset to array

                data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, gaps=True,
                                                                           normalize=False)

                # Add predicted values to cloud-covered pixel positions
                prediction_img = np.zeros(shape)
                prediction_img[:] = np.nan
                rows, cols = zip(data_ind_test)
                prediction_img[rows, cols] = predictions

                # Remove perm water from predictions and actual
                feat_list_keep = [self.feat_list_new[i] for i in feat_keep]  # Removed if feat was deleted in preprocessing
                perm_index = feat_list_keep.index('GSW_perm')
                flood_index = feat_list_keep.index('flooded')
                data_vector_test[data_vector_test[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
                data_shape = data_vector_test.shape
                with rasterio.open(stack_path, 'r') as ds:
                    perm_feat = ds.read(perm_index+1)
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

                # Saving FN/FP comparison image
                comparison_img = np.dstack((red_actual, green_combo, blue_preds))
                comparison_img_file = plot_path / '{}'.format('false_map' + str(pctl) + '.png')
                print('Saving FN/FP image for', str(pctl) + '{}'.format('%'))
                matplotlib.image.imsave(comparison_img_file, comparison_img)

                # Load comparison image
                flood_overlay = Image.open(comparison_img_file)

                # Convert black pixels to transparent in comparison image so it can overlay RGB
                datas = flood_overlay.getdata()
                newData = []
                for item in datas:
                    if item[0] == 0 and item[1] == 0 and item[2] == 0:
                        newData.append((255, 255, 255, 0))
                    else:
                        newData.append(item)
                flood_overlay.putdata(newData)

                # Superimpose comparison image and RGB image, then save and close
                rgb_img.paste(flood_overlay, (0, 0), flood_overlay)
                plt.imshow(rgb_img)
                print('Saving overlay image for', str(pctl) + '{}'.format('%'))
                rgb_img.save(plot_path / '{}'.format('false_map_overlay' + str(pctl) + '.png'))
                plt.close('all')

    def metric_plots_multi(self):
        plt.ioff()
        data_path = self.data_path
        metrics_path = data_path / self.batch / 'metrics' / 'testing_nn'
        plot_path = data_path / self.batch / 'plots' / 'nn'

        try:
            plot_path.mkdir(parents=True)
        except FileExistsError:
            pass

        file_list = [metrics_path / img / 'metrics.csv' for img in self.img_list]
        df_concat = pd.concat(pd.read_csv(file) for file in file_list)
        # Average of metric values together in one plot
        mean_plot = df_concat.groupby('cloud_cover').mean().plot(ylim=(0, 1))
        mean_plot_fig = mean_plot.get_figure()
        mean_plot_fig.savefig(plot_path / 'mean_metrics.png')

        # Scatter of cloud_cover vs. metric for each metric, with all image metrics represented as a point
        for j, val in enumerate(df_concat.columns):
            name = val + 's.png'
            all_metric = df_concat.plot.scatter(x='cloud_cover', y=val, ylim=(0, 1))
            all_metric_fig = all_metric.get_figure()
            all_metric_fig.savefig(plot_path / name)

        # file_list_np = [metrics_path / img / 'metrics_np.csv' for img in self.img_list]
        # df_concat_np = pd.concat(pd.read_csv(file) for file in file_list_np)
        # # Average of metric values together in one plot
        # mean_plot_np = df_concat.groupby('cloud_cover').mean().plot(ylim=(0, 1))
        # mean_plot_np_fig = mean_plot_np.get_figure()
        # mean_plot_np_fig.savefig(plot_path / 'mean_metrics_np.png')

        # for j, val in enumerate(df_concat_np.columns):
        #     name = val + 's_np.png'
        #     all_metric = df_concat.plot.scatter(x='cloud_cover', y=val, ylim=(0, 1))
        #     all_metric_fig = all_metric.get_figure()
        #     all_metric_fig.savefig(plot_path / name)

        plt.close('all')

    def time_size(self):
        plt.ioff()
        data_path = self.data_path
        metrics_path = data_path / self.batch / 'metrics' / 'training_nn'
        plot_path = data_path / self.batch / 'plots' / 'nn'

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
        cover_times_fig.savefig(plot_path / 'cloud_cover_times.png')

        pixel_times = times_sizes.plot.scatter(x='pixels', y='training_time')
        pixel_times_fig = pixel_times.get_figure()
        pixel_times_fig.savefig(plot_path / 'size_times.png')

        plt.close('all')


# # # Create histogram of pixel values
# # from rasterio.plot import show_hist
# # with rasterio.open(spectra_stack_path, 'r') as f:
# #     dat = f.read((4, 3, 2))
# #     dat[dat==-999999] = np.nan
# #     show_hist(
# #         dat, bins=50, lw=0.0, stacked=True, alpha=0.3,
# #         histtype='stepfilled', title="Histogram")