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
# ==================================================================================


class VizFuncs:

    def __init__(self, atts):
        self.img_list = None
        self.pctls = None
        self.data_path = None
        # self.feat_list_new = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        self.uncertainty = False
        for k, v in atts.items():
            setattr(self, k, v)

    def metric_plots(self):
        plt.ioff()
        for i, img in enumerate(self.img_list):
            print('Making metric plots for {}'.format(img))
            if self.uncertainty:
                metrics_path = data_path / 'metrics' / 'testing_nn_mcd' / img
                plot_path = data_path / 'plots' / 'nn_mcd' / img
                bin_file = data_path / 'predictions' / 'nn_mcd' / img / 'predictions.h5'
            else:
                metrics_path = data_path / 'metrics' / 'testing_nn' / img
                plot_path = data_path / 'plots' / 'nn' / img
                bin_file = data_path / 'predictions' / 'nn' / img / 'predictions.h5'

            stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

            try:
                plot_path.mkdir(parents=True)
            except FileExistsError:
                pass

            metrics = pd.read_csv(metrics_path / 'metrics.csv')
            metrics_plot = metrics.plot(x='cloud_cover', y=['recall', 'precision', 'f1', 'accuracy'],
                                               ylim=(0, 1))
            metrics_fig = metrics_plot.get_figure()
            metrics_fig.savefig(plot_path / 'metrics_plot.png')

            metrics = pd.read_csv(metrics_path / 'metrics_np.csv')
            metrics_plot = metrics.plot(x='cloud_cover', y=['recall', 'precision', 'f1', 'accuracy'],
                                               ylim=(0, 1))
            metrics_fig = metrics_plot.get_figure()
            metrics_fig.savefig(plot_path / 'metrics_np_plot.png')

            times = pd.read_csv(metrics_path / 'training_times.csv')
            time_plot = times.plot(x='cloud_cover', y=['training_time'])
            time_plot = time_plot.get_figure()
            time_plot.savefig(plot_path / 'time_plot.png')

            plt.close('all')

    def false_map(self):
        plt.ioff()
        data_path = self.data_path
        for i, img in enumerate(self.img_list):
            print('Creating FN/FP map for {}'.format(img))
            if self.uncertainty:
                plot_path = data_path / 'plots' / 'nn_mcd' / img
                bin_file = data_path / 'predictions' / 'nn_mcd' / img / 'predictions.h5'
            else:
                plot_path = data_path / 'plots' / 'nn' / img
                bin_file = data_path / 'predictions' / 'nn' / img / 'predictions.h5'

            stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

            # Get RGB image
            band_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
            tif_stacker(data_path, img, band_list, features=False, overwrite=True)
            spectra_stack_path = data_path / 'images' / img / 'stack' / 'spectra_stack.tif'

            # Function to normalize the grid values
            def normalize(array):
                """Normalizes numpy arrays into scale 0.0 - 1.0"""
                array_min, array_max = np.nanmin(array), np.nanmax(array)
                return ((array - array_min) / (array_max - array_min))

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
            rgb_img = ImageEnhance.Brightness(rgb_img).enhance(3.5)

            rgb_file = plot_path / '{}'.format('rgb_img' + '.png')
            rgb_img.save(rgb_file)

            # Reshape predicted values back into image band
            with rasterio.open(stack_path, 'r') as ds:
                shape = ds.read(1).shape  # Shape of full original image

            for j, pctl in enumerate(self.pctls):
                # Read predictions
                with h5py.File(bin_file, 'r') as f:
                    predictions = f[str(pctl)]
                    predictions = np.array(predictions)  # Copy h5 dataset to array

                data_test, data_vector_test, data_ind_test = preprocessing(data_path, img, pctl, gaps=True,
                                                                           normalize=False)
                data_shape = data_vector_test.shape
                # Add predicted values to cloud-covered pixel positions
                prediction_img = np.zeros(shape)
                prediction_img[:] = np.nan
                rows, cols = zip(data_ind_test)
                prediction_img[rows, cols] = predictions

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
                rgb_img.save(plot_path / '{}'.format('false_map_overlay' + str(pctl) + '.png'))
                plt.close('all')

    def metric_plots_multi(self):
        plt.ioff()
        if self.uncertainty:
            metrics_path = data_path / 'metrics' / 'testing_nn_mcd'
            plot_path = data_path / 'plots' / 'nn_mcd'
        else:
            metrics_path = data_path / 'metrics' / 'testing_nn'
            plot_path = data_path / 'plots' / 'nn'

        try:
            plot_path.mkdir(parents=True)
        except FileExistsError:
            pass

        file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
        df_concat = pd.concat(pd.read_csv(file) for file in file_list)
        # Average of metric values together in one plot
        mean_plot = df_concat.groupby('cloud_cover').mean().plot(ylim=(0, 1))
        mean_plot_fig = mean_plot.get_figure()
        mean_plot_fig.savefig(plot_path / 'mean_metrics.png')

        # Scatter of cloud_cover vs. metric for each metric, with all image metrics represented as a point
        for j, val in enumerate(df_concat.columns):
            name = val + 's.csv'
            all_metric = df_concat.plot.scatter(x='cloud_cover', y=val, ylim=(0, 1))
            all_metric_fig = all_metric.get_figure()
            all_metric_fig.savefig(plot_path / name)

        file_list_np = [metrics_path / img / 'metrics_np.csv' for img in img_list]
        df_concat_np = pd.concat(pd.read_csv(file) for file in file_list_np)
        # Average of metric values together in one plot
        mean_plot_np = df_concat.groupby('cloud_cover').mean().plot(ylim=(0, 1))
        mean_plot_np_fig = mean_plot_np.get_figure()
        mean_plot_np_fig.savefig(plot_path / 'mean_metrics_np.png')

        for j, val in enumerate(df_concat_np.columns):
            name = val + 's_np.csv'
            all_metric = df_concat.plot.scatter(x='cloud_cover', y=val, ylim=(0, 1))
            all_metric_fig = all_metric.get_figure()
            all_metric_fig.savefig(plot_path / name)

        plt.close('all')



# # # Create histogram of pixel values
# # from rasterio.plot import show_hist
# # with rasterio.open(spectra_stack_path, 'r') as f:
# #     dat = f.read((4, 3, 2))
# #     dat[dat==-999999] = np.nan
# #     show_hist(
# #         dat, bins=50, lw=0.0, stacked=True, alpha=0.3,
# #         histtype='stepfilled', title="Histogram")