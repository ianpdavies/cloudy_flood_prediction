import pandas as pd
import sys
import rasterio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from CPR.utils import preprocessing, tif_stacker
import h5py
sys.path.append('../')
from CPR.configs import data_path
# ==================================================================================


class VizFuncs:

    def __init__(self, atts):
        self.img_list = None
        self.pctls = None
        self.data_path = None
        self.feat_list_new = None
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
            tif_stacker(data_path, img, feat_list_new, features=False, overwrite=True)
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
                vmin, vmax = np.nanpercentile(rgb, (5, 95))  # 5-95% stretch
                img_plt = plt.imshow(rgb, cmap='gray', vmin=vmin, vmax=vmax)
                plt.show()
                # rgb = np.dstack((red, green, blue))
                # plt.imshow(rgb, vmin=0.05, vmax=0.2)
                # plt.show()

            rgb_file = plot_path / '{}'.format('rgb_img' + '.png')
            matplotlib.image.imsave(rgb_file, rgb)

            # Reshape predicted values back into image band
            with rasterio.open(stack_path, 'r') as ds:
                shape = ds.read(1).shape  # Shape of full original image

            for j, pctl in enumerate(self.pctls):
                # Read predictions
                with h5py.File(bin_file, 'r') as f:
                    predictions = f[str(pctl)]
                    predictions = np.array(predictions)  # Copy h5 dataset to array

                data_test, data_vector_test, data_ind_test = preprocessing(data_path, img, pctl, gaps=True)
                # Add predicted values to cloud-covered pixel positions
                prediction_img = np.zeros(shape)
                prediction_img[:] = np.nan
                rows, cols = zip(data_ind_test)
                prediction_img[rows, cols] = predictions

                # Add actual flood values to cloud-covered pixel positions
                flooded_img = np.zeros(shape)
                flooded_img[:] = np.nan
                flooded_img[rows, cols] = data_vector_test[:, 14]

                # Visualizing FNs/FPs
                ones = np.ones(shape=shape)
                red_actual = np.where(ones, flooded_img, 0.5)  # Actual
                blue_preds = np.where(ones, prediction_img, 0.5)  # Predictions
                green_combo = np.minimum(red_actual, blue_preds)

                # Saving FN/FP comparison image
                comparison_img = np.dstack((red_actual, green_combo, blue_preds))
                comparison_img_file = plot_path / '{}'.format('false_map' + str(pctl) + '.png')
                matplotlib.image.imsave(comparison_img_file, comparison_img)

                # Load RGB and comparison images
                rgb_img = Image.open(rgb_file)
                flood_overlay = Image.open(comparison_img_file)

                # Enhance RGB image
                rgb_img = ImageEnhance.Contrast(rgb_img).enhance(2.5)
                rgb_img = ImageEnhance.Sharpness(rgb_img).enhance(2)
                rgb_img = ImageEnhance.Brightness(rgb_img).enhance(1.5)

                # Convert black pixels to transparent in comparison image so it can overlay RGB
                datas = flood_overlay.getdata()
                newData = []
                for item in datas:
                    if item[0] == 0 and item[1] == 0 and item[2] == 0:
                        newData.append((255, 255, 255, 0))
                    else:
                        newData.append(item)
                flood_overlay.putdata(newData)

                # Superimpose comparison image and RGB image
                rgb_img.paste(flood_overlay, (0, 0), flood_overlay)
                plt.imshow(rgb_img)

                matplotlib.image.imsave(plot_path / '{}'.format('false_map_overlay' + str(pctl) + '.png'), rgb_img)


# ==================================================================================
#
# plt.ioff()  # Turn off interactive plotting popups
#
# uncertainty = False
#
# img_list = ['4101_LC08_027038_20131103_1',
#             '4101_LC08_027038_20131103_2',
#             '4101_LC08_027039_20131103_1',
#             '4115_LC08_021033_20131227_1',
#             '4337_LC08_026038_20160325_1']
#
# pctls = [10, 20, 30, 40, 50, 60, 70, 80, 90]
#
# # ----------------------------------------------------------
# # Visualizing plots of performance metrics vs. cloud cover
#
# for i, img in enumerate(img_list):
#     if uncertainty:
#         metrics_path = data_path / 'metrics' / 'testing_nn_mcd' / img
#         plot_path = data_path / 'plots' / 'nn_mcd' / img
#         bin_file = data_path / 'predictions' / 'nn_mcd' / img / 'predictions.h5'
#     else:
#         metrics_path = data_path / 'metrics' / 'testing_nn' / img
#         plot_path = data_path / 'plots' / 'nn' / img
#         bin_file = data_path / 'predictions' / 'nn' / img / 'predictions.h5'
#
#     stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'
#
#     try:
#         plot_path.mkdir(parents=True)
#     except FileExistsError:
#         pass
#
#     gapMetricsList = pd.read_csv(metrics_path / 'gapMetrics.csv')
#
#     metrics_plot = gapMetricsList.plot(x='cloud_cover', y=['recall', 'precision', 'f1', 'accuracy'], ylim=(0, 1))
#     metrics_fig = metrics_plot.get_figure()
#     metrics_fig.savefig(plot_path / 'metrics_plot.png')
#
#     time_plot = gapMetricsList.plot(x='cloud_cover', y=['time'])
#     time_plot = time_plot.get_figure()
#     time_plot.savefig(plot_path / 'time_plot.png')
#
#     plt.close('all')
#
# # ----------------------------------------------------------
# # Visualizing map of FPs and FNs
#
# for i, img in enumerate(img_list):
#
#     if uncertainty:
#         plot_path = data_path / 'plots' / 'nn_mcd' / img
#         bin_file = data_path / 'predictions' / 'nn_mcd' / img / 'predictions.h5'
#     else:
#         plot_path = data_path / 'plots' / 'nn' / img
#         bin_file = data_path / 'predictions' / 'nn' / img / 'predictions.h5'
#
#     stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'
#
#     # Reshape predicted values back into image band
#     with rasterio.open(stack_path, 'r') as ds:
#         shape = ds.read(1).shape  # Shape of full original image
#
#     for j, pctl in enumerate(pctls):
#         # Read predictions
#         with h5py.File(bin_file, 'r') as f:
#             predictions = f[str(pctl)]
#             predictions = np.array(predictions)  # Copy h5 dataset to array
#
#         data_test, data_vector_test, data_ind_test = preprocessing(data_path, img, pctl, gaps=True)
#         # Add predicted values to cloud-covered pixel positions
#         # prediction_img = arr_empty
#         prediction_img = np.zeros(shape)
#         prediction_img[:] = np.nan
#         rows, cols = zip(data_ind_test)
#         prediction_img[rows, cols] = predictions
#
#         # Add actual flood values to cloud-covered pixel positions
#         # flooded_img = arr_empty
#         flooded_img = np.zeros(shape)
#         flooded_img[:] = np.nan
#         flooded_img[rows, cols] = data_vector_test[:, 14]
#
#         # Visualizing FNs/FPs
#         ones = np.ones(shape=shape)
#         red = np.where(ones, flooded_img, 0.5)  # Actual
#         blue = np.where(ones, prediction_img, 0.5)  # Predictions
#         green = np.minimum(red, blue)
#
#         comparison_img = np.dstack((red, green, blue))
#         matplotlib.image.imsave(plot_path / '{}'.format('false_map' + str(pctl) + '.png'), comparison_img)
#
# # ==================================================================================
# # import importlib
# import matplotlib
# # importlib.reload(CPR.utils)
# # import CPR.utils
# from CPR.utils import tif_stacker
# from PIL import Image
# from PIL import ImageEnhance
# import h5py
#
# img_list = ['4101_LC08_027038_20131103_1']
# pctls = [90]
# feat_list_new = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
# uncertainty = False
#
# for i, img in enumerate(img_list):
#
#     if uncertainty:
#         plot_path = data_path / 'plots' / 'nn_mcd' / img
#         bin_file = data_path / 'predictions' / 'nn_mcd' / img / 'predictions.h5'
#     else:
#         plot_path = data_path / 'plots' / 'nn' / img
#         bin_file = data_path / 'predictions' / 'nn' / img / 'predictions.h5'
#
#     stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'
#
#     # Get RGB image
#     tif_stacker(data_path, img, feat_list_new, features=False, overwrite=True)
#     spectra_stack_path = data_path / 'images' / img / 'stack' / 'spectra_stack.tif'
#
#     # Function to normalize the grid values
#     def normalize(array):
#         """Normalizes numpy arrays into scale 0.0 - 1.0"""
#         array_min, array_max = np.nanmin(array), np.nanmax(array)
#         return ((array - array_min) / (array_max - array_min))
#
#     with rasterio.open(spectra_stack_path, 'r') as f:
#         red, green, blue = f.read(4), f.read(3), f.read(2)
#         red[red == -999999] = np.nan
#         green[green == -999999] = np.nan
#         blue[blue == -999999] = np.nan
#         redn = normalize(red)
#         greenn = normalize(green)
#         bluen = normalize(blue)
#         rgb = np.dstack((redn, greenn, bluen))
#         vmin, vmax = np.nanpercentile(rgb, (5, 95))  # 5-95% stretch
#         img_plt = plt.imshow(rgb, cmap='gray', vmin=vmin, vmax=vmax)
#         plt.show()
#         # rgb = np.dstack((red, green, blue))
#         # plt.imshow(rgb, vmin=0.05, vmax=0.2)
#         # plt.show()
#
#     rgb_file = plot_path / '{}'.format('rgb_img' + '.png')
#     matplotlib.image.imsave(rgb_file, rgb)
#
#     # Reshape predicted values back into image band
#     with rasterio.open(stack_path, 'r') as ds:
#         shape = ds.read(1).shape  # Shape of full original image
#
#     for j, pctl in enumerate(pctls):
#         # Read predictions
#         with h5py.File(bin_file, 'r') as f:
#             predictions = f[str(pctl)]
#             predictions = np.array(predictions)  # Copy h5 dataset to array
#
#         data_test, data_vector_test, data_ind_test = preprocessing(data_path, img, pctl, gaps=True)
#         # Add predicted values to cloud-covered pixel positions
#         # prediction_img = arr_empty
#         prediction_img = np.zeros(shape)
#         prediction_img[:] = np.nan
#         rows, cols = zip(data_ind_test)
#         prediction_img[rows, cols] = predictions
#
#         # Add actual flood values to cloud-covered pixel positions
#         # flooded_img = arr_empty
#         flooded_img = np.zeros(shape)
#         flooded_img[:] = np.nan
#         flooded_img[rows, cols] = data_vector_test[:, 14]
#
#         # Visualizing FNs/FPs
#         ones = np.ones(shape=shape)
#         red_actual = np.where(ones, flooded_img, 0.5)  # Actual
#         blue_preds = np.where(ones, prediction_img, 0.5)  # Predictions
#         green_combo = np.minimum(red_actual, blue_preds)
#
#         # Saving FN/FP comparison image
#         comparison_img = np.dstack((red_actual, green_combo, blue_preds))
#         comparison_img_file = plot_path / '{}'.format('false_map' + str(pctl) + '.png')
#         matplotlib.image.imsave(comparison_img_file, comparison_img)
#
#         # Load RGB and comparison images
#         rgb_img = Image.open(rgb_file)
#         flood_overlay = Image.open(comparison_img_file)
#
#         # Enhance RGB image
#         rgb_img = ImageEnhance.Contrast(rgb_img).enhance(2.5)
#         rgb_img = ImageEnhance.Sharpness(rgb_img).enhance(2)
#         rgb_img = ImageEnhance.Brightness(rgb_img).enhance(1.5)
#
#         # Convert black pixels to transparent in comparison image so it can overlay RGB
#         datas = flood_overlay.getdata()
#         newData = []
#         for item in datas:
#             if item[0] == 0 and item[1] == 0 and item[2] == 0:
#                 newData.append((255, 255, 255, 0))
#             else:
#                 newData.append(item)
#         flood_overlay.putdata(newData)
#
#         # Superimpose comparison image and RGB image
#         rgb_img.paste(flood_overlay, (0, 0), flood_overlay)
#         plt.imshow(rgb_img)
#
#         matplotlib.image.imsave(plot_path / '{}'.format('false_map_overlay' + str(pctl) + '.png'), rgb_img)
#
# # # Create histogram of pixel values
# # from rasterio.plot import show_hist
# # with rasterio.open(spectra_stack_path, 'r') as f:
# #     dat = f.read((4, 3, 2))
# #     dat[dat==-999999] = np.nan
# #     show_hist(
# #         dat, bins=50, lw=0.0, stacked=True, alpha=0.3,
# #         histtype='stepfilled', title="Histogram")