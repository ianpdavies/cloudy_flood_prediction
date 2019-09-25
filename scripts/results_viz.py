import pandas as pd
import sys
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from CPR.utils import preprocessing
import h5py
from PIL import Image as Img
sys.path.append('../')
from CPR.configs import data_path
# ==================================================================================

plt.ioff()  # Turn off interactive plotting popups

uncertainty = False

img_list = ['4101_LC08_027038_20131103_1',
            '4101_LC08_027038_20131103_2',
            '4101_LC08_027039_20131103_1',
            '4115_LC08_021033_20131227_1',
            '4337_LC08_026038_20160325_1']

pctls = [10, 20, 30, 40, 50, 60, 70, 80, 90]

for i, img in enumerate(img_list):
    if uncertainty:
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

    gapMetricsList = pd.read_csv(metrics_path / 'gapMetrics.csv')

    metrics_plot = gapMetricsList.plot(x='cloud_cover', y=['recall', 'precision', 'f1', 'accuracy'], ylim=(0, 1))
    metrics_fig = metrics_plot.get_figure()
    metrics_fig.savefig(plot_path / 'metrics_plot.png')

    time_plot = gapMetricsList.plot(x='cloud_cover', y=['time'])
    time_plot = time_plot.get_figure()
    time_plot.savefig(plot_path / 'time_plot.png')

    plt.close('all')

    # ----------------------------------------------------------
    # Visualizing correct/incorrect predictions

    # Reshape predicted values back into image band
    with rasterio.open(stack_path, 'r') as ds:
        shape = ds.read(1).shape  # Shape of full original image
        arr_empty = np.zeros(shape)  # Create empty array with this shape
        arr_empty[:] = np.nan  # Convert all zeroes to NaN

    for i, pctl in enumerate(pctls):
        # Read predictions
        with h5py.File(bin_file, 'r') as f:
            predictions = f[str(pctl)]

            data_test, data_vector_test, data_ind_test = preprocessing(data_path, img, pctl, gaps=True)
            # Add predicted values to cloud-covered pixel positions
            prediction_img = arr_empty
            rows, cols = zip(data_ind_test)
            prediction_img[rows, cols] = predictions[i]

            # Add actual flood values to cloud-covered pixel positions
            flooded_img = arr_empty
            flooded_img[rows, cols] = data_vector_test[:, 14]

            # Visualizing FNs/FPs
            ones = np.ones(shape=shape)
            red = np.where(ones, prediction_img, 0.5)
            blue = np.where(ones, flooded_img, 0.5)
            green = np.minimum(red, blue)

            comparison_img = np.dstack((red, green, blue))
            export = Img.fromarray(comparison_img.astype('uint8'))
            export.save(plot_path / '{}'.format('false_map' + str(pctl) + '.png'))


# =======================================================
