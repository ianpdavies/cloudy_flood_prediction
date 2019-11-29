from models import get_nn_bn as model_func
import tensorflow as tf
import os
from training import training3, SGDRScheduler, LrRangeFinder
from prediction import prediction
from results_viz import VizFuncs
import sys
from collections import OrderedDict

sys.path.append('../../')
from CPR.configs import data_path

# Version numbers
print('Tensorflow version:', tf.__version__)
print('Python Version:', sys.version)

# ==================================================================================
# Examining how well the model works with small, sparse clouds
# ==================================================================================
# Parameters

uncertainty = False
batch = 'v8'
pctls = [50]
BATCH_SIZE = 8192
EPOCHS = 100
DROPOUT_RATE = 0.3  # Dropout rate for MCD
HOLDOUT = 0.3  # Validation data size

try:
    (data_path / batch).mkdir()
except FileExistsError:
    pass

# To get list of all folders (images) in directory
# img_list = os.listdir(data_path / 'images')

# img_list = ['4115_LC08_021033_20131227_test']
img_list = ['4444_LC08_044033_20170222_2',
            '4101_LC08_027038_20131103_1',
            #             '4101_LC08_027038_20131103_2',
            #             '4101_LC08_027039_20131103_1',
            '4115_LC08_021033_20131227_1',
            '4115_LC08_021033_20131227_2',
            #             '4337_LC08_026038_20160325_1',
            #             '4444_LC08_043034_20170303_1',
            '4444_LC08_043035_20170303_1',
            '4444_LC08_044032_20170222_1',
            '4444_LC08_044033_20170222_1',
            #             '4444_LC08_044033_20170222_3',
            #             '4444_LC08_044033_20170222_4',
            #             '4444_LC08_044034_20170222_1',
            '4444_LC08_045032_20170301_1',
            '4468_LC08_022035_20170503_1',
            '4468_LC08_024036_20170501_1',
            #             '4468_LC08_024036_20170501_2',
            #             '4469_LC08_015035_20170502_1',
            #             '4469_LC08_015036_20170502_1',
            '4477_LC08_022033_20170519_1',
            '4514_LC08_027033_20170826_1']

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'GSW_perm', 'aspect', 'curve', 'developed', 'elevation',
                 'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'flooded']

model_params = {'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'verbose': 2,
                'use_multiprocessing': True}

viz_params = {'img_list': img_list,
              'pctls': pctls,
              'data_path': data_path,
              'uncertainty': uncertainty,
              'batch': batch,
              'feat_list_new': feat_list_new}

# ==================================================================================
# Training and prediction
#
# training3(img_list, pctls, model_func, feat_list_new, uncertainty,
#           data_path, batch, DROPOUT_RATE, HOLDOUT, **model_params)
#
# prediction(img_list, pctls, feat_list_new, data_path, batch, remove_perm=True, **model_params)
#
# viz = VizFuncs(viz_params)
# viz.metric_plots()
# viz.time_plot()
# viz.metric_plots_multi()
# viz.false_map()
# viz.time_size()

# Because all of the batch size tests were with the same cloud cover %, have to make a special average plot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

batches = ['v4', 'v5', 'v6', 'v7', 'v8']
batch_sizes = [256, 512, 1024, 4096, 8192]


#
# metrics_all = pd.DataFrame(columns=['cloud_cover', 'precision', 'recall', 'f1', 'batch_size'])
# for i, batch in enumerate(batches):
#     if uncertainty:
#         metrics_path = data_path / batch / 'metrics' / 'testing_nn_mcd'
#         plot_path = data_path / batch / 'plots' / 'nn_mcd'
#     else:
#         metrics_path = data_path / batch / 'metrics' / 'testing_nn'
#         plot_path = data_path / batch / 'plots'
#
#     file_list = [metrics_path / img / 'metrics.csv' for img in img_list]
#     df_concat = pd.concat(pd.read_csv(file) for file in file_list)
#     batch_df = pd.DataFrame(np.tile(batch_sizes[i], len(df_concat)), columns=['batch_size'])
#     batch_df.index = list(range(len(batch_df)))
#     df_concat.index = list(range(len(df_concat)))
#     df_concat = pd.concat([df_concat, batch_df], axis=1)
#     metrics_all = metrics_all.append(df_concat)
# mean_plot = metrics_all.groupby('batch_size').mean().plot(y=['recall', 'precision', 'f1', 'accuracy'], ylim=(0, 1))
# plt.show()

# Display learning rates
def smooth(y, box_pts):
    """smoothes an array by taking the average of the `box_pts` point around each point"""
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


lr_loss = pd.DataFrame(columns=['lr', 'losses', 'batch'])
for i, batch in enumerate(batches):
    if uncertainty:
        metrics_path = data_path / batch / 'metrics' / 'training_nn_mcd'
        plot_path = data_path / batch / 'plots' / 'nn_mcd'
    else:
        metrics_path = data_path / batch / 'metrics' / 'training_nn'
        plot_path = data_path / batch / 'plots'

    lr_vals_path = metrics_path / 'lr_vals'
    df_concat = pd.DataFrame(columns=['lr', 'losses', 'img'])
    for k, img in enumerate(img_list):
        loss_csv_path = lr_vals_path / '{}'.format('losses_' + img + '.csv')
        loss_csv = pd.read_csv(loss_csv_path)
        loss_csv = pd.concat([loss_csv, pd.DataFrame(np.tile(img, loss_csv.shape[0]), columns=['img'])], axis=1)
        loss_csv['losses'] = smooth(loss_csv['losses'], 20)
        df_concat = df_concat.append(loss_csv)
    batch_df = pd.DataFrame(np.tile(batch, len(df_concat)), columns=['batch'])
    batch_df.index = list(range(len(batch_df)))
    df_concat.index = list(range(len(df_concat)))
    df_concat = pd.concat([df_concat, batch_df], axis=1)
    lr_loss = lr_loss.append(df_concat)

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['grey', 'red', 'blue', 'orange', 'green']

for i, batch in enumerate(batches):
    myGroup = lr_loss.groupby(['batch']).get_group(batch)
    for label, df in myGroup.groupby('img'):
        df.plot(ax=ax, label=batch, x='lr', y='losses', color=colors[i], linewidth=1, alpha=0.4)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

# ------------------------------------------------------------
# Each batch is a separate subplot
columns = 5
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(6, 10))
axes = [ax1, ax2, ax3, ax4, ax5]
colors = ['grey', 'red', 'blue', 'orange', 'green']
for i, batch in enumerate(batches):
    myGroup = lr_loss.groupby(['batch']).get_group(batch)
    for label, df in myGroup.groupby('img'):
        ax = axes[i]
        df.plot(ax=ax, label=batch, x='lr', y='losses', color=colors[i], linewidth=1, alpha=0.4)
        ax.set_title('{}'.format('batch size:' + str(batch_sizes[i])), fontdict={'fontsize': 10})

for i, ax in enumerate(axes):
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    print(by_label)
    ax.legend(by_label.values(), by_label.keys())
plt.tight_layout()


