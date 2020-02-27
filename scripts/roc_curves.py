import __init__
import os
import time
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from CPR.utils import tif_stacker, cloud_generator, preprocessing, timer
import pandas as pd
from results_viz import VizFuncs
import sys
import numpy as np
from pathlib import Path
import h5py
from LR_conf_intervals import get_se, get_probs

sys.path.append('../')
from CPR.configs import data_path

# ======================================================================================================================
# Parameters
pctls = [10, 30, 50, 70, 90]

# Get all images in image directory
img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_distSeasonal', 'aspect', 'curve', 'developed', 'elevation',
                 'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm',
                 'flooded']

# ======================================================================================================================
# Create and save ROC data
batches = ['LR_allwater', 'NN_allwater']

for batch in batches:
    for img in img_list:
        bin_file = data_path / batch / 'predictions' / img / 'predictions.h5'
        metrics_path = data_path / batch / 'metrics' / 'testing' / img

        for pctl in pctls:
            data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, feat_list_new,
                                                                                  test=True)
            perm_index = feat_keep.index('GSW_perm')
            flood_index = feat_keep.index('flooded')
            data_vector_test[data_vector_test[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
            data_vector_test = np.delete(data_vector_test, perm_index, axis=1)  # Remove GSW_perm column
            data_shape = data_vector_test.shape
            X_test, y_test = data_vector_test[:, 0:data_shape[1] - 1], data_vector_test[:, data_shape[1] - 1]

            with h5py.File(bin_file, 'r') as f:
                pred_probs = f[str(pctl)]
                pred_probs = np.array(pred_probs)  # Copy h5 dataset to array
                preds = np.argmax(pred_probs, axis=1)

                fpr, tpr, thresh = roc_curve(y_test, pred_probs[:, 1], drop_intermediate=True)
                fpr = np.array(fpr)
                tpr = np.array(tpr)

                roc_vals = pd.DataFrame(np.column_stack([fpr, tpr, thresh]), columns=['fpr', 'tpr', 'thresh'])
                roc_vals.to_csv(metrics_path / '{}'.format('roc_curve_' + str(pctl) + '.csv'), index=False)

# Delete ROC data to save space
# for batch in batches:
#     for img in img_list:
#         metrics_path = data_path / batch / 'metrics' / 'testing' / img
#         for pctl in pctls:
#             Path(metrics_path / '{}'.format('roc_curve_' + str(pctl) + '.csv')).unlink()


# ======================================================================================================================
# Plot ROC curves

for batch in batches:
    viz_params = {'img_list': img_list,
                  'pctls': pctls,
                  'data_path': data_path,
                  'batch': batch,
                  'feat_list_new': feat_list_new}

    viz = VizFuncs(viz_params)


