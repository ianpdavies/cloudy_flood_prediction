# Overwrite stacked tifs

# Logistic regression
# But trained on all water, tested on perm=0, preds=0 for metrics
# Based on batch LR_perm_3 in LR_perm script

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
import h5py
from LR_conf_intervals import get_se, get_probs

sys.path.append('../')
from CPR.configs import data_path

# ==================================================================================
# Parameters
pctls = [10, 30, 50, 70, 90]

# Get all images in image directory
img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]

# Order in which features should be stacked to create stacked tif
feat_list_new = ['GSW_distSeasonal', 'aspect', 'curve', 'elevation', 'hand', 'slope',
                 'spi', 'twi', 'sti', 'GSW_perm', 'flooded']

feat_list_all = ['developed', 'forest', 'planted', 'wetlands', 'openspace', 'carbonate', 'noncarbonate', 'akl_intrusive',
                 'silicic_resid', 'silicic_resid', 'extrusive_volcanic', 'colluvial_sed', 'glacial_till_clay',
                 'glacial_till_loam', 'glacial_till_coarse', 'glacial_lake_sed_fine', 'glacial_outwash_coarse',
                 'hydric', 'eolian_sed_coarse', 'eolian_sed_fine', 'saline_lake_sed', 'alluv_coastal_sed_fine',
                 'coastal_sed_coarse', 'GSW_distSeasonal', 'aspect', 'curve', 'elevation', 'hand', 'slope', 'spi',
                 'twi', 'sti', 'GSW_perm', 'flooded']

for img in img_list:
    tif_stacker(data_path, img, feat_list_new, features=True, overwrite=True)

