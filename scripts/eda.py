import __init__
import rasterio
import numpy as np
import pandas as pd
import sys
import os

sys.path.append('../')
from CPR.configs import data_path

# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pointbiserialr.html
# Strong assumptions about normality and homoscedasticity
# What about a plot?

from scipy.stats import pointbiserialr, pearsonr

feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

dtypes = ['float32', 'float32', 'float32', 'float32', 'int', 'float32', 'int', 'float32', 'int', 'int', 'float32',
          'float32', 'float32', 'int', 'int', 'int']


img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test'}
img_list = [x for x in img_list if x not in removed]

# # ======================================================================================================================
# # # Correlation of ALL images
# # with rasterio.open(str(data_path / 'images' / img_list[0] / 'stack' / 'stack.tif'), 'r') as ds:
#     # data_vectors = np.zeros((0, ds.count))
#
# # for img in img_list:
#     # img_path = data_path / 'images' / img
#     # stack_path = img_path / 'stack' / 'stack.tif'
#     # with rasterio.open(str(stack_path), 'r') as ds:
#         # data = ds.read()
#         # data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
#         # data[data == -999999] = np.nan
#         # data[np.isneginf(data)] = np.nan
#         # data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
#         # data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
#         # data_vectors = np.append(data_vectors, data_vector, axis=0)
#
# # cor_arr = []
# # j = 0
# # for i in range(0, data_vectors.shape[1], 1):
#     # for k in range(0, data_vectors.shape[1], 1):
#         # if dtypes[j] != dtypes[k]:
#             # if dtypes[i] is 'int':
#                 # cor_arr.append(pointbiserialr(data_vectors[:, j], data_vectors[:, k])[0])
#             # else:
#                 # cor_arr.append(pointbiserialr(data_vectors[:, k], data_vectors[:, j])[0])
#         # else:
#             # cor_arr.append(pearsonr(data_vectors[:, j], data_vectors[:, k])[0])
#     # j += 1
#
# # cor_arr = np.round(cor_arr, 3).reshape([data_vectors.shape[1], data_vectors.shape[1]])
# # np.save(data_path / 'images' / 'corr_matrix.npy', cor_arr)
#
# # Correlation of ALL images, perm water = 0
# with rasterio.open(str(data_path / 'images' / img_list[0] / 'stack' / 'stack.tif'), 'r') as ds:
#     data_vectors = np.zeros((0, ds.count))
#
# perm_index = feat_list_new.index('GSW_perm')
# flood_index = feat_list_new.index('flooded')
#
# for img in img_list:
#     img_path = data_path / 'images' / img
#     stack_path = img_path / 'stack' / 'stack.tif'
#     with rasterio.open(str(stack_path), 'r') as ds:
#         data = ds.read()
#         data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
#         data[data == -999999] = np.nan
#         data[np.isneginf(data)] = np.nan
#         data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
#         data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
#         data_vector[data_vector[:, perm_index] == 1, flood_index] = 0
#         data_vectors = np.append(data_vectors, data_vector, axis=0)
#
# cor_arr = []
# j = 0
# for i in range(0, data_vectors.shape[1], 1):
#     for k in range(0, data_vectors.shape[1], 1):
#         if dtypes[j] != dtypes[k]:
#             if dtypes[i] is 'int':
#                 cor_arr.append(pointbiserialr(data_vectors[:, j], data_vectors[:, k])[0])
#             else:
#                 cor_arr.append(pointbiserialr(data_vectors[:, k], data_vectors[:, j])[0])
#         else:
#             cor_arr.append(pearsonr(data_vectors[:, j], data_vectors[:, k])[0])
#     j += 1
#
# cor_arr = np.round(cor_arr, 3).reshape([data_vectors.shape[1], data_vectors.shape[1]])
# np.save(data_path / 'images' / 'corr_matrix_permzero.npy', cor_arr)
#
#
# # Correlation of ALL images, perm water = Removed
# with rasterio.open(str(data_path / 'images' / img_list[0] / 'stack' / 'stack.tif'), 'r') as ds:
#     data_vectors = np.zeros((0, ds.count))
#
# for img in img_list:
#     img_path = data_path / 'images' / img
#     stack_path = img_path / 'stack' / 'stack.tif'
#     with rasterio.open(str(stack_path), 'r') as ds:
#         data = ds.read()
#         data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
#         data[data == -999999] = np.nan
#         data[np.isneginf(data)] = np.nan
#         data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
#         data_vector[data_vector[:, perm_index] == 1, flood_index] = np.nan
#         data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
#         data_vectors = np.append(data_vectors, data_vector, axis=0)
#
# cor_arr = []
# j = 0
# for i in range(0, data_vectors.shape[1], 1):
#     for k in range(0, data_vectors.shape[1], 1):
#         if dtypes[j] != dtypes[k]:
#             if dtypes[i] is 'int':
#                 cor_arr.append(pointbiserialr(data_vectors[:, j], data_vectors[:, k])[0])
#             else:
#                 cor_arr.append(pointbiserialr(data_vectors[:, k], data_vectors[:, j])[0])
#         else:
#             cor_arr.append(pearsonr(data_vectors[:, j], data_vectors[:, k])[0])
#     j += 1
#
# cor_arr = np.round(cor_arr, 3).reshape([data_vectors.shape[1], data_vectors.shape[1]])
# np.save(data_path / 'images' / 'corr_matrix_permremoved.npy', cor_arr)
#
# # Visualizing correlation matrices
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
#                  'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']
#
# col_names = ['Extnt', 'Dst', 'Asp', 'Crve', 'Dev', 'Elev', 'For', 'HAND', 'Othr', 'Crop', 'Slpe', 'SPI', 'TWI',
#              'Wtlnd', 'Perm', 'Flood']
#
# plt.figure()
# cor_arr = np.load(data_path / 'images' / 'corr_matrix.npy')
# cor_df = pd.DataFrame(data=cor_arr, columns=col_names)
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# mask = np.zeros_like(cor_arr, dtype=np.bool)
# mask[np.triu_indices_from(mask, k=1)] = True
# g = sns.heatmap(cor_df, mask=mask, cmap=cmap, vmax=1, center=0,
#                 square=True, linewidths=.5, cbar_kws={"shrink": .5},
#                 xticklabels=col_names, yticklabels=col_names)
# g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=8)
# bottom, top = g.get_ylim()
# g.set_ylim(bottom + 0.5, top - 0.5)
#
# plt.figure()
# cor_arr = np.load(data_path / 'images' / 'corr_matrix_permzero.npy')
# cor_df = pd.DataFrame(data=cor_arr, columns=col_names)
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# mask = np.zeros_like(cor_arr, dtype=np.bool)
# mask[np.triu_indices_from(mask, k=1)] = True
# g = sns.heatmap(cor_df, mask=mask, cmap=cmap, vmax=1, center=0,
#                 square=True, linewidths=.5, cbar_kws={"shrink": .5},
#                 xticklabels=col_names, yticklabels=col_names)
# g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=8)
# bottom, top = g.get_ylim()
# g.set_ylim(bottom + 0.5, top - 0.5)
#
# plt.figure()
# cor_arr = np.load(data_path / 'images' / 'corr_matrix_permremoved.npy')
# cor_df = pd.DataFrame(data=cor_arr, columns=col_names)
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# mask = np.zeros_like(cor_arr, dtype=np.bool)
# mask[np.triu_indices_from(mask, k=1)] = True
# g = sns.heatmap(cor_df, mask=mask, cmap=cmap, vmax=1, center=0,
#                 square=True, linewidths=.5, cbar_kws={"shrink": .5},
#                 xticklabels=col_names, yticklabels=col_names)
# g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=8)
# bottom, top = g.get_ylim()
# g.set_ylim(bottom + 0.5, top - 0.5)

# ====================================================================================================================
# Corr matrix for each image

import seaborn as sns
import pandas as pd
from scipy.stats import pointbiserialr, pearsonr
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test'}
img_list = [x for x in img_list if x not in removed]

figure_path = data_path / 'corr_matrices'
try:
    figure_path.mkdir(parents=True)
except FileExistsError:
    pass

feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

col_names = ['Extent', 'Distance', 'Aspect', 'Curve', 'Develop', 'Elev', 'Forest', 'HAND', 'Other', 'Crop', 'Slope', 'SPI', 'TWI',
             'Wetland', 'Flood']

feat_list_fancy = ['Max SW extent', 'Dist from max SW', 'Aspect', 'Curve', 'Developed', 'Elevation', 'Forested',
                 'HAND', 'Other LULC', 'Planted', 'Slope', 'SPI', 'TWI', 'Wetlands', 'Flooded']

dtypes = ['float32', 'float32', 'float32', 'float32', 'int', 'float32', 'int', 'float32', 'int', 'int', 'float32',
          'float32', 'float32', 'int', 'int', 'int']

perm_index = feat_list_new.index('GSW_perm')

with rasterio.open(str(data_path / 'images' / img_list[0] / 'stack' / 'stack.tif'), 'r') as ds:
    data_vectors = np.zeros((0, ds.count))

perm_index = feat_list_new.index('GSW_perm')
flood_index = feat_list_new.index('flooded')

img_list = img_list[img_list.index('4337_LC08_026038_20160325_1'):]
for img in img_list:
    img_path = data_path / 'images' / img
    stack_path = img_path / 'stack' / 'stack.tif'
    with rasterio.open(str(stack_path), 'r') as ds:
        data = ds.read()
        data = data.transpose((1, -1, 0))
        data[data == -999999] = np.nan
        data[np.isneginf(data)] = np.nan
        data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
        data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
        data_vector[data_vector[:, perm_index] == 1, flood_index] = 0

    cor_arr = []
    j = 0
    for i in range(0, data_vectors.shape[1], 1):
        for k in range(0, data_vectors.shape[1], 1):
            if dtypes[j] != dtypes[k]:
                if dtypes[i] is 'int':
                    cor_arr.append(pointbiserialr(data_vector[:, j], data_vector[:, k])[0])
                else:
                    cor_arr.append(pointbiserialr(data_vector[:, k], data_vector[:, j])[0])
            else:
                cor_arr.append(pearsonr(data_vector[:, j], data_vector[:, k])[0])
        j += 1

    cor_arr = np.round(cor_arr, 3).reshape([data_vectors.shape[1], data_vectors.shape[1]])
    # np.save(figure_path / '{}'.format(img + '_corr.npy'), cor_arr)
    np.savetxt(figure_path / '{}'.format(img + '_corr.npy'), cor_arr, delimiter=',')

    plt.figure()
    cor_arr = np.delete(cor_arr, perm_index, axis=1)
    cor_arr = np.delete(cor_arr, perm_index, axis=0)
    cor_df = pd.DataFrame(data=cor_arr, columns=col_names)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.zeros_like(cor_arr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    g = sns.heatmap(cor_df, mask=mask, cmap=cmap, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5},
                    xticklabels=col_names, yticklabels=col_names, annot=True, annot_kws={"size": 5})
    g.set_yticklabels(g.get_yticklabels(), rotation=0)
    bottom, top = g.get_ylim()
    g.set_ylim(bottom + 1, top - 1)
    plt.tight_layout()
    plt.savefig(figure_path / '{}'.format(img + '_corr.png'), dpi=300)
    plt.close('all')

