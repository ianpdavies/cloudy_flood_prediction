import matplotlib.pyplot as plt
import numpy as np
import rasterio
import os
import sys
sys.path.append('../')
from CPR.configs import data_path

img_list = os.listdir(data_path / 'images')
removed = {'4115_LC08_021033_20131227_test', '4444_LC08_044034_20170222_1',
           '4101_LC08_027038_20131103_2', '4594_LC08_022035_20180404_1', '4444_LC08_043035_20170303_1'}
img_list = [x for x in img_list if x not in removed]

img = img_list[9]
stack_path = data_path / 'images' / img / 'stack' / 'stack.tif'

# feat_list_new = ['GSW_distSeasonal', 'aspect', 'curve', 'developed', 'elevation', 'forest',
#                  'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'aspect', 'curve', 'developed', 'elevation', 'forest',
                 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'GSW_perm', 'flooded']

flood_index = feat_list_new.index('flooded')
perm_index = feat_list_new.index('GSW_perm')
with rasterio.open(stack_path, 'r') as ds:
    data = ds.read()
    data = data.transpose((1, -1, 0))
    data[data == -999999] = np.nan
    data[np.isneginf(data)] = np.nan
    flood = data[:, :, flood_index]
    perm = data[:, :, perm_index]

plt.imshow(flood)


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10,10)
y = x**2

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(flood)
plt.tight_layout()

coords = []

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d'%(
        ix, iy))

    global coords
    coords.append((ix, iy))

    if len(coords) == 200:
        fig.canvas.mpl_disconnect(cid)

    return coords
cid = fig.canvas.mpl_connect('button_press_event', onclick)

myCoords = coords.copy()
myCoords = np.floor(myCoords).astype('int')
cols, rows = zip(*myCoords)
data_train = data[rows, cols, 1:]
data_vector_train = data_train[~np.isnan(data_train).any(axis=1)]

from sklearn.linear_model import LogisticRegression
shape = data_vector_train.shape
X_train, y_train = data_vector_train[:, 0:shape[1] - 2], data_vector_train[:, shape[1] - 2]

logreg = LogisticRegression(n_jobs=-1, solver='sag')
logreg.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score


from CPR.utils import tif_stacker, cloud_generator, preprocessing, timer
pctl = 30
data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, feat_list_new, test=True)
data_vector_test = data_vector_test[:, 1:]
X_test, y_test = data_vector_test[:, 0:shape[1] - 2], data_vector_test[:, shape[1] - 2]
pred_probs = logreg.predict_proba(X_test)
preds = np.argmax(pred_probs, axis=1)

perm_mask = data_test[:, :, perm_index]
perm_mask = perm_mask.reshape([perm_mask.shape[0] * perm_mask.shape[1]])
perm_mask = perm_mask[~np.isnan(perm_mask)]
preds[perm_mask.astype('bool')] = 0
y_test[perm_mask.astype('bool')] = 0

accuracy_score(y_test, preds)
precision_score(y_test, preds)
recall_score(y_test, preds)
f1_score(y_test, preds)
roc_auc_score(y_test, pred_probs[:, 1])