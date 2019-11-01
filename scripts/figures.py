from CPR.utils import preprocessing
import matplotlib.pyplot as plt
from CPR.configs import data_path

# Displaying all feature layers of an image
# Need to figure out how to reproject them to make them not tilted
img = '4444_LC08_044033_20170222_3'
batch = 'v2'
pctl = 50
feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'GSW_perm', 'aspect', 'curve', 'developed', 'elevation',
                 'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'flooded']

data_train, data_vector_train, data_ind_train, feat_keep = preprocessing(data_path, img, pctl, gaps=False)

titles = feat_list_new
fig, axes = plt.subplots(4, 4, figsize=(30, 20))
i = 0
for j, col in enumerate(axes):
    for k, row in enumerate(axes):
        ax = axes[j][k]
        ax.imshow(data_train[:, :, i])
        ax.set_title(titles[i], fontdict={'fontsize': 10})
        ax.axis('off')
        i += 1
