



img_path = data_path / 'images' / img
stack_path = img_path / 'stack' / 'stack.tif'

# load cloudmasks
clouds_dir = data_path / 'clouds'

# Check for any features that have all zeros/same value and remove. For both train and test sets.
# Get local image
with rasterio.open(str(stack_path), 'r') as ds:
    data = ds.read()
    data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
    data[data == -999999] = np.nan
    data[np.isneginf(data)] = np.nan

    # Getting std of train dataset
    # Remove NaNs (real clouds, ice, missing data, etc). from cloudmask
    clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
    clouds[np.isnan(data[:, :, 0])] = np.nan
    cloudmask = np.less(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))
    data[cloudmask] = -999999
    data[data == -999999] = np.nan
    data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
    data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
    train_std = data_vector[:, 0:data_vector.shape[1] - 2].std(0)

    # Getting std of test dataset
    # Remove NaNs (real clouds, ice, missing data, etc). from cloudmask
    data = ds.read()
    data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)
    data[data == -999999] = np.nan
    data[np.isneginf(data)] = np.nan
    clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
    clouds[np.isnan(data[:, :, 0])] = np.nan
    cloudmask = np.greater(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))
    data[cloudmask] = -999999
    data[data == -999999] = np.nan
    data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
    data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]
    test_std = data_vector[:, 0:data_vector.shape[1] - 2].std(0)

# Now adjust feat_list_new to account for a possible removed feature because of std=0
feat_keep = feat_list_new.copy()
with rasterio.open(str(stack_path), 'r') as ds:
    data = ds.read()
    data = data.transpose((1, -1, 0))  # Not sure why the rasterio.read output is originally (D, W, H)

if 0 in train_std.tolist():
    print('Removing', feat_keep[train_std.tolist().index(0)], 'because std=0 in training data')
    zero_feat = train_std.tolist().index(0)
    data = np.delete(data, zero_feat, axis=2)
    feat_keep.pop(zero_feat)

# Now checking stds of test data if not already removed because of train data
if 0 in test_std.tolist():
    zero_feat_ind = test_std.tolist().index(0)
    zero_feat = feat_list_new[zero_feat_ind]
    try:
        zero_feat_ind = feat_keep.index(zero_feat)
        feat_keep.pop(feat_list_new.index(zero_feat))
        data = np.delete(data, zero_feat_ind, axis=2)
    except ValueError:
        pass

# Convert -999999 and -Inf to Nans
data[data == -999999] = np.nan
data[np.isneginf(data)] = np.nan
# Now remove NaNs (real clouds, ice, missing data, etc). from cloudmask
clouds = np.load(clouds_dir / '{0}'.format(img + '_clouds.npy'))
clouds[np.isnan(data[:, :, 0])] = np.nan
if test:
    cloudmask = np.greater(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))
if not test:
    cloudmask = np.less(clouds, np.nanpercentile(clouds, pctl), where=~np.isnan(clouds))

# And mask clouds
data[cloudmask] = -999999
data[data == -999999] = np.nan

# Get indices of non-nan values. These are the indices of the original image array
nans = np.sum(data, axis=2)
data_ind = np.where(~np.isnan(nans))

# Reshape into a 2D array, where rows = pixels and cols = features
data_vector = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
shape = data_vector.shape

# Remove NaNs
data_vector = data_vector[~np.isnan(data_vector).any(axis=1)]

data_mean = data_vector[:, 0:shape[1] - 2].mean(0)
data_std = data_vector[:, 0:shape[1] - 2].std(0)

# Normalize data - only the non-binary variables
data_vector[:, 0:shape[1] - 2] = (data_vector[:, 0:shape[1] - 2] - data_mean) / data_std

# Make sure NaNs are in the same position element-wise in image
mask = np.sum(data, axis=2)
data[np.isnan(mask)] = np.nan



for img in img_list:
    plot_path = data_path / self.batch / 'plots' / img
    for pctl in pctls:
        os.remove(plot_path / '{}'.format('map_uncertainty_s' + str(pctl) + '.png'))
        os.remove(plot_path / 'uncertainty.png')
        os.remove(plot_path / 'fpfn.png')