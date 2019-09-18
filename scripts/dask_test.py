import os
import numpy as np
import h5py
import random
import dask.array as da
from netCDF4 import Dataset


# Maybe try appending each set of MC preds to a binary file
# like https://stackoverflow.com/questions/45222878/append-element-to-binary-file
# Then read in the binary file in smaller pieces with d22ask to compute the mean
# https://github.com/dask/dask-tutorial/blob/master/03_array.ipynb

def predict_with_uncertainty(model, X, MC_PASSES):
    """
    Runs a number of forward passes through the model as Monte Carlo simulations.

    Reqs: tensorflow.keras

    Parameters
    ----------
    model : trained tensorflow.keras model
    X : np.array
        Input features
    MC_PASSES : int
        Number of Monte Carlo simulations to run

    Returns
    ----------
    preds : list
        Average prediction for each set of MC simulations (len = len of X)
    variances : list
    """
    MC_preds = []
    for i in range(MC_PASSES):
        if i % 10 == 0 or i == MC_PASSES - 1:
            print('Running MC {}/{}'.format(i, MC_PASSES))
        prob_of_flood = model.predict(X, batch_size=7000, use_multiprocessing=True)
        print('pred', prob_of_flood.shape)
        MC_preds.append(prob_of_flood)  # Only pass prob of 1, not 0. (X, 2)
    MC_preds = np.array(MC_preds)  # (MC_PASSES, X, 2)
    means = np.mean(MC_preds, axis=0)  # (X, 2)
    variances = np.var(MC_preds, axis=0)  # (X, 2)
    # stds = np.std(preds, axis=0)
    preds = np.argmax(means, axis=1)  # (X, 2)
    # pred = np.argmax(means)
    preds = list(preds)
    #     return pred, preds, means, variances, stds
    return preds, variances


def predict_with_uncertainty2(model, X, MC_PASSES):
    preds_path = data_path / 'predictions' / img / '{}'.format(img + '_clouds_' + str(pctl))
    bin_file = preds_path / 'mc_preds.h5'
    try:
        preds_path.mkdir(parents=True)
    except FileExistsError:
        pass
    # Initialize binary file to hold predictions
    with h5py.File(bin_file, 'w') as f:
        mc_preds = f.create_dataset('mc_preds', shape=(1, X.shape[1]),
                                    maxshape=(None, X.shape[1]),
                                    chunks=True, compression='gzip')  # Create empty dataset with shape of data
    for i in range(MC_PASSES):
        if i % 10 == 0 or i == MC_PASSES - 1:
            print('Running MC {}/{}'.format(i, MC_PASSES))
        flood_prob = model.predict(X, batch_size=7000, use_multiprocessing=True)  # Predict
        flood_prob = flood_prob[0]  # Drop probability of not flooded (0) to save space
        with h5py.File(bin_file, 'a') as f:
            f['mc_preds'][-flood_prob.shape[0]:] = flood_prob  # Append preds to h5 file
            if i < MC_PASSES:  # Resize to append next pass, if there is one
                f['mc_preds'].resize((f['mc_preds'].shape[0] + flood_prob.shape[0]), axis=0)

    # Calculate statistics
    with h5py.File('mc_preds.h5', 'r') as f:
        dset = f['mc_preds']
        preds_da = da.from_array(dset)  # Open h5 file as dask array
        means = preds_da.mean(axis=0)
        means = means.compute()
        variances = preds_da.var(axis=0)
        variances = variances.compute()
        preds = means.round()

    return preds, variances

# X = np.random.rand(100, 10000)
# Y = np.random.rand(100, 10000)


X = np.zeros(shape=(50, 5000000)) + 1
Y = np.zeros(shape=(50, 5000000)) + 2

with h5py.File('mc_preds.h5', 'w') as f:  # initialize h5 file
    mc_preds = f.create_dataset('mc_preds', shape=(X.shape[0], X.shape[1]), maxshape=(None, X.shape[1]),
                                chunks=True, compression='gzip')  # create empty dataset with shape

with h5py.File('mc_preds.h5', 'a') as f:  # initialize h5 file
    f['mc_preds'][-X.shape[0]:] = Y
    f['mc_preds'].resize((f['mc_preds'].shape[0] + Y.shape[0]), axis=0)
    print(f['mc_preds'].shape)

with h5py.File('mc_preds.h5', 'r') as f:
    dset = f['mc_preds']
    preds_da = da.from_array(dset)
    means = preds_da.mean(axis=0)
    means = means.compute()
    variances = preds_da.var(axis=0)
    variances = variances.compute()
    preds = means.round()
