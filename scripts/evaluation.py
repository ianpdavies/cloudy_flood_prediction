from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from CPR.utils import preprocessing
import numpy as np
import pandas as pd
import h5py

# ==================================================================================


def evaluation(img_list, pctls, feat_list_new, data_path, batch, remove_perm=False):

    for j, img in enumerate(img_list):
        accuracy, precision, recall, f1 = [], [], [], []
        accuracy_np, precision_np, recall_np, f1_np = [], [], [], []

        preds_path = data_path / batch / 'predictions' / 'nn' / img
        bin_file = preds_path / 'predictions.h5'
        metrics_path = data_path / batch / 'metrics' / 'testing_nn' / img

        try:
            metrics_path.mkdir(parents=True)
        except FileExistsError:
            print('Metrics directory already exists')

        for i, pctl in enumerate(pctls):
            print('Evaluating for {} at {}% cloud cover'.format(img, pctl))
            with h5py.File(bin_file, 'r') as f:
                preds = f[str(pctl)]
                preds = np.array(preds)  # Copy h5 dataset to array

            print('Preprocessing')
            data_test, data_vector_test, data_ind_test, feat_keep = preprocessing(data_path, img, pctl, gaps=True, normalize=False)
            data_shape = data_vector_test.shape
            feat_list_keep = [feat_list_new[i] for i in feat_keep]  # Removed if feat was deleted in preprocessing
            if remove_perm:
                perm_index = feat_list_keep.index('GSW_perm')
                flood_index = feat_list_keep.index('flooded')
                data_vector_test[data_vector_test[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
            X_test, y_test = data_vector_test[:, 0:data_shape[1]-1], data_vector_test[:, data_shape[1]-1]

            # Metrics including perm water
            print('Evaluating with perm water')
            accuracy.append(accuracy_score(y_test, preds))
            precision.append(precision_score(y_test, preds))
            recall.append(recall_score(y_test, preds))
            f1.append(f1_score(y_test, preds))

            # Metrics excluding perm water
            print('Evaluating without perm water')
            perm_index = feat_list_new.index('GSW_perm')
            non_perm = data_vector_test[data_vector_test[:, perm_index]!=1, :]
            non_perm_preds = preds[data_vector_test[:, perm_index]!=1]
            X_test, y_test = non_perm[:, 0:data_shape[1]-1], non_perm[:, data_shape[1]-1]

            accuracy_np.append(accuracy_score(y_test, non_perm_preds))
            precision_np.append(precision_score(y_test, non_perm_preds))
            recall_np.append(recall_score(y_test, non_perm_preds))
            f1_np.append(f1_score(y_test, non_perm_preds))

        metrics = pd.DataFrame(np.column_stack([pctls, accuracy, precision, recall, f1]),
                                  columns=['cloud_cover', 'accuracy', 'precision', 'recall', 'f1'])

        metrics.to_csv(metrics_path / 'metrics.csv', index=False)

        metrics_np = pd.DataFrame(np.column_stack([pctls, accuracy_np, precision_np, recall_np, f1_np]),
                                  columns=['cloud_cover', 'accuracy', 'precision', 'recall', 'f1'])
        metrics_np.to_csv(metrics_path / 'metrics_np.csv', index=False)