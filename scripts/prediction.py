import pandas as pd
import numpy as np
import time
import h5py
from models import get_nn1 as get_model
from CPR.utils import preprocessing, timer

# ==================================================================================

def prediction(img_list, pctls, feat_list_new, data_path):
    for j, img in enumerate(img_list):
        times = []
        preds_path = data_path / 'predictions' / 'nn' / img
        bin_file = preds_path / 'predictions.h5'
        metrics_path = data_path / 'metrics' / 'testing_nn' / img

        try:
            metrics_path.mkdir(parents=True)
        except FileExistsError:
            print('Metrics directory already exists')

        for i, pctl in enumerate(pctls):

            data_test, data_vector_test, data_ind_test = preprocessing(data_path, img, pctl, gaps=True)
            data_shape = data_vector_test.shape
            perm_index = feat_list_new.index('GSW_perm')
            data_vector_test = np.delete(data_vector_test, perm_index, axis=1)  # Remove GSW_perm column

            X_test, y_test = data_vector_test[:, 0:data_shape[1]-1], data_vector_test[:, data_shape[1]-1]
            INPUT_DIMS = X_test.shape[1]

            print('Predicting for {} at {}% cloud cover'.format(img, pctl))

            # There is a problem loading keras models: https://github.com/keras-team/keras/issues/10417
            # But loading the trained weights into an identical compiled (but untrained) model works
            start_time = time.time()
            trained_model = get_model(INPUT_DIMS)  # Get untrained model to add trained weights into
            model_path = data_path / 'models' / 'nn' / img / '{}'.format(img + '_clouds_' + str(pctl) + '.h5')
            trained_model.load_weights(str(model_path))
            preds = trained_model.predict(X_test, batch_size=7000, use_multiprocessing=True)
            preds = np.argmax(preds, axis=1)  # Display most probable value

            try:
                preds_path.mkdir(parents=True)
            except FileExistsError:
                pass

            with h5py.File(bin_file, 'a') as f:
                if str(pctl) in f:
                    print('Deleting earlier mean predictions')
                    del f[str(pctl)]
                f.create_dataset(str(pctl), data=preds)

            times.append(timer(start_time, time.time(), False))  # Elapsed time for MC simulations


        times = [float(i) for i in times]  # Convert time objects to float, otherwise valMetrics will be non-numeric
        times_df = pd.DataFrame(np.column_stack([pctls, times]),
                                columns=['cloud_cover', 'training_time'])
        times_df.to_csv(metrics_path / 'training_times.csv', index=False)




