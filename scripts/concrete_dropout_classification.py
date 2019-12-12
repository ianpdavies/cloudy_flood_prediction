import numpy as np
from sklearn.datasets import make_classification
import tensorflow as tf
import sys
import tensorflow.keras
from tensorflow.keras import initializers
from tensorflow.keras.layers import Input, Dense, Dropout, \
    concatenate, BatchNormalization, Activation
from tensorflow_probability import distributions
from tensorflow.keras import models
import tensorflow.keras.backend as K
import os

# tf.config.experimental_run_functions_eagerly(True)

# Set some optimized config parameters
NUM_PARALLEL_EXEC_UNITS = 4
tf.config.threading.set_intra_op_parallelism_threads(NUM_PARALLEL_EXEC_UNITS)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.set_soft_device_placement(True)
# tf.config.experimental.set_visible_devices(NUM_PARALLEL_EXEC_UNITS, 'CPU')
os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
# ======================================================================================================================

def logsumexp(a):
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max

def test(Y_true, MC_samples):
    """
    Estimate predictive log likelihood:
    log p(y|x, D) = log int p(y|x, w) p(w|D) dw
                 ~= log int p(y|x, w) q(w) dw
                 ~= log 1/K sum p(y|x, w_k) with w_k sim q(w)
                  = LogSumExp log p(y|x, w_k) - log K
    :Y_true: a 2D array of size N x dim
    :MC_samples: a 3D array of size samples K x N x 2*D
    """
    # assert len(MC_samples.shape) == 3
    # assert len(Y_true.shape) == 2
    k = MC_samples.shape[0]
    N = Y_true.shape[0]
    mean = MC_samples[:, :, :D]  # K x N x D
    logvar = MC_samples[:, :, D:]
    test_ll = -0.5 * np.exp(-logvar) * (mean - Y_true[None])**2. - 0.5 * logvar - 0.5 * np.log(2 * np.pi)
    test_ll = np.sum(np.sum(test_ll, -1), -1)
    test_ll = logsumexp(test_ll) - np.log(k)
    pppp = test_ll / N  # per point predictive probability
    rmse = np.mean((np.mean(mean, 0) - Y_true)**2.)**0.5
    return pppp, rmse

# Plot function to make sure stuff makes sense
import pylab

def plot(X_train, Y_train, X_val, Y_val, means):
    indx = np.argsort(X_val[:, 0])
    _, (ax1, ax2, ax3, ax4) = pylab.subplots(1, 4,figsize=(12, 1.5), sharex=True, sharey=True)
    ax1.scatter(X_train[:, 0], Y_train[:, 0], c='y')
    ax1.set_title('Train set')
    ax2.plot(X_val[indx, 0], np.mean(means, 0)[indx, 0], color='skyblue', lw=3)
    ax2.scatter(X_train[:, 0], Y_train[:, 0], c='y')
    ax2.set_title('+Predictive mean')
    for mean in means:
        ax3.scatter(X_val[:, 0], mean[:, 0], c='b', alpha=0.2, lw=0)
    ax3.plot(X_val[indx, 0], np.mean(means, 0)[indx, 0], color='skyblue', lw=3)
    ax3.set_title('+MC samples on validation X')
    ax4.scatter(X_val[:, 0], Y_val[:, 0], c='r', alpha=0.2, lw=0)
    ax4.set_title('Validation set')
    pylab.show()

# ======================================================================================================================

def fit_model(nb_epoch, X_train, Y_train, input_shape, T, D):
    # optimizer = tf.keras.optimizers.Adam()
    tf.keras.backend.clear_session()
    inp = Input(shape=input_shape[1:])
    x = inp
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x, training=True)
    x = Dense(24, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x, training=True)
    x = Dense(12, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x, training=True)

    means = Dense(D, name='means')(x)
    log_var_pre = Dense(1, name='log_var_pre')(x)
    log_var = Activation('softplus', name='log_var')(log_var_pre)  # What does this do?
    logits_variance = concatenate([means, log_var], name='logits_variance')
    softmax_output = Activation('softmax', name='softmax_output')(means)
    model = models.Model(inputs=inp, outputs=[logits_variance, softmax_output])
    # model = models.Model(inputs=inp, outputs=logits_variance)

    iterable = K.variable(np.ones(T))
    def heteroscedastic_categorical_crossentropy(true, pred):
        mean = pred[:, :D]
        log_var = pred[:, D:]
        log_std = K.sqrt(log_var)
        # variance depressor
        logvar_dep = K.exp(log_var) - K.ones_like(log_var)
        # undistorted loss
        undistorted_loss = K.categorical_crossentropy(mean, true, from_logits=True)
        # apply montecarlo simulation
        # T = 20
        dist = distributions.Normal(loc=K.zeros_like(log_std), scale=log_std)
        monte_carlo_results = K.map_fn(gaussian_categorical_crossentropy(true, mean, dist,
                                                                         undistorted_loss, D), iterable,
                                       name='monte_carlo_results')

        var_loss = K.mean(monte_carlo_results, axis=0) * undistorted_loss

        return var_loss + undistorted_loss + K.sum(logvar_dep, -1)

    def gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes):
        def map_fn(i):
            std_samples = dist.sample(1)
            distorted_loss = K.categorical_crossentropy(pred + std_samples, true, from_logits=True)
            diff = undistorted_loss - distorted_loss
            return -K.elu(diff)
        return map_fn

    model.compile(optimizer='Adam',
                  # loss=heteroscedastic_categorical_crossentropy)
                  loss={'logits_variance': heteroscedastic_categorical_crossentropy,
                        'softmax_output': 'categorical_crossentropy'},
                  metrics={'softmax_output': 'categorical_accuracy'},
                  loss_weights={'logits_variance': .2, 'softmax_output': 1.}
                  )
    hist = model.fit(X_train,
                     {'logits_variance': Y_train, 'softmax_output': Y_train},
                     epochs=nb_epoch, batch_size=batch_size, verbose=1, validation_split=0.3)
    loss = hist.history['loss'][-1]
    return model, -0.5 * loss  # return ELBO up to const.

# ------------------------------------------------------------
# Create fake data

# Ns = [10, 25, 50, 100, 1000, 10000]
# Ns = np.array(Ns)
# nb_epochs = [2000, 1000, 500, 200, 20, 2]
N = 10000
nb_epoch = 10
val_size = np.ceil(N * 0.3).astype('int')
nb_features = 1024
Q = 3  # Q vs. D? Both seem to be number of features (maybe Q is input dim)
D = 2  # I think Q is num of features, D is num of target classes (output dim). Gonna try just one class
K_test = 5  # Number of MC samples for aleatoric uncertainty
# nb_reps = 3  # Not sure if this is necessary
batch_size = 1024
l = 1e-4
T = 5  # Number of MC passes

X, Y = make_classification(n_samples=N+val_size, n_features=Q, n_redundant=0, n_classes=D,
                           n_informative=Q, n_clusters_per_class=1)

from tensorflow.keras.utils import to_categorical
Y = Y.astype('float64')
Y = to_categorical(Y)

# ======================================================================================================================
from tensorflow.keras.utils import plot_model
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

results = []
X_train, Y_train = X[:N], Y[:N]
X_val, Y_val = X[N:], Y[N:]
print('training')
model, ELBO = fit_model(nb_epoch, X_train, Y_train, input_shape=X_train.shape, T=T, D=D)
plot_model(model, to_file='graph.png', show_shapes=True)

results = model.predict(X_val)
aleatoric_uncertainties = np.reshape(results[0][:, D:], (-1))
logits = results[0][:, 0:D]

softmax = results[1]
#
# print('predicting')
# MC_samples = []
# for k in range(K_test):
#     print('predicting MC_sample: ', k)
#     MC_samples.append(model.predict(X_val))
# MC_samples = np.array(MC_samples)
#
# results = []
# # get results for multiple N
# for N, nb_epoch in zip(Ns, nb_epochs):
#     # repeat exp multiple times
#     rep_results = []
#     for i in range(nb_reps):
#         # X, Y = gen_data(N + nb_val_size)
#         X_train, Y_train = X[:N], Y[:N]
#         X_val, Y_val = X[N:], Y[N:]
#         model, ELBO = fit_model(nb_epoch, X_train, Y_train, input_shape=X_train.shape, T=T, D=D)
#         print('predicting')
#         # MC_samples = np.array([model.predict(X_val) for _ in range(K_test)])
#         MC_samples = []
#         for k in range(K_test):
#             print('predicting MC_sample: ', k)
#             MC_samples.append(model.predict(X_val))
#         MC_samples = np.array(MC_samples)
#         print('testing')
#         pppp, rmse = test(Y_val, MC_samples)  # per point predictive probability
#         means = MC_samples[:, :, :D]  # K x N
#         epistemic_uncertainty = np.var(means, 0).mean(0)
#         logvar = np.mean(MC_samples[:, :, D:], 0)
#         aleatoric_uncertainty = np.exp(logvar).mean(0)
#         ps = np.array([K.eval(layer.p) for layer in model.layers if hasattr(layer, 'p')])
#         # plot(X_train, Y_train, X_val, Y_val, means)
#         rep_results += [(rmse, ps, aleatoric_uncertainty, epistemic_uncertainty)]
#     test_mean = np.mean([r[0] for r in rep_results])
#     test_std_err = np.std([r[0] for r in rep_results]) / np.sqrt(nb_reps)
#     ps = np.mean([r[1] for r in rep_results], 0)
#     aleatoric_uncertainty = np.mean([r[2] for r in rep_results])
#     epistemic_uncertainty = np.mean([r[3] for r in rep_results])
#     print(N, nb_epoch, '-', test_mean, test_std_err, ps, ' - ', aleatoric_uncertainty**0.5, epistemic_uncertainty**0.5)
#     sys.stdout.flush()
#     results += [rep_results]

