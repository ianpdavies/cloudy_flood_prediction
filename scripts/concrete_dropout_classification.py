import sys
import numpy as np
np.random.seed(0)

from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import initializers
# from tensorflow.keras.engine import InputSpec
from sklearn.datasets import make_classification
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Dropout, \
    concatenate, BatchNormalization, Activation, Lambda, InputSpec, Wrapper
from tensorflow_probability import distributions
from tensorflow.keras import models
import os

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
class ConcreteDropout(layers.Wrapper):
    """This wrapper allows to learn the dropout probability for any given input Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = layers.InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                            shape=(1,),
                                            initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                            trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        assert len(input_shape) == 2, 'this wrapper only supports Dense layers'
        input_dim = np.prod(input_shape[-1])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=K.shape(x))
        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)


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

def fit_model(nb_epoch, X, Y, input_shape, T, D):
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

    means = Dense(D)(x)
    log_var_pre = Dense(1)(x)
    log_var = Activation('softplus', name='log_var')(log_var_pre)  # What does this do?
    means_variance = concatenate([means, log_var], name='means_variance')
    softmax_output = Activation('softmax', name='softmax_output')(means)
    # model = models.Model(inputs=inp, outputs=[means_variance, softmax_output])
    model = models.Model(inputs=inp, outputs=means_variance)

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
        iterable = K.variable(np.ones(T))
        dist = distributions.Normal(loc=K.zeros_like(log_std), scale=log_std)
        monte_carlo_results = K.map_fn(gaussian_categorical_crossentropy(true, mean, dist,
                                                                         undistorted_loss, D), iterable,
                                       name='monte_carlo_results')

        var_loss = K.mean(monte_carlo_results, axis=0) * undistorted_loss

        return var_loss + undistorted_loss + K.sum(logvar_dep, -1)

    def gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes):
        def map_fn(i):
            std_samples = dist.sample(1)
            distorted_loss = K.categorical_crossentropy(pred + std_samples[0], true,
                                                        from_logits=True)
            diff = undistorted_loss - distorted_loss
            return -K.elu(diff)

        return map_fn

    model.compile(optimizer='Adam',
                  # loss=heteroscedastic_categorical_crossentropy)
                  loss={'means_variance': heteroscedastic_categorical_crossentropy},
                        # 'softmax_output': 'categorical_crossentropy'},
                  # metrics={'softmax_output': 'categorical_accuracy'}
                  )
    hist = model.fit(X, Y, epochs=nb_epoch, batch_size=batch_size, verbose=1)
    loss = hist.history['loss'][-1]
    return model, -0.5 * loss  # return ELBO up to const.

# # ======================================================================================================================
# Create fake data

# Ns = [10, 25, 50, 100, 1000, 10000]
# Ns = np.array(Ns)
# nb_epochs = [2000, 1000, 500, 200, 20, 2]
N = 10000
nb_epoch = 20
val_size = np.ceil(N * 0.3).astype('int')
nb_features = 1024
Q = 3  # Q vs. D? Both seem to be number of features (maybe Q is input dim)
D = 2  # I think Q is num of features, D is num of target classes (output dim). Gonna try just one class
K_test = 5  # Number of MC samples for aleatoric uncertainty
# nb_reps = 3  # Not sure if this is necessary
batch_size = 1024
l = 1e-4
T = 5  # Number of MC passes

X, Y = make_classification(n_samples=N+val_size, n_features=Q, n_redundant=0, n_classes=2,
                           n_informative=Q, n_clusters_per_class=1)

from tensorflow.keras.utils import to_categorical
# Y = Y.astype('float64')
Y = to_categorical(Y)

# ======================================================================================================================
# Run
Ns = [N]
nb_epochs = [nb_epoch]
nb_reps = 1

results = []
# get results for multiple N
for N, nb_epoch in zip(Ns, nb_epochs):
    # repeat exp multiple times
    rep_results = []
    for i in range(nb_reps):
        # X, Y = gen_data(N + nb_val_size)
        X_train, Y_train = X[:N], Y[:N]
        X_val, Y_val = X[N:], Y[N:]
        model, ELBO = fit_model(nb_epoch, X_train, Y_train, input_shape=X_train.shape, T=T, D=D)
        print('predicting')
        # MC_samples = np.array([model.predict(X_val) for _ in range(K_test)])
        MC_samples = []
        for k in range(K_test):
            print('predicting MC_sample: ', k)
            MC_samples.append(model.predict(X_val))
        MC_samples = np.array(MC_samples)
        print('testing')
        pppp, rmse = test(Y_val, MC_samples)  # per point predictive probability
        means = MC_samples[:, :, :D]  # K x N
        epistemic_uncertainty = np.var(means, 0).mean(0)
        logvar = np.mean(MC_samples[:, :, D:], 0)
        aleatoric_uncertainty = np.exp(logvar).mean(0)
        ps = np.array([K.eval(layer.p) for layer in model.layers if hasattr(layer, 'p')])
        # plot(X_train, Y_train, X_val, Y_val, means)
        rep_results += [(rmse, ps, aleatoric_uncertainty, epistemic_uncertainty)]
    test_mean = np.mean([r[0] for r in rep_results])
    test_std_err = np.std([r[0] for r in rep_results]) / np.sqrt(nb_reps)
    ps = np.mean([r[1] for r in rep_results], 0)
    aleatoric_uncertainty = np.mean([r[2] for r in rep_results])
    epistemic_uncertainty = np.mean([r[3] for r in rep_results])
    print(N, nb_epoch, '-', test_mean, test_std_err, ps, ' - ', aleatoric_uncertainty**0.5, epistemic_uncertainty**0.5)
    sys.stdout.flush()
    results += [rep_results]


# ======================================================================================================================
# Using image data instead of fake data

# sys.path.append('../')
# from CPR.configs import data_path
# from CPR.utils import preprocessing, train_val
#
# img = '4514_LC08_027033_20170826_1'
# pctl = 40
#
# feat_list_new = ['GSW_maxExtent', 'GSW_distExtent', 'GSW_perm', 'aspect', 'curve', 'developed', 'elevation',
#                  'forest', 'hand', 'other_landcover', 'planted', 'slope', 'spi', 'twi', 'wetlands', 'flooded']
#
# data_train, data_vector_train, data_ind_train, feat_keep = preprocessing(data_path, img, pctl, gaps=False)
# feat_list_keep = [feat_list_new[i] for i in feat_keep]  # Removed if feat was deleted in preprocessing
# perm_index = feat_list_keep.index('GSW_perm')
# flood_index = feat_list_keep.index('flooded')
# data_vector_train[data_vector_train[:, perm_index] == 1, flood_index] = 0  # Remove flood water that is perm water
# data_vector_train = np.delete(data_vector_train, perm_index, axis=1)  # Remove perm water column
# training_data, validation_data = train_val(data_vector_train, holdout=HOLDOUT)
# shape = data_vector_train.shape
# X_train, Y_train = training_data[:, 0:shape[1]-1], training_data[:, shape[1]-1]
# X_val, Y_val = validation_data[:, 0:shape[1]-1], validation_data[:, shape[1]-1]
#
# nb_epoch = 20
#
# nb_features = 1024
# Q = shape[1]
# D = 2
# K_test = 20
# # nb_reps = 3
# batch_size = 8192
# l = 1e-4