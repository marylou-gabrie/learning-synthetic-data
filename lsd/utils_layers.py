import numpy as np
from keras import backend as K
from keras.constraints import Constraint
from keras.engine.topology import Layer
from keras.layers.core import Activation
from keras.utils.generic_utils import get_custom_objects


def binary(x):
    return K.sign(x)


get_custom_objects().update({'binary': Activation(binary)})


class Probit(Activation):

    def __init__(self, activation):
        super(Probit, self).__init__(activation)
        self.__name__ = 'probit'


def probit(x):
    return K.sign(x)


get_custom_objects().update({'probit': Probit(probit)})


class HardTanh(Activation):

    def __init__(self, activation):
        super(HardTanh, self).__init__(activation)
        self.__name__ = 'hardtanh'


def hardtanh(x):
    return 2 * K.hard_sigmoid(2.5 * x) - 1


get_custom_objects().update({'hardtanh': HardTanh(hardtanh)})


class GaussianNoise_(Layer):
    """Apply additive zero-centered Gaussian noise also at test time.
    This is to have noisy datasets genreted by datamodels.
    # Arguments
        stddev: float, standard deviation of the noise distribution.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

    def __init__(self, stddev, **kwargs):
        super(GaussianNoise_, self).__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs):
        return inputs + K.random_normal(shape=K.shape(inputs),
                                        mean=0.,
                                        stddev=self.stddev)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(GaussianNoise_, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Diagonal(Constraint):
    def __init__(self):
        super(Diagonal, self).__init__()

    def __call__(self, w):
        mask = np.zeros(w.shape)
        np.fill_diagonal(mask, 1.)
        return w * mask


get_custom_objects().update({'Diagonal': Diagonal})
