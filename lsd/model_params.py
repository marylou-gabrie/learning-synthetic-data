import keras
import numpy as np

from lsd.options import Prior
from lsd.utils_layers import GaussianNoise_, Diagonal
from keras.initializers import RandomNormal
from keras.layers import GaussianNoise
from keras.layers.core import Dense, Activation
from keras.models import Sequential


class ModelParams(object):

    def __init__(self, layer_sizes, activations, layer_noises=[]):
        super(ModelParams, self).__init__()

        self.n_layers = len(layer_sizes) - 1
        self.layer_sizes = layer_sizes
        self.activations = activations
        if layer_noises == []:
            self.layer_noises = ([0] * (len(layer_sizes) - 1))
        else:
            self.layer_noises = layer_noises


class TraineeModelParams(ModelParams):

    def __init__(self, layer_sizes, activations, layer_noises=[],
                 coef_weights_init=1., layer_free=[], use_bias=False):
        super(TraineeModelParams, self).__init__(layer_sizes,
                                                 activations, layer_noises)

        self.n_features = layer_sizes[0]
        self.coef_weights_init = coef_weights_init
        self.layer_free = ([0] * (len(layer_sizes) - 1)) if layer_free == [] else layer_free
        self.use_bias = use_bias

    def add_usv_layer(self, model, input_shape, output_shape, coef_weights_init):
        W_init = np.random.randn(input_shape, output_shape) / (np.sqrt(input_shape))
        W_init *= coef_weights_init
        U, S, V = np.linalg.svd(W_init)
        mask = np.zeros(W_init.shape)
        np.fill_diagonal(mask, 1.)
        S = mask.dot(np.diag(np.hstack((S,[0] * (output_shape - input_shape)))))

        model.add(Dense(input_shape,
                        activation='linear',
                        use_bias=False,
                        weights=[U, ],
                        input_shape=(input_shape,),
                        trainable=False))
        model.add(Dense(output_shape,
                        activation='linear',
                        use_bias=False,
                        weights=[S, ],
                        kernel_constraint=Diagonal(),
                        input_shape=(input_shape,)))
        model.add(Dense(output_shape,
                        activation='linear',
                        use_bias=False,
                        weights=[V, ],
                        input_shape=(output_shape,),
                        trainable=False))

    def add_to_model(self, model, output_shape, input_shape, layer, coef_weights_init):
        '''
        Function adding block of 
            - matrix multiplication
            - noise
            - activation
        '''
        if self.layer_free[layer] == 1:
            self.add_usv_layer(model, input_shape, output_shape, coef_weights_init)
            if self.use_bias:
                print(RuntimeWarning("USV layer {:d} don't have a bias".format(layer + 1)))
        else:
            model.add(Dense(
                output_shape,
                activation="linear",
                use_bias=self.use_bias,
                kernel_initializer=RandomNormal(mean=0.0, 
                    stddev=coef_weights_init / np.sqrt(input_shape), seed=None),
                input_shape=(input_shape,)
            ))

        # using homemade GaussianNoise_ to have noise in prediction
        model.add(GaussianNoise(np.sqrt(self.layer_noises[layer])))
        model.add(Activation(self.activations[layer]))

    def keras_model(self):
        '''
        has noise outside activation, to use only for regularization
        '''
        model = Sequential()

        for layer in range(self.n_layers):
            self.add_to_model(
                model,
                self.layer_sizes[layer + 1],
                self.layer_sizes[layer],
                layer,
                self.coef_weights_init
                )

        return model


class DataModelParams(ModelParams):

    def __init__(self, layer_sizes, activations, layer_noises=[], 
                 fraction_relevant=1., prior=(Prior.NORMAL, 0, 1), weights=[]):
        super(DataModelParams, self).__init__(layer_sizes, activations, layer_noises)

        self.fraction_relevant = fraction_relevant
        self.prior = prior
        self.weights = weights

    def generate_weights(self):
        weights = []

        for layer in range(self.n_layers):

            if self.activations[layer] == 'binary':
                W = np.random.randn(self.layer_sizes[layer], 1)
                if self.layer_sizes[layer + 1] != 2:
                    raise ValueError("Size of layer at output of binary should be 2")

            elif layer > 0 and self.activations[layer - 1] == 'binary':
                raise ValueError("Binary should only be added as a last layer.")

            else:
                W = np.random.randn(self.layer_sizes[layer], self.layer_sizes[layer + 1])

            W /= np.sqrt(self.layer_sizes[layer])

            weights.append(W)

        self.weights = weights

        return weights

    def generate_data_raw(self, weights, n_samples):
        """
        Goes through the feed forward generation, the output and input are 
        then inverted in the Decoder class.
        """
        input_size = self.layer_sizes[0]

        if self.prior[0] == Prior.NORMAL:
            _input = self.prior[1] + np.sqrt(self.prior[2]) * np.random.randn(n_samples, input_size)
        elif self.prior[0] == Prior.GB:
            _input = self.prior[2] + np.sqrt(self.prior[3]) * np.random.randn(n_samples, input_size)
            _input *= (np.random.rand(n_samples, input_size) < self.prior[1])
        elif self.prior[0] == Prior.BERNOULLI:
            _input = 2. * (np.random.rand(n_samples, input_size) < self.prior[1]) - 1.
        else:
            raise ValueError("Prior not understood !")

        if weights == [] :
            print("no weights given to datamodel, returning only inputs !")
            return _input, 0
        else:
            model = self.keras_model(weights)

            _output = model.predict(_input)

            return _input, _output

    def transform_data_for_classif(self, Y, trainee):
        if Y.shape[1] != 1:
            raise ValueError("Classification is only possible with 1 single bit relevant."
                             "For generative adjust size of code or fraction relevant.")
        if self.prior[0] != Prior.BERNOULLI:
            print("considering true class as the sign of the conitnuous bit")
            Y = np.sign(Y)
        if trainee.layer_sizes[-1] != 2 or trainee.activations[-1] != 'softmax':
            raise ValueError("Only binary classification for categorical_crossentropy:"
                             " ouput should be a softmax of size 2")

        y = Y.squeeze()
        Y = keras.utils.to_categorical(.5 * (y + 1), num_classes=2)
        return Y

    def add_to_model(self, model, output_shape, input_shape, layer, weights):
        '''
        Function adding block of 
            - matrix multiplication
            - noise
            - activation
        '''

        model.add(Dense(
            output_shape,
            activation="linear",
            use_bias=False,
            weights=weights,
            input_shape=(input_shape,)
        ))

        # using homemade GaussianNoise_ to have noise in prediction
        model.add(GaussianNoise(np.sqrt(self.layer_noises[layer])))
        model.add(Activation(self.activations[layer]))

    def keras_model(self, weights):
        '''
        Note that noise is added inside activations
        '''
        model = Sequential()

        for layer in range(self.n_layers):
            if layer == self.n_layers - 1 and self.activations[-1] == 'binary':
                # binary -> break loop 1 entry before the end and add custom ouput layer
                self.add_to_model(
                    model=model,
                    output_shape=1,
                    input_shape=self.layer_sizes[-2],
                    layer=layer,
                    weights=[weights[-1], ]
                )
                break

            self.add_to_model(
                model=model,
                output_shape=self.layer_sizes[layer + 1],
                input_shape=self.layer_sizes[layer],
                layer=layer,
                weights=[weights[layer], ]
            )

        return model

