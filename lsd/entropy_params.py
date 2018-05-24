import numpy as np
from dnner.activations import Linear, Probit, ReLU, HardTanh  # LeakyReLU
from dnner.priors import Normal, Bimodal, SpikeSlab

from lsd.decoder_encoder.encoder_decoder_utils import Decoder
from lsd.options import Prior


class Information(object):
    def __init__(self, up_to, mi_noise, inter_mi_noise=0., compute_info_every=1,
                 max_iter=1000, tol=1e-6, persist=True, use_vamp=False):
        self.compute_info_every = compute_info_every
        self.up_to = up_to
        self.mi_noise = mi_noise
        self.inter_mi_noise = inter_mi_noise
        self.max_iter = max_iter
        self.tol = tol
        self.persist = persist
        self.use_vamp = use_vamp
        self.mi_layers = None
        self.dnner_weights = []

    def match_mi_layers(self, datamodel, trainee):
        '''
        Returns the list [prior, activation1, activation2, ..] according to 
        the prior, the generative model (if present) and the trainee for 
        layers until the activation after the "up_to"th Activation layer of 
        the trainee ! (i.e. counting layer as we usually do, and not necessary 
        as it is implemented in Keras).
        '''

        self.mi_layers = []

        # taking care of the prior
        if datamodel.prior[0] == Prior.NORMAL:
            self.mi_layers.append(Normal(datamodel.prior[1], datamodel.prior[2]))
        elif datamodel.prior[0] == Prior.BERNOULLI:
            self.mi_layers.append(Bimodal(datamodel.prior[1]))
        elif datamodel.prior[0] == Prior.GB:
            self.mi_layers.append(SpikeSlab(datamodel.prior[1],
                                        datamodel.prior[2],
                                        datamodel.prior[3]))
        else:
            raise ValueError("Prior not available in dnner")


        # taking care of generative/datamodel
        if isinstance(datamodel, Decoder):
            for layer in range(len(datamodel.activations)):
                if datamodel.activations[layer] == 'relu':
                    self.mi_layers.append(ReLU(datamodel.layer_noises[layer]))
                    # self.mi_layers.append(LeakyReLU(datamodel.layer_noises[layer], 0))
                elif datamodel.activations[layer] == 'linear':
                    self.mi_layers.append(Linear(datamodel.layer_noises[layer]))
                elif datamodel.activations[layer] == 'probit':
                    self.mi_layers.append(Probit(datamodel.layer_noises[layer]))
                else:
                    raise ValueError("interface not available in dnner")

            self.dnner_weights = [(weight.shape[1]/weight.shape[0],
                                np.linalg.eigvalsh(weight.dot(weight.T))) for
                                weight in datamodel.weights]

        # taking care of student/encoder (inter_mi_noise)
        for layer in range(self.up_to - 1):
            if trainee.activations[layer] == 'relu':
                self.mi_layers.append(ReLU(self.inter_mi_noise))
                # self.mi_layers.append(LeakyReLU(datamodel.layer_noises[layer], 0))
            elif trainee.activations[layer] == 'linear':
                self.mi_layers.append(Linear(self.inter_mi_noise))
            elif trainee.activations[layer] == 'probit':
                self.mi_layers.append(Probit(self.inter_mi_noise))
            elif trainee.activations[layer] == 'hardtanh':
                self.mi_layers.append(HardTanh(self.inter_mi_noise))
            else:
                raise ValueError("Activation {:d} not available in dnner")

        # passing mi_noise to last layer
        if trainee.activations[self.up_to - 1] == 'relu':
            self.mi_layers.append(ReLU(self.mi_noise))
            # self.mi_layers.append(LeakyReLU(self.mi_noise, 0))
        elif trainee.activations[self.up_to - 1] == 'linear':
            self.mi_layers.append(Linear(self.mi_noise))
        elif trainee.activations[self.up_to - 1] == 'probit':
            self.mi_layers.append(Probit(self.mi_noise))
        elif trainee.activations[self.up_to - 1] == 'hardtanh':
            self.mi_layers.append(HardTanh(self.mi_noise))
        else:
            raise ValueError("Activation not available in dnner")

        if len(self.mi_layers) < 2:
            raise ValueError("Issue matching MI layers")
