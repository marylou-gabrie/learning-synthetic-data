import dnner
import h5py
import keras.callbacks
import numpy as np
import pickle
import traceback

from dnner.activations import Linear
from lsd.options import Prior
from lsd.decoder_encoder.encoder_decoder_utils import Decoder
from lsd.teacher_student.teacher_student_utils import Teacher
from lsd.utils_io import save_mutual_info, load_opt
from keras.layers.core import Dense


def load_mi(folder, opt=None, recomputed_mi_noise=None, recomputed_inter_mi_noise=None):
    opt = load_opt(folder) if opt is None else opt
    filename = folder + "/MI"

    if recomputed_mi_noise is not None:
        opt.information.mi_noise = recomputed_mi_noise
        filename += "_minoise{:0.1E}".format(recomputed_mi_noise)
    if recomputed_inter_mi_noise is not None:
        opt.information.inter_mi_noise = recomputed_inter_mi_noise
        filename += "_internoise{:0.1E}".format(recomputed_inter_mi_noise)

    mutual_info = MIHistory(opt.information)
    mutual_info.on_train_begin()


    f = h5py.File(filename + ".hdf5", "r")
    mutual_info.IXT = np.array(f["IXT"])
    mutual_info.HT = np.array(f["HT"])
    mutual_info.V = np.array(f["V"])
    mutual_info.Eigvals = [np.array(f["Eigvals/" + key]) for
                            key in f["Eigvals"]]
    f.close()
    return mutual_info


class MIHistory(keras.callbacks.Callback):

    def __init__(self, information, resume_from=None, folder=None):
        super(MIHistory, self).__init__()
        self.layers = information.mi_layers
        self.n_layers = information.up_to   # only counting layers for computation
        self.compute_info_every = information.compute_info_every
        self.dnner_weights = information.dnner_weights
        self.always_include = len(information.dnner_weights) + 1
        self.resume_from = resume_from
        self.folder = folder
        self.max_iter = information.max_iter
        self.tol = information.tol
        self.persist = information.persist
        self.use_vamp = information.use_vamp

    def on_train_begin(self, logs={}):
        if self.resume_from is not None:
            f = h5py.File(self.resume_from + "/MI.hdf5", "r")
            self.IXT = np.array(f["IXT"])
            self.HT = np.array(f["HT"])
            self.V = np.array(f["V"])
            self.Eigvals = [np.array(f["Eigvals/" + key]) for
                                    key in f["Eigvals"]]
            f.close()
            self.start_at = pickle.load(open(self.resume_from + "/MI_start_at.pickle", "rb"))
        else:
            self.IXT = np.empty((0, self.n_layers))
            self.HT = np.empty((0, self.n_layers))
            self.V = np.empty((0, self.n_layers))
            self.start_at = None

            try:
                self.Eigvals = [np.empty((0,layer.input_shape[1])) for layer in
                                self.model.layers if (isinstance(layer, Dense) and layer.trainable)]
            except:
                #initialization when loading from scratch
                self.Eigvals = np.empty((self.n_layers))

    def on_epoch_end(self, epoch, logs={}):
        if self.n_layers == 0:
            return

        # list of tuples (alpha, eigvals)
        weights = self.dnner_weights[:]
        # list of weight matrices
        weights_layers = [layer.get_weights()[0] for layer in self.model.layers
                          if (isinstance(layer, Dense) and layer.trainable)]

        for layer in range(self.n_layers):
            weight = weights_layers[layer]
            alpha = float(weight.shape[1]) / weight.shape[0]
            eigvals = np.linalg.eigvalsh(weight.dot(weight.T))
            weights.append((alpha, eigvals))
            self.Eigvals[layer] = np.vstack((self.Eigvals[layer],
                                            eigvals.reshape(1, len(eigvals))))

        if self.compute_info_every > 0 and epoch % self.compute_info_every == 0:
            try:
                entropies, extras = dnner.compute_all(self.layers, weights,
                                            always_include=self.always_include,
                                            max_iter=self.max_iter, tol=self.tol,
                                            use_vamp=self.use_vamp,
                                            start_at=self.start_at,
                                            verbose=1)

                self.start_at = extras if self.persist else None
                self.IXT = np.vstack((self.IXT, [extra["mi"] for extra in extras]))
                self.HT = np.vstack((self.HT, entropies))
                self.V = np.vstack((self.V, [extra["mmse"] for extra in extras]))

                for layer in range(self.n_layers):
                    print(' \n' + 4 * '\t' + ' - IXT%d %05.4f - h(T%d) %05.4f - v%d %05.4f '
                          % (layer + 1, self.IXT[-1, layer], layer + 1,
                             self.HT[-1, layer], layer + 1, self.V[-1, layer])
                            )

            except Exception as e:
                with open(self.folder + "/dnner_error_{:d}.txt".format(epoch), "w") as logfile:
                    traceback.print_exc(file=logfile)
                traceback.print_exc()

                to_keep = {"weights": weights, "layers": self.layers}
                with open(self.folder + "/dnner_layers_upon_error_{:d}.pickle".format(epoch), "wb") as f:
                    pickle.dump(to_keep, f)

                self.IXT = np.vstack((self.IXT, [float('nan')] * self.n_layers))
                self.HT = np.vstack((self.HT, [float('nan')] * self.n_layers))
                self.V = np.vstack((self.V, [float('nan')] * self.n_layers))

        return


def on_recompute_begin(mutual_info_, mutual_info):
    mutual_info_.IXT = np.empty((0, mutual_info_.n_layers))
    mutual_info_.HT = np.empty((0, mutual_info_.n_layers))
    mutual_info_.V = np.empty((0, mutual_info_.n_layers))
    mutual_info_.start_at = None
    mutual_info_.Eigvals = mutual_info.Eigvals
    mutual_info_.dnner_weights = mutual_info.dnner_weights
    mutual_info_.use_vamp = mutual_info.use_vamp
    mutual_info_.max_iter = mutual_info.max_iter
    mutual_info_.tol = mutual_info.tol
    mutual_info_.persist = mutual_info.persist
    mutual_info_.always_include = mutual_info.always_include
    mutual_info_.n_layers = mutual_info.n_layers

    return mutual_info_


def on_epoch_recompute(mutual_info_, epoch, alphas):
    weights = [decoder_weights for decoder_weights in mutual_info_.dnner_weights]

    for layer in range(mutual_info_.n_layers):
        alpha = alphas[layer]
        eigvals = mutual_info_.Eigvals[layer][epoch, :]
        weights.append((alpha, eigvals))

    entropies, extras = dnner.compute_all(mutual_info_.layers, weights,
        max_iter=mutual_info_.max_iter,
        tol=mutual_info_.tol,
        use_vamp=mutual_info_.use_vamp,
        verbose=0,
        start_at=mutual_info_.start_at,
        always_include=mutual_info_.always_include)

    mutual_info_.start_at = extras if mutual_info_.persist else None
    mutual_info_.IXT = np.vstack((mutual_info_.IXT, [extra["mi"] for extra in extras]))
    mutual_info_.HT = np.vstack((mutual_info_.HT, entropies))
    mutual_info_.V = np.vstack((mutual_info_.V, [extra["mmse"] for extra in extras]))

    for layer in range(mutual_info_.n_layers):
        print(' \n' + 4 * '\t' + ' - IXT%d %05.4f - h(T%d) %05.4f - v%d %05.4f '
              % (layer + 1, mutual_info_.IXT[-1, layer], layer + 1,
                     mutual_info_.HT[-1, layer], layer + 1, mutual_info_.V[-1, layer])
                )


def recompute_mi(opt, mutual_info, up_to='all', recompute_every=1,
                 recompute_mi_noise=None,
                 recompute_inter_mi_noise=None):
    """
    Will recompute the mutual information.
    If noises are set to None then the original values are retrieved.
    """
    information_ = opt.information
    trainee = opt.trainee
    datamodel = opt.datamodel

    if up_to == 'all':
        up_to = len(trainee.layer_sizes) - 1

    information_.compute_info_every = recompute_every
    information_.up_to = up_to
    if recompute_mi_noise is not None:
        information_.mi_noise = recompute_mi_noise
    if recompute_inter_mi_noise is not None:
        information_.inter_mi_noise = recompute_inter_mi_noise
        print("changing inter mi noise")
    information_.match_mi_layers(datamodel, trainee)
    print(information_.mi_layers[1].var_noise)

    mutual_info_ = MIHistory(information_)

    alphas = [trainee.layer_sizes[i + 1] / trainee.layer_sizes[i]
              for i in range(len(trainee.layer_sizes) - 1)]

    mutual_info_ = on_recompute_begin(mutual_info_, mutual_info)

    for epoch in range(opt.training.epochs):
        if mutual_info_.compute_info_every > 0 and epoch % mutual_info_.compute_info_every == 0:
            print("epoch {:d} - entropies ".format(epoch), )
            on_epoch_recompute(mutual_info_, epoch, alphas)

    return mutual_info_


class MIHistoryExact(keras.callbacks.Callback):
    '''
    Considering the fully linear case with normal prior
    '''

    def __init__(self, information, datamodel, trainee):
        super(MIHistoryExact, self).__init__()
        '''
        Initializations take into account that the Callback can be initialized 
        without a model (loaded from .hdf5).
        '''
        self.layers = information.mi_layers
        self.mi_noise = information.mi_noise
        self.inter_mi_noise = information.inter_mi_noise
        self.n_layers = information.up_to #only counting layers for which will compute
        self.compute_info_every = information.compute_info_every
        self.datamodel = datamodel
        self.trainee = trainee

        if datamodel.prior != (Prior.NORMAL, 0, 1):
            raise ValueError("Prior must be normal(0,1) when calling exact computation. "
                            "Change kwarg test to False in train.")
        elif (
                (list(np.unique(datamodel.activations + trainee.activations[:information.up_to])) != ['linear']
                    and isinstance(datamodel, Decoder)
                    )
                or 
                (list(np.unique(trainee.activations[:information.up_to])) != ['linear']
                    and isinstance(datamodel, Teacher)
                    )
                ):
            raise ValueError("Activations should only be linear when calling exact computation. "
                            "Change kwarg test to False in train.")
        elif len(datamodel.weights) > 1 and isinstance(datamodel, Decoder):
            raise ValueError("Decoder supposed to have only one layer when calling exact computation. "
                            "Change kwarg test to False in train.")

    def on_train_begin(self, logs={}):
        self.IXT = np.empty((0, self.n_layers))
        self.HT = np.empty((0, self.n_layers))

    def on_epoch_end(self, epoch, logs={}):
        if self.compute_info_every > 0 and epoch % self.compute_info_every == 0:
            if isinstance(self.datamodel, Decoder):
                print("Decoder !")
                decoder_weight = self.datamodel.weights[0]
                norm = decoder_weight.shape[0]
                var = decoder_weight.T.dot(decoder_weight) + \
                        self.datamodel.layer_noises[-1] * np.eye(decoder_weight.shape[1])
                alpha = decoder_weight.shape[1] / decoder_weight.shape[0]
            else:
                print("Teacher !")
                alpha = 1.
                var = np.eye(self.trainee.n_features)
                norm = self.trainee.n_features

            weights_dense_layers = [layer.get_weights()[0] for layer in self.model.layers
                            if (isinstance(layer, Dense))]
            weights_dense_layers_trainflag = [layer.trainable for layer in self.model.layers
                            if (isinstance(layer, Dense))]
            # reconstructing weights by performing necessary USV products
            weight_layers = []
            layer = 0
            while len(weight_layers) < self.n_layers:
                if weights_dense_layers_trainflag[layer]:
                    weight_layers.append(weights_dense_layers[layer])
                    layer += 1
                else:
                    weight = weights_dense_layers[layer].dot(weights_dense_layers[layer + 1])
                    weight = weight.dot(weights_dense_layers[layer + 2])
                    weight_layers.append(weight)
                    layer += 3

            entropies = []
            for layer in range(self.n_layers):
                weight = weight_layers[layer]
                alpha = alpha * weight.shape[1] / weight.shape[0]
                var = weight.T.dot(var).dot(weight)
                s, logdet = np.linalg.slogdet(
                            2 * np.pi * np.e 
                            * (var + self.mi_noise * np.eye(weight.shape[1]))
                            )
                entropy = .5 * logdet / norm
                entropies.append(entropy)

                # new line to allow to incorporate a different intermediary noise
                # variance being computed recursively
                var += self.inter_mi_noise * np.eye(weight.shape[1])

            self.HT = np.vstack((self.HT, entropies))

            for layer in range(self.n_layers):
                print(' \n' + 5 * '\t' + ' - h(T%d) exact %05.4f  '
                      % (layer + 1, self.HT[-1, layer])
                        )

            return


class MICheckpoint(keras.callbacks.Callback):
    def __init__(self, mutual_info, filepath, period=10):
        super(MICheckpoint, self).__init__()
        self.mutual_info = mutual_info
        self.filepath = filepath
        self.period = period

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.period == 0:
            save_mutual_info(self.mutual_info, self.filepath,
                             rewrite=True, save_all=True)
