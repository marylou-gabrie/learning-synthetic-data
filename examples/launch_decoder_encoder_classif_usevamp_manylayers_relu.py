import numpy.random

from lsd.options import Prior
from lsd.set_params import set_param, launch_training
from keras.optimizers import SGD

numpy.random.seed(276635)

exp_type = "decoder-encoder"
exp_description = "biary_classification"
display = 1
save = 1

## model architecture params
fraction_relevant = .05  # class = sign(first spin of first layer in decoder)
decoder_layer_sizes = [20, 100]
decoder_activations = ['linear']
decoder_layer_noises = [.01]
prior = (Prior.NORMAL, 0, 1)

encoder_coef_weights_init = 1.
encoder_layer_sizes = [100, 200, 100, 50, 20, 2]
encoder_activations = ['linear', 'relu', 'linear', 'relu', 'softmax']
encoder_layer_noises = [0., 0., 0., 0., 0.]  # training time noise, inside activation
encoder_free_layer = [1, 1, 1, 1, 0]  # flag = 1 for USV-layer, flag = 0 for unconstrained

## entropy computation parameters
compute_mi_up_to = 4  # numbers of layers of the student to compute entropy of
compute_info_every = 2  # period of computation of the entropy
mi_noise = 1e-5  # entropy computation final layer noise
inter_mi_noise = 0  # entropy computation intermediate layer noise

## dnner parameters for entropy computation
max_iter = 1000
tol = 1e-6
persist = True
use_vamp = True

## training parameters
gamma = 100.  # nb training samples = gamma * teacher_layer_sizes[0] 
lr = 0.01
updates = 1  # to specify training time with parameter updates rather than epochs
min_epochs = 100  # minimum number of epochs that will be run even if "updates" corresponds to fewer
loss = 'categorical_crossentropy'
batchsize = 100
validation_split = .5
optimizer = SGD


opt, folder, save = set_param(
                            exp_type,
                            exp_description,
                            save,
                            decoder_layer_sizes,
                            decoder_activations,
                            decoder_layer_noises,
                            fraction_relevant,
                            prior,
                            encoder_layer_sizes,
                            encoder_activations,
                            encoder_layer_noises,
                            encoder_coef_weights_init,
                            encoder_free_layer,
                            mi_noise,
                            compute_mi_up_to,
                            compute_info_every,
                            gamma,
                            lr,
                            updates,
                            min_epochs,
                            optimizer,
                            loss,
                            batchsize,
                            validation_split,
                            prefix="", # path to data folder
                            inter_mi_noise=inter_mi_noise,
                            max_iter=max_iter,
                            tol=tol,
                            persist=persist,
                            use_vamp=use_vamp)


X, Y, model, history, callbacks = launch_training(opt, folder, save,
                                                  exp_description,
                                                  display=display,
                                                  test=False)
