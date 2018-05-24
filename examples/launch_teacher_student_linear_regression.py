import numpy.random

from lsd.options import Prior
from lsd.set_params import set_param, launch_training
from keras.optimizers import SGD


numpy.random.seed(276635)

exp_type = "teacher-student"
exp_description = "linear"
display = 1
save = 1
test = True  # in the case of Gaussian prior and linear student, compare with replica and analytical entropies

## model architecture params
teacher_activations = ['linear']
teacher_layer_noises = [0.01]
teacher_layer_sizes = [100, 4]
fraction_relevant = 1.  # fraction of input actually used to generate labels
prior = (Prior.NORMAL, 0., 1.)

student_layer_sizes = [100, 100, 100, 100, 4]
student_activations = ['linear', 'linear', 'linear', 'linear']
student_layer_noises = [0., 0., 0., 0.]  # training time noise, inside activation
student_coef_weights_init = 1.
student_layer_free = [1, 1, 1, 0]  # flag = 1 for USV-layer, flag = 0 for unconstrained

## entropy computation parameters
compute_mi_up_to = 3  # numbers of layers of the student to compute entropy of
compute_info_every = 1  # period of computation of the entropy
mi_noise = 1e-5  # entropy computation final layer noise
inter_mi_noise = 0  # entropy computation intermediate layer noise

## dnner parameters for entropy computation
max_iter = 1000
tol = 1e-8
persist = True
use_vamp = True

## training parameters
gamma = 100  # nb training samples = gamma * teacher_layer_sizes[0] 
lr = 0.01
updates = 1  # to specify training time with parameter updates rather than epochs
min_epochs = 100  # minimum number of epochs that will be run even if "updates" corresponds to fewer
optimizer = SGD
loss = 'MSE'
batchsize = 50
validation_split = .5


opt, folder, save = set_param(exp_type,
                              exp_description,
                              save,
                              teacher_layer_sizes,
                              teacher_activations,
                              teacher_layer_noises,
                              fraction_relevant,
                              prior,
                              student_layer_sizes,
                              student_activations,
                              student_layer_noises,
                              student_coef_weights_init,
                              student_layer_free,
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
                              prefix="",  # path to data folder
                              inter_mi_noise=inter_mi_noise,
                              max_iter=max_iter,
                              tol=tol,
                              persist=persist,
                              use_vamp=use_vamp)


X, Y, model, history, callbacks = launch_training(opt, folder, save,
                                                  exp_description,
                                                  display=display,
                                                  test=test)
