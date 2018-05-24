import keras.backend as K
import matplotlib.pyplot as plt

from lsd.decoder_encoder.encoder_decoder_utils import Encoder, Decoder
from lsd.entropy_params import Information
from lsd.options import ExpOptions
from lsd.teacher_student.teacher_student_utils import Teacher, Student
from lsd.training_params import Training
from lsd.utils_io import fold_gen, save_opt, load_dataset, load_opt
from lsd.utils_plots import plot_setting, plot_All_multilayer, plot_info_history_exact
from lsd.utils_callbacks import MIHistoryExact

def set_param(
    exp_type,  # teacher-student or decoder-encoder
    exp_description,
    save,
    datamodel_layer_sizes,
    datamodel_activations,
    datamodel_layer_noises,
    fraction_relevant,
    prior,
    trainee_layer_sizes,
    trainee_activations,
    trainee_layer_noises,
    trainee_coef_weights_init,
    trainee_layer_free,
    mi_noise,
    up_to,
    compute_info_every,
    gamma,
    lr,
    updates,
    min_epochs,
    optimizer,
    loss,
    batchsize,
    validation_split,
    prefix="",
    inter_mi_noise=1e-10,
    max_iter=1000,
    tol=1e-6,
    persist=True,
    use_vamp=False,
    trainee_use_bias=False):

    folder =""
    if save > 0:
        folder = fold_gen(exp_type, exp_description, date=None, rewrite=False, prefix=prefix)

    n_samples = 2 * int(gamma * trainee_layer_sizes[0])
    epochs = max(int(updates * batchsize / (gamma * trainee_layer_sizes[0])), min_epochs)

    """
    All options are kept in opt, which contains
        - a datamodel,
        - a trainee,
        - a training
        - an information,
    The following lines pass the desired parameters values to the classes.
    """

    training = Training(
        lr=lr,
        epochs=epochs,
        loss=loss,
        optimizer=optimizer,
        n_samples=n_samples,
        batchsize=batchsize,
        validation_split=validation_split
    )

    information = Information(
        up_to, mi_noise,
        compute_info_every=compute_info_every,
        inter_mi_noise=inter_mi_noise,
        max_iter=max_iter,
        tol=tol,
        persist=persist,
        use_vamp=use_vamp
    )

    if compute_info_every > 0:
        if trainee_use_bias :
            raise ValueError("dnner does not consider the use of biases yet !")
        if up_to > 0 and any(trainee_layer_free[:up_to]) == 0:
            raise ValueError("entropy computation is valid only when training free layers")

    if exp_type == "teacher-student":

        datamodel = Teacher(
            datamodel_layer_sizes,
            datamodel_activations,
            layer_noises=datamodel_layer_noises,
            fraction_relevant=fraction_relevant,
            prior=prior
        )

        trainee = Student(
            trainee_layer_sizes,
            trainee_activations,
            layer_noises=trainee_layer_noises,
            coef_weights_init=trainee_coef_weights_init,
            layer_free=trainee_layer_free,
            use_bias=trainee_use_bias
        )

    elif exp_type == "decoder-encoder":

        datamodel = Decoder(
            datamodel_layer_sizes,
            datamodel_activations,
            layer_noises=datamodel_layer_noises,
            fraction_relevant=fraction_relevant,
            prior=prior
        )

        trainee = Encoder(
            trainee_layer_sizes,
            trainee_activations,
            layer_noises=trainee_layer_noises,
            coef_weights_init=trainee_coef_weights_init,
            layer_free=trainee_layer_free,
            use_bias=trainee_use_bias
        )

    opt = ExpOptions(
        datamodel=datamodel,
        trainee=trainee,
        training=training,
        information=information,
    )

    if isinstance(datamodel, Decoder) and training.loss == 'MSE':
        if int(datamodel.fraction_relevant * datamodel_layer_sizes[0]) != trainee_layer_sizes[-1]: 
            raise ValueError("The model training must output only relevant "
                             "dimensions.\nDecoder and encoder dimensions "
                             "passed are inconsistent with fraction_relevant parameter.")

    if save > 0:
        save_opt(opt, folder)
        plot_setting(opt, folder)

    return opt, folder, save


def launch_training(opt, folder, save, exp_description, test=False, display=1):
    datamodel = opt.datamodel
    trainee = opt.trainee
    training = opt.training
    information = opt.information

    K.clear_session()
    data_weights = datamodel.generate_weights()
    X, Y = datamodel.generate_data(data_weights, training.n_samples)
    if training.loss == 'categorical_crossentropy':
        Y = datamodel.transform_data_for_classif(Y, trainee)
     
    model, history, callbacks = training.train(X, Y, datamodel, trainee, information,
                                               save=save, folder=folder, test=test)

    if save > 0 and display == 1:
        plot_setting(opt, folder)
    if display == 1:
        fig_all = plot_All_multilayer(history, callbacks[0], folder=folder)
        if len(callbacks) > 1 and isinstance(callbacks[1], MIHistoryExact):
            print('Exact comparison will be plotted')
            fig_comp = plot_info_history_exact(callbacks[0], callbacks[1], folder=folder)
        plt.show(block=False)
    if save > 0:
        save_opt(opt, folder)
        plt.figure(fig_all.number)
        plt.savefig(folder + "/history-info_plane_" + exp_description + ".pdf")
        if len(callbacks) > 1 and isinstance(callbacks[1], MIHistoryExact):
            plt.figure(fig_comp.number)
            plt.savefig(folder + "/comparison_MIexact_" + exp_description + ".pdf")

    return X, Y, model, history, callbacks


def resume_training(folder, epochs, save, display=1):
    opt = load_opt(folder)
    initial_epoch = opt.training.epochs
    opt.training.epochs = epochs

    X, Y = load_dataset(folder)
    model, history, callbacks = opt.training.resume_train(X, Y, initial_epoch, 
                                                          opt.information,
                                                          save=save, folder=folder)

    if save > 0 and display == 1:
        plot_setting(opt, folder)
    if display == 1:
        plot_All_multilayer(history, callbacks[0], folder=folder)
        if len(callbacks) > 1 and isinstance(callbacks[1], MIHistoryExact):
            print('exact comparison will be plotted')
            plot_info_history_exact(callbacks[0], callbacks[1], folder=folder)
        plt.show(block=False)
    if save > 0:
        save_opt(opt, folder)
        plt.savefig(folder + "/history-info_plane_.pdf")

    return X, Y, model, history, callbacks
