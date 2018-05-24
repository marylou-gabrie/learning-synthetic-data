import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from decimal import Decimal
from lsd.decoder_encoder.encoder_decoder_utils import Decoder
from lsd.options import Prior
from lsd.teacher_student.teacher_student_utils import Teacher
from lsd.utils_callbacks import load_mi
from lsd.utils_io import load_history #, load_mi
from graphviz import Digraph

def plot_loss_history(history):
    loss_history = history.history
    plt.figure("history", figsize=(12, 5))
    plt.clf()
    ax = plt.subplot(121)
    ax.plot(np.arange(len(loss_history["loss"])), loss_history["loss"],"o-", label="train")
    ax.plot(np.arange(len(loss_history["val_loss"])), loss_history["val_loss"], "o-", label="test")
    ax.legend(numpoints=1)
    ax.set_xlabel("epochs")
    ax.set_ylabel("error")

    ax = plt.subplot(122)
    ax.plot(np.array(loss_history["loss"]) * -1, loss_history["loss"], "-", label="train")
    ax.scatter(np.array(loss_history["loss"]) * -1, loss_history["val_loss"],
               c=np.arange(len(loss_history["val_loss"])),
                    edgecolor="none", cmap="magma", s=5)
    ax.set_xlabel("- train error")
    ax.set_ylabel("error")

    plt.tight_layout()


def plot_infoPlane(history, mutual_info):
    loss_history = history.history
    fig = plt.figure("info plane",figsize=(12,5))
    plt.clf()
    ax = plt.subplot(121)
    sc = ax.scatter(mutual_info.IXT,
                    -1*np.array(loss_history["val_loss"]),
                    c=np.arange(len(loss_history["val_loss"])),
                    edgecolor="none", cmap="magma", s=5)
    ax.set_ylabel("- test error")
    ax.set_xlabel("I(X,T_1)")
    clb = fig.colorbar(sc)
    clb.set_label("Epochs")

    ax = plt.subplot(122)
    sc = ax.scatter(np.arange(len(loss_history["val_loss"])),
                    mutual_info.IXT,
                    c=np.arange(len(loss_history["val_loss"])),
                    edgecolor="none", cmap="magma", s=5)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("I(X,T_1)")
    plt.tight_layout()


def prior_to_str(prior):
    if prior[0]==Prior.NORMAL:
        strg = 'normal({:0.2f},{:0.2f})'.format(prior[1],prior[2])
    elif prior[0]==Prior.BERNOULLI:
        strg = 'bernoulli({:0.2f})'.format(prior[1])
    elif prior[0]==Prior.GB:
        strg = 'gb(rho={:0.2f},{:0.2f},{:0.2f})'.format(prior[3],prior[1],prior[2])
    return strg


def plot_setting_teacher_student(opt, folder):
    nn = Digraph(comment='experimental setting', format='png')
    nn.node('X', 'X \n N = {:d} \n prior: {:s} \n {:0.1f} relevant'.format(opt.trainee.n_features,
                                                                        prior_to_str(opt.datamodel.prior),
                                                                        opt.datamodel.fraction_relevant))
    previous = 'X'
    for layer in range(opt.datamodel.n_layers - 1):
        nn.node('H{:d}'.format(layer + 1), 'H{:d} \n N ={:d} \n {:s} \n noise = {:0.1E} '.format(layer + 1,
                                                                    opt.datamodel.layer_sizes[layer + 1],
                                                                    opt.datamodel.activations[layer],
                                                                    Decimal(opt.datamodel.layer_noises[layer])))
        nn.edge(previous, 'H{:d}'.format(layer + 1))
        previous = 'H{:d}'.format(layer + 1)
    nn.node('Y', 'Y \n N = {:d} \n {:s} \n noise = {:0.2f} '.format(opt.datamodel.layer_sizes[-1],
                                            opt.datamodel.activations[-1],
                                            opt.datamodel.layer_noises[-1]))
    nn.edge(previous, 'Y')

    previous = 'X'
    for layer in range(opt.trainee.n_layers - 1):
        nn.node('T{:d}'.format(layer + 1), 'T{:d} \n N ={:d} \n {:s} \n noise = {:0.1E} '.format(layer + 1,
                                                                    opt.trainee.layer_sizes[layer + 1],
                                                                    opt.trainee.activations[layer],
                                                                    Decimal(opt.trainee.layer_noises[layer])))
        nn.edge(previous, 'T{:d}'.format(layer + 1), label='{:s}'.format('\t free' if opt.trainee.layer_free[layer] == 1 else '\t cdc' if opt.trainee.layer_free[layer] == 2 else ''))
        previous = 'T{:d}'.format(layer + 1)
    nn.node('hY', 'Yh \n N = {:d} \n {:s} \n noise = {:0.2f} '.format(opt.trainee.layer_sizes[-1],
                                            opt.trainee.activations[-1],
                                            opt.trainee.layer_noises[-1]))
    nn.edge(previous, 'hY', label='{:s}'.format('\t free' if opt.trainee.layer_free[-1] == 1 else ''))

    nn.attr('node', shape='plaintext')
    try:  # new information with all parameters
        nn.node('TRNG', 'training params: \n \n'
                        '{:s} \n lr = {:0.1E} \n'
                        'coef init weights {:0.1E} \n'
                        'gamma = {:0.1f} \n batch = {:d} \n'
                        'loss = {:s}\n \n'
                        'entropy params: \n \n'
                        'mi noise = {:0.1E} \n'
                        'inter mi noise = {:0.1E} \n'
                        'use_vamp = {:s} \n'
                        'max_iter = {:d} \n'
                        'tol = {:0.1E}'.format(opt.training.optimizer.__name__,
                                Decimal(opt.training.lr),
                                Decimal(opt.trainee.coef_weights_init),
                                .5 * opt.training.n_samples / opt.trainee.n_features,
                                opt.training.batchsize,
                                opt.training.loss,
                                opt.information.mi_noise,
                                opt.information.inter_mi_noise,
                                str(opt.information.use_vamp),
                                opt.information.max_iter,
                                opt.information.tol))
    except:  # maintining compuatibility with older ones
        if opt.information.mi_noise is None:
            nn.node('TRNG', 'training params: \n \n'
                        '{:s} \n lr = {:0.1E} \n'
                        'coef init weights {:0.1E} \n'
                        'gamma = {:0.1f} \n batch = {:d} \n'
                        'loss = {:s}\n \n'.format(opt.training.optimizer.__name__,
                                Decimal(opt.training.lr),
                                Decimal(opt.trainee.coef_weights_init),
                                .5 * opt.training.n_samples / opt.trainee.n_features,
                                opt.training.batchsize,
                                opt.training.loss,))
        else:
            nn.node('TRNG', 'training params: \n \n'
                            '{:s} \n lr = {:0.1E} \n'
                            'coef init weights {:0.1E} \n'
                            'gamma = {:0.1f} \n batch = {:d} \n'
                            'loss = {:s}\n \n'
                            'entropy params: \n \n'
                            'mi noise = {:0.1E} \n'
                            'inter mi noise = {:0.1E}'.format(opt.training.optimizer.__name__,
                                    Decimal(opt.training.lr),
                                    Decimal(opt.trainee.coef_weights_init),
                                    .5 * opt.training.n_samples / opt.trainee.n_features,
                                    opt.training.batchsize,
                                    opt.training.loss,
                                    opt.information.mi_noise,
                                    opt.information.inter_mi_noise))
    nn.attr('edge', color='white')
    nn.edge('Y', 'TRNG')

    nn.render(folder + "/Network_param.gv")


def plot_setting_encoder_decoder(opt, folder):
    nn = Digraph(comment='experimental setting', format='png')
    nn.node('X', 'X \n N = {:d} \n {:s} \n noise = {:0.2f}'.format(opt.trainee.n_features,
                                                                        opt.datamodel.activations[-1],
                                                                        opt.datamodel.layer_noises[-1]))
    previous = 'X'
    for layer in range(opt.datamodel.n_layers - 1)[::-1]:
        print("plotting layer ", layer, "\n")
        nn.node('H{:d}'.format(layer + 1), 'H{:d} \n N ={:d} \n {:s} \n noise = {:0.1E} '.format(layer+1,
                                                                    opt.datamodel.layer_sizes[layer+1],
                                                                    opt.datamodel.activations[layer],
                                                                    Decimal(opt.datamodel.layer_noises[layer])))
        nn.edge('H{:d}'.format(layer + 1), previous)
        previous = 'H{:d}'.format(layer + 1)
    nn.node('Y', 'Y \n N = {:d} \n prior: {:s} \n {:0.1f} relevant'.format(opt.datamodel.layer_sizes[0],
                                                                    prior_to_str(opt.datamodel.prior),
                                                                    opt.datamodel.fraction_relevant))
    nn.edge('Y', previous)

    previous = 'X'
    for layer in range(opt.trainee.n_layers - 1):
        nn.node('T{:d}'.format(layer + 1), 'T{:d} \n N ={:d} \n {:s} \n noise = {:0.1E}'.format(layer + 1,
                                                                    opt.trainee.layer_sizes[layer + 1],
                                                                    opt.trainee.activations[layer],
                                                                    Decimal(opt.trainee.layer_noises[layer])))
        nn.edge(previous, 'T{:d}'.format(layer + 1), label='{:s}'.format('\t free' if opt.trainee.layer_free[layer] == 1 else '\t cdc' if opt.trainee.layer_free[layer] == 2 else ''))
        previous = 'T{:d}'.format(layer + 1)
    nn.node('hY', 'Yh \n N = {:d} \n {:s} \n noise = {:0.1E} '.format(opt.trainee.layer_sizes[-1],
                                            opt.trainee.activations[-1],
                                            Decimal(opt.trainee.layer_noises[-1])))
    nn.edge(previous, 'hY', label='{:s}'.format('\t free' if opt.trainee.layer_free[-1] == 1 else ''))

    nn.attr('node', shape='plaintext')
    try:  # new information with all parameters
        nn.node('TRNG', 'training params: \n \n'
                        '{:s} \n lr = {:0.1E} \n'
                        'coef init weights {:0.1E} \n'
                        'gamma = {:0.1f} \n batch = {:d} \n'
                        'loss = {:s}\n \n'
                        'entropy params: \n \n'
                        'mi noise = {:0.1E} \n'
                        'inter mi noise = {:0.1E} \n'
                        'use_vamp = {:s} \n'
                        'max_iter = {:d} \n'
                        'tol = {:0.1E}'.format(opt.training.optimizer.__name__,
                                Decimal(opt.training.lr),
                                Decimal(opt.trainee.coef_weights_init),
                                .5 * opt.training.n_samples / opt.trainee.n_features,
                                opt.training.batchsize,
                                opt.training.loss,
                                opt.information.mi_noise,
                                opt.information.inter_mi_noise,
                                str(opt.information.use_vamp),
                                opt.information.max_iter,
                                opt.information.tol))
    except:  # maintining compuatibility with older ones
        if opt.information.mi_noise is None:
            nn.node('TRNG', 'training params: \n \n'
                        '{:s} \n lr = {:0.1E} \n'
                        'coef init weights {:0.1E} \n'
                        'gamma = {:0.1f} \n batch = {:d} \n'
                        'loss = {:s}\n \n'.format(opt.training.optimizer.__name__,
                                Decimal(opt.training.lr),
                                Decimal(opt.trainee.coef_weights_init),
                                .5 * opt.training.n_samples / opt.trainee.n_features,
                                opt.training.batchsize,
                                opt.training.loss,))
        else:
            nn.node('TRNG', 'training params: \n \n'
                            '{:s} \n lr = {:0.1E} \n'
                            'coef init weights {:0.1E} \n'
                            'gamma = {:0.1f} \n batch = {:d} \n'
                            'loss = {:s}\n \n'
                            'entropy params: \n \n'
                            'mi noise = {:0.1E} \n'
                            'inter mi noise = {:0.1E}'.format(opt.training.optimizer.__name__,
                                    Decimal(opt.training.lr),
                                    Decimal(opt.trainee.coef_weights_init),
                                    .5 * opt.training.n_samples / opt.trainee.n_features,
                                    opt.training.batchsize,
                                    opt.training.loss,
                                    opt.information.mi_noise,
                                    opt.information.inter_mi_noise))
    nn.attr('edge', color='white')
    nn.edge('hY', 'TRNG')

    nn.render(folder + "/Network_param.gv")


def plot_setting(opt, folder):
    if isinstance(opt.datamodel, Teacher):
        plot_setting_teacher_student(opt, folder)
    elif isinstance(opt.datamodel, Decoder):
        plot_setting_encoder_decoder(opt, folder)


def plot_eigvals(mutual_info):
    eigvals = np.array(mutual_info.Eigvals)
    if eigvals == []:
        raise ValueError("No stored eigenvalues")

    plt.figure("spectral density")
    cmap = plt.cm.get_cmap('magma')
    norm = matplotlib.colors.Normalize(vmin=1., vmax=eigvals.shape[0])
    for epoch in range(eigvals.shape[0]):
        sns.kdeplot(eigvals[epoch,:],c=cmap(norm(epoch)))
    plt.show(block=False)


# Subplot functions taking axes as argument to build summary plot
 
def subplot_history(axs, loss_history):

    axs[0].plot(np.arange(len(loss_history["loss"])),
                loss_history["loss"], "-", label="train", c='k')
    sc = axs[0].scatter(np.arange(len(loss_history["val_loss"])),
                        loss_history["val_loss"], c=np.arange(len(loss_history["val_loss"])),
                        edgecolor="none", cmap="magma", s=5, label="test")
    axs[0].legend(numpoints=1)
    axs[0].set_xlabel("epochs")
    axs[0].set_ylabel("error")

    if 'categorical_accuracy' in loss_history.keys():
        ax2 = axs[0].twinx()
        ax2.plot(np.arange(len(loss_history["loss"])),
                loss_history["categorical_accuracy"], "-", label="train", c='grey', alpha=0.7)
        ax2.scatter(np.arange(len(loss_history["val_categorical_accuracy"])),
                        loss_history["val_categorical_accuracy"], c=np.arange(len(loss_history["val_categorical_accuracy"])),
                        edgecolor="grey", cmap="magma", s=5, label="test", alpha=0.9)
        ax2.set_ylabel("accuracy", color="grey")
        ax2.tick_params(axis='y', labelcolor="grey")
        ax2.set_xlabel("epochs")


    axs[1].plot(np.array(loss_history["loss"]) * -1, loss_history["loss"], "-",
                label="train", c='k')
    axs[1].scatter(np.array(loss_history["loss"]) * -1, loss_history["val_loss"],
                    c=np.arange(len(loss_history["val_loss"])),
                    edgecolor="none", cmap="magma", s=5)
    axs[1].set_xlabel("- train error")
    axs[1].set_ylabel("error")
    axs[1].legend()

    return sc


def subplot_infoplane(axs, loss_history, mutual_info):
    epochs = np.arange(mutual_info.IXT.shape[0]) * mutual_info.compute_info_every

    for layer in range(mutual_info.HT.shape[1]):
        axs[0].scatter(mutual_info.IXT[:,layer],
                        -1 * np.array(loss_history["val_loss"])[epochs],
                        c=epochs,
                        edgecolor="none", cmap="magma", s=5)
        axs[0].set_ylabel("- test error")
        axs[0].set_xlabel("I(T,T+1)")

        axs[1].scatter(mutual_info.HT[:, layer],
                        -1 * np.array(loss_history["val_loss"])[epochs],
                        c=epochs,
                        edgecolor="none", cmap="magma", s=5)
        axs[1].set_ylabel("- test error")
        axs[1].set_xlabel("H(T)")


def subplot_info_history(axs, mutual_info):
    ''' pass one axis per layer
    '''
    epochs = np.arange(mutual_info.HT.shape[0]) * mutual_info.compute_info_every

    for layer, [ax1, ax2] in enumerate(axs):
        ax1.set_title("layer "+str(layer))
        ax1.scatter(epochs,
            mutual_info.HT[:, layer],
            c=epochs,
            edgecolor="none", cmap="magma", s=5)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("HT")
        ax1.autoscale(tight=True)

        ax2.scatter(epochs,
            mutual_info.V[:, layer], alpha=.5, label="mmse",
            c=epochs,
            edgecolor="none", cmap="magma", s=5)
        # ax.set_xlabel("Epochs")
        ax2.set_ylabel("mmse")
        ax2.autoscale(tight=True)


def subplot_eigvals(axs, mutual_info):
    ''' pass one axis per layer
    '''
    for layer, [ax1, ax2] in enumerate(axs):
        ax1.set_title("layer "+str(layer))
        eigvals = np.array(mutual_info.Eigvals[layer])
        if eigvals == []:
            raise ValueError("No stored eigenvalues")
        cmap = plt.cm.get_cmap('magma')
        norm = matplotlib.colors.Normalize(vmin=1., vmax=eigvals.shape[0])
        plt.gca()
        for epoch in range(eigvals.shape[0]):
            sns.kdeplot(eigvals[epoch, :], clip=(0.,np.max(eigvals[epoch,:])),  #bw=.1,
                        ax=ax1, c=cmap(norm(epoch)))
        ax1.set_xlabel("lambda")
        ax1.set_ylabel("spectral density")

        ax2.plot(np.arange(eigvals.shape[0]), np.max(eigvals, axis=1), label="max", c="k")
        ax2.set_ylabel("lambda max")
        ax3 = ax2.twinx()
        ax3.plot(np.arange(eigvals.shape[0]), np.mean(eigvals, axis=1), label="mean", c="grey")
        ax3.set_ylabel("lambda mean", color="grey")
        ax3.tick_params(axis='y', labelcolor="grey")
        ax2.set_xlabel("epochs")


def plot_All_multilayer(history, mutual_info, folder=None):
    ''' Will plot also the network if given a folder
    '''
    try:
        loss_history = history.history
    except AttributeError:
        loss_history = history

    n_layers = max(mutual_info.HT.shape[1], 1)  # handling case where nothing stored

    if folder:
        fig = plt.figure("all info", figsize=(18, 7.5))
        plt.clf()
        plt.suptitle(folder)
        gs = matplotlib.gridspec.GridSpec(int(2 * n_layers), 6,
                       width_ratios=[1.35, 1.1, 1.1, 1, 1, .075],
                       )

        ax_net = plt.subplot(gs[:, 0])
        axs_history = [plt.subplot(gs[:n_layers, 1]),
                        plt.subplot(gs[:n_layers, 2])]
        axs_info_history = [[plt.subplot(gs[int(0 + layer), 3]),
                            plt.subplot(gs[int(0 + layer), 4])] for
                            layer in range(n_layers)]
        axs_infoplane = [plt.subplot(gs[n_layers:, 1]),
                        plt.subplot(gs[n_layers:, 2])]
        axs_evs = [[plt.subplot(gs[int(n_layers + layer), 3]),
                    plt.subplot(gs[int(n_layers + layer), 4])] for
                            layer in range(n_layers)]

        ax_net.imshow(plt.imread(folder + "/Network_param.gv.png"), interpolation="spline36")
        ax_net.axis("off")

        ax_clb = plt.subplot(gs[:, 5])

    else:
        fig = plt.figure("all info", figsize=(15.8, 7.5))
        plt.clf()
        plt.suptitle(folder)
        gs = matplotlib.gridspec.GridSpec(int(2 * n_layers), 5,
                       width_ratios=[1.2, 1.1, 1, 1, .075],
                       )

        axs_history = [plt.subplot(gs[:n_layers, 0]),
                        plt.subplot(gs[:n_layers, 1])]
        axs_info_history = [[plt.subplot(gs[int(0 + layer), 2]),
                            plt.subplot(gs[int(0 + layer), 3])] for
                            layer in range(n_layers)]
        axs_infoplane = [plt.subplot(gs[n_layers:, 0]),
                        plt.subplot(gs[n_layers:, 1])]
        axs_evs = [[plt.subplot(gs[int(n_layers + layer), 2]),
                    plt.subplot(gs[int(n_layers + layer), 3])] for
                            layer in range(n_layers)]

        ax_clb = plt.subplot(gs[:, 4])

    sc = subplot_history(axs_history, loss_history)

    if mutual_info.HT.shape[1] > 0:
        subplot_info_history(axs_info_history, mutual_info)
        subplot_infoplane(axs_infoplane, loss_history, mutual_info)

    if len(mutual_info.Eigvals) > 0:
        subplot_eigvals(axs_evs, mutual_info)

    clb = plt.colorbar(sc, cax=ax_clb)
    clb.set_label("Epochs")
    plt.tight_layout()
    plt.show(block=False)
    return fig


def replot_All_multilayer(folder):
    history = load_history(folder)
    mutual_info = load_mi(folder)
    return plot_All_multilayer(history, mutual_info, folder)


def plot_info_history_exact(mutual_info, mutual_info_exact, folder=None):
    n_layers = mutual_info.HT.shape[1]
    epochs = np.arange(mutual_info.HT.shape[0]) * mutual_info.compute_info_every

    fig = plt.figure("test against exact", figsize=(10, 5))
    plt.clf()
    plt.suptitle(folder)
    gs = matplotlib.gridspec.GridSpec(1, n_layers)

    for layer in range(n_layers):
        ax1 = plt.subplot(gs[0, layer])
        ax1.set_title("layer " + str(layer))
        ax1.scatter(epochs,
            mutual_info.HT[:, layer],
            c=epochs,
            edgecolor="none", cmap="magma", s=5)
        ax1.plot(epochs, mutual_info_exact.HT[:, layer], c='k', label='exact')
        ax1.legend()
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("HT")
        ax1.autoscale(tight=True)

    return fig
