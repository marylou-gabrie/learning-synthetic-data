import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
from lsd.utils_io import *
from lsd.utils_callbacks import load_mi
# import seaborn as sns
import matplotlib

matplotlib.rcParams['font.size'] = 8

def subplot_history(axs, loss_history):

    axs[0].plot(np.arange(len(loss_history["loss"])),
                loss_history["loss"], "-", label="train", c='k')
    sc = axs[0].scatter(np.arange(len(loss_history["val_loss"])),
                        loss_history["val_loss"], c=np.arange(len(loss_history["val_loss"])),
                        edgecolor="none", cmap="magma", s=5, label="generalization")
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

    return sc


def subplot_infoplane(axs, loss_history, mutual_info):
    epochs = np.arange(mutual_info.IXT.shape[0]) * mutual_info.compute_info_every

    for layer in range(mutual_info.HT.shape[1]):
        axs[layer].scatter(mutual_info.IXT[:,layer],
                        -1 * np.array(loss_history["val_loss"])[epochs],
                        c=epochs,
                        edgecolor="none", cmap="magma", s=5)
        axs[layer].set_ylabel("- test error")
        axs[layer].set_xlabel("I(T,T+1)")


def subplot_info_history(axs, mutual_info):
    ''' pass one axis per layer
    '''
    epochs = np.arange(mutual_info.HT.shape[0]) * mutual_info.compute_info_every

    for layer, [ax1, ax2, ax3] in enumerate(axs):
        # ax2 = ax1.twinx()
        ax1.set_title("layer " + str(layer))
        ax1.scatter(epochs,
            mutual_info.HT[:, layer],
            c=epochs,
            edgecolor="none", cmap="magma", s=5)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("HT")
        ax1.autoscale(tight=True)

        ax2.scatter(epochs,
            mutual_info.IXT[:, layer], label="mi",
            c=epochs,
            edgecolor="none", cmap="magma", s=5)
        # ax.set_xlabel("Epochs")
        ax2.set_ylabel("mi")
        ax2.autoscale(tight=True)

        ax3.scatter(epochs,
            mutual_info.V[:, layer], label="mmse",
            c=epochs,
            edgecolor="none", cmap="magma", s=5)
        # ax.set_xlabel("Epochs")
        ax3.set_ylabel("mmse")
        ax3.autoscale(tight=True)


def plot_paper(folder):

    history = load_history(folder)
    mutual_info = load_mi(folder)

    try:
        loss_history = history.history
    except AttributeError:
        loss_history = history

    n_layers = mutual_info.HT.shape[1]

    fig = plt.figure("all info", figsize=(8, 4))
    plt.clf()

    # option 2
    gs = matplotlib.gridspec.GridSpec(int(n_layers), 4 + int(n_layers),
                   width_ratios=[1.] * (3 + int(n_layers)) + [.075]
                   )

    ax_history = [plt.subplot(gs[0, :n_layers])]
    axs_info_history = [[plt.subplot(gs[int(0 + layer), int(0 + n_layers)]),
                        plt.subplot(gs[int(0 + layer), int(1 + n_layers)]),
                        plt.subplot(gs[int(0 + layer), int(2 + n_layers)])] for
                        layer in range(n_layers)]
    axs_infoplane = [plt.subplot(gs[1:, int(0 + layer)]) for layer in
                        range(n_layers)]


    ax_clb = plt.subplot(gs[:, -1])

    sc = subplot_history(ax_history, loss_history)
    subplot_infoplane(axs_infoplane, loss_history, mutual_info)
    subplot_info_history(axs_info_history, mutual_info)

    clb = plt.colorbar(sc, cax=ax_clb)
    clb.set_label("Epochs", labelpad=-20, y=1.05, rotation=0)

    plt.subplots_adjust(left=0.05,  # the left side of the subplots of the figure
                right=.95,    # the right side of the subplots of the figure
                bottom=0.1,   # the bottom of the subplots of the figure
                top=0.9,      # the top of the subplots of the figure
                wspace=0.05,   # the amount of width reserved for space between subplots,
                                # expressed as a fraction of the average axis width
                hspace=0.05,)  # the amount of height reserved for space between subplots,
                                # expressed as a fraction of the average axis height)
    # plt.tight_layout()
    plt.show(block=False)
    return fig
