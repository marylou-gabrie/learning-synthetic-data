import h5py
import keras
import numpy as np
import os
import pickle
import time
# from lsd.utils_callbacks import MIHistory


def fold_gen(exp_type, exp_description, date=None, rewrite=True, prefix="",
             postfix=""):
        if date is None:
                date = time.strftime('%d-%m-%Y')

        folder = prefix + "data/" + postfix + date + "_" + exp_type + "_" 
        folder += exp_description

        if os.path.isdir(folder) is False:
                os.mkdir(folder)
        else:
                if rewrite is False:
                        count = 1
                        while os.path.isdir(folder + '_{:d}'.format(count)):
                                count += 1
                        os.mkdir(folder + "_{:d}".format(count))
                        folder = folder + "_{:d}".format(count)
        return folder


def save_dataset(X, Y, folder):
    f = h5py.File(folder + '/dataset.hdf5', 'w')
    f.create_dataset('X', data=X)
    f.create_dataset('Y', data=Y)
    f.close()


def load_dataset(folder):
    f = h5py.File(folder + '/dataset.hdf5', 'r')
    X = np.array(f["X"])
    Y = np.array(f["Y"])
    f.close()
    return X, Y


def save_opt(opt, folder):
    pickle.dump(opt, open(folder + "/opt.pickle", "wb"))


def load_opt(folder):
    return pickle.load(open(folder + "/opt.pickle", "rb"))


def load_decoder_weights(folder):
    opt = pickle.load(open(folder + "/opt.pickle", "rb"))
    return opt.decoder.weights


def load_teacher_weights(folder):
    opt = pickle.load(open(folder + "/opt.pickle", "rb"))
    return opt.teacher.weights


def save_mutual_info(mutual_info, filename, rewrite=True, save_all=True):
    if not rewrite and os.path.isfile(filename + '.hdf5'):
        count = 1
        while os.path.isfile(filename + '_{:d}.hdf5'.format(count)):
            count += 1

        filename += "_{:d}".format(count)

    f = h5py.File(filename + '.hdf5', 'w')
    f.create_dataset('HT', data=mutual_info.HT)
    if save_all:
        f.create_dataset('IXT', data=mutual_info.IXT)
        f.create_dataset('V', data=mutual_info.V)
        for layer in range(len(mutual_info.Eigvals)):
            f.create_dataset('Eigvals/layer' + str(layer),
                                data=np.array(mutual_info.Eigvals[layer]))
    f.close()

    if save_all:
        pickle.dump(mutual_info.start_at, open(filename + "_start_at.pickle", "wb"))


def save_recompute_mutual_info(folder, mutual_info_, rewrite=False,
                              recompute_mi_noise=None,
                              recompute_inter_mi_noise=None):
    filename = folder + "/MI"
    if recompute_mi_noise is not None:
        filename += "_minoise{:0.1E}".format(recompute_mi_noise)
    if recompute_inter_mi_noise is not None:
        filename += "_internoise{:0.1E}".format(recompute_inter_mi_noise)

    filename_count = filename
    if not rewrite and os.path.isfile(filename + '.hdf5'):
        count = 1
        while os.path.isfile(filename_count + '_{:d}.hdf5'.format(count)):
            count += 1
        filename_count += "_{:d}".format(count)
        os.rename(filename + ".hdf5", filename_count + ".hdf5")

    save_mutual_info(mutual_info_, filename)


def save_history(history, filename, resume=False):
    if resume:
        print(filename)
        history_all = pickle.load(open(filename + ".pickle", "rb"))
        for key in history_all.keys():
            history_all[key] += history.history[key]
        pickle.dump(history_all, open(filename + ".pickle", "wb"))
        return history_all
    else:
        pickle.dump(history.history, open(filename + ".pickle", "wb"))


def load_history(folder):
    try:
        return pickle.load(open(folder + "/loss_history_accuracy.pickle", "rb"))
    except FileNotFoundError:
        return load_history_hdf5(folder)


def load_history_hdf5(folder):
    history = keras.callbacks.History()
    history.on_train_begin()
    f = h5py.File(folder + "/loss_history_accuracy.hdf5", 'r')
    history.history["loss"] = list(f["loss"])
    history.history["val_loss"] = list(f["val_loss"])
    f.close()
    return history
