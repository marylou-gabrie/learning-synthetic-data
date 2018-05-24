from lsd.utils_callbacks import MIHistory, MIHistoryExact, MICheckpoint
from lsd.utils_io import save_history, save_mutual_info, save_dataset
from lsd.decoder_encoder.encoder_decoder_utils import Decoder
from lsd.options import Prior
from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


class Training(object):
    def __init__(self, lr=0.01, epochs=0., optimizer=None, n_samples=0,
                 batchsize=0, validation_split=.5, loss='MSE'):
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.n_samples = n_samples
        self.batchsize = batchsize
        self.validation_split = validation_split

    def train(self, X, Y, datamodel, trainee, information, save=0,
              folder="temp", test=False):

        X_train = X[:int(self.validation_split * self.n_samples), :]
        Y_train = Y[:int(self.validation_split * self.n_samples), :]
        X_val = X[int(self.validation_split * self.n_samples):, :]
        Y_val = Y[int(self.validation_split * self.n_samples):, :]

        model = trainee.keras_model()
        metrics_list = [metrics.categorical_accuracy] if self.loss == 'categorical_crossentropy' else []
        model.compile(loss=self.loss, optimizer=self.optimizer(lr=self.lr),
                      metrics=metrics_list)

        # Preparing callbacks
        callbacks = []

        if information.up_to > 0:
            information.match_mi_layers(datamodel, trainee)

        mutual_info = MIHistory(information, folder=folder)
        callbacks.append(mutual_info)

        if test:
            if information.up_to == 0:
                raise ValueError("Set test to False if not computing entropies")
            mutual_info_exact = MIHistoryExact(information, datamodel, trainee)
            callbacks.append(mutual_info_exact)

        # Option to save weights 10 times during training
        if save > 0:
            callbacks.append(MICheckpoint(mutual_info, folder + "/MI"))

        if save > 1:
            period = int(self.epochs / 10)
            callbacks.append(ModelCheckpoint(folder + "/weights.{epoch:02d}.hdf5",
                                             monitor='val_loss', verbose=0,
                                             save_best_only=False,
                                             save_weights_only=False,
                                             mode='auto', period=period))

        history = model.fit(X_train, Y_train,
                            batch_size=self.batchsize, epochs=self.epochs,
                            verbose=1, validation_data=(X_val, Y_val),
                            callbacks=callbacks)

        if save > 0:
            save_history(history, folder + "/loss_history_accuracy")
            save_mutual_info(mutual_info, folder + "/MI", rewrite=True)
            model.save(folder + "/model.h5")
            save_dataset(X, Y, folder)

            if test:
                save_mutual_info(mutual_info_exact, folder + "/MI_exact",
                                rewrite=True, save_all=False)

        return model, history, callbacks

    def resume_train(self, X, Y, initial_epoch, information, save=0, folder="temp"):
        X_train = X[:int(self.validation_split * self.n_samples), :]
        Y_train = Y[:int(self.validation_split * self.n_samples), :]
        X_val = X[int(self.validation_split * self.n_samples):, :]
        Y_val = Y[int(self.validation_split * self.n_samples):, :]

        model = load_model(folder + "/model.h5")

        # Preparing callbacks
        callbacks = []

        mutual_info = MIHistory(information, resume_from=folder, folder=folder)
        callbacks.append(mutual_info)

        # Option to save weights 10 times during training
        if save > 1:
            period = int((self.epochs - initial_epoch) / 10)
            callbacks.append(ModelCheckpoint(folder + "/weights.{epoch:02d}.hdf5",
                                             monitor='val_loss', verbose=0,
                                             save_best_only=False,
                                             save_weights_only=False,
                                             mode='auto', period=period))

        history = model.fit(X_train, Y_train,
                            batch_size=self.batchsize,
                            epochs=self.epochs, initial_epoch=initial_epoch,
                            verbose=1, validation_data=(X_val, Y_val),
                            callbacks=callbacks,
                            )

        if save > 0:
            history_all = save_history(history,
                                       folder + "/loss_history_accuracy",
                                       resume=True)
            model.save(folder + "/model.h5", overwrite=True)
            if information.up_to > 0:
                save_mutual_info(mutual_info, folder + "/MI", rewrite=True)

        return model, history_all, callbacks
