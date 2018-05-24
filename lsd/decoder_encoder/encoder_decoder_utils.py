from lsd.model_params import DataModelParams, TraineeModelParams


class Decoder(DataModelParams):

    def generate_data(self, weights, n_samples):
        Y, X = self.generate_data_raw(weights, n_samples)
        Y = Y[:, :int(self.fraction_relevant * self.layer_sizes[0])]
        return X, Y


class Encoder(TraineeModelParams):
    pass
