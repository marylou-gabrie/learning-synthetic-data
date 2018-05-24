import numpy as np
from lsd.model_params import DataModelParams, TraineeModelParams


class Teacher(DataModelParams):

    def generate_data(self, weights, n_samples):
        rho = self.fraction_relevant
        if rho < 1.:
            weights[0][int(rho * self.layer_sizes[0]):, :] = np.zeros(
                (int((1 - rho) * self.layer_sizes[0]), self.layer_sizes[1])
            )

        return self.generate_data_raw(weights, n_samples)


class Student(TraineeModelParams):
    pass
