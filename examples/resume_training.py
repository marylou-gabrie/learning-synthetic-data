import numpy.random
import sys

from lsd.set_params import resume_training

numpy.random.seed(276635)

folder = sys.argv[1]
epochs = int(sys.argv[2])

resume_training(folder, epochs, 1, display=1)
