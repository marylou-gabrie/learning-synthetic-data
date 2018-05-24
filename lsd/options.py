
class ExpOptions(object):
    """
    Storage for all options in experiment
    """
    training = None
    information = None
    datamodel = None
    trainee = None

    def __init__(self, training, information, datamodel, trainee):
        self.training = training
        self.information = information
        self.datamodel = datamodel
        self.trainee = trainee


class Prior(object):
    BERNOULLI = "bernoulli"
    GB = "gb"
    NORMAL = "normal"
