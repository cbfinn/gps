from gps.algorithm.cost.cost import Cost


class CostIOCWrapper(Cost):
    """ Set up weighted neural network norm loss with learned parameters. """
    def __init__(self, hyperparams):
        super(CostIOCWrapper, self).__init__(hyperparams)
        self.cost = hyperparams['wrapped_cost']  # Ground truth cost
        self.cost = self.cost['type'](self.cost)

    def copy(self):
        return CostIOCWrapper(self._hyperparams)

    def eval(self, sample):
        return self.cost.eval(sample)

    def update(self, demoU, demoX, demoO, d_log_iw, sampleU, sampleX, sampleO, s_log_iw):
        return
