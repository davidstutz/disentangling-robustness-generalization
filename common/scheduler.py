import torch
from . import utils
from .log import log


class Scheduler:
    """
    Schedule wrapper for optimizers.
    """

    def __init__(self, optimizer, params):
        """
        Constructor.

        :param optimizer: underlying optimizer
        :type optimizer: torch.optim.Optimizer
        :param params: dictionary with parameters
        :type params: dict
        """

        self.optimizer = optimizer
        """ (torch.optim.Optimizer) Underlying optimizer. """

        self.check_params(params)
        self.params = params
        """ (dict) Parameters. """

    def check_params(self, params):
        """
        Check parameters.

        :param params: dictionary with params
        :type params: dict
        """

        for param_group in self.optimizer.param_groups:
            for key in params.keys():
                assert key in param_group.keys()

    def initialize(self):
        """
        Initialize all parameters.
        """

        for param_group in self.optimizer.param_groups:
            for key in self.params.keys():
                param_group[key] = self.params[key][0]

    def update(self, epoch, fraction=0):
        """
        Update all parameters.

        :param epoch: epoch to update parameters for
        :type epoch: int
        :param fraction: optional fraction of the current epoch
        :type fraction: float
        """

        for param_group in self.optimizer.param_groups:
            for key in self.params.keys():
                param_group[key] = max(min(self.params[key][0]*(self.params[key][1]**(epoch + 1 + fraction)), self.params[key][2]), self.params[key][3])
                #log('[Scheduler] updated %s to %g (%g %g %g %g)' % (key, param_group[key], self.params[key][0], self.params[key][1], self.params[key][2], self.params[key][3]))

    def report(self):
        """
        Report current parameters.
        """

        report = ''
        for param_group in self.optimizer.param_groups:
            for key in self.params.keys():
                report += '[%s: %g]' % (key, param_group[key])
            break
        return report


class SGDScheduler(Scheduler):
    """
    SGD scheduler, initializes the SGD optimizer itself.
    """

    def __init__(self, mixed, **kwargs):
        """
        Constructor.

        :param parameters: model parameters or optimizer
        :type parameters: torch.autograd.Variable or torch.optim.Optimizer
        :param params: dictionary with parameters
        :type params: dict
        """

        params = {
            'lr': [0.01, 0.9, 1, 0.000000001],
            'momentum': [0.9, 1, 1, 0],
            'weight_decay': [0.0001, 1, 1, 0]
            # start, decay, max, min
        }

        keys = kwargs.keys()
        if 'lr' in keys:
            params['lr'][0] = utils.to_float(kwargs['lr'])
        if 'lr_decay' in keys:
            params['lr'][1] = utils.to_float(kwargs['lr_decay'])
        if 'lr_min' in keys:
            params['lr'][3] = utils.to_float(kwargs['lr_min'])
        if 'momentum' in keys:
            params['momentum'][0] = utils.to_float(kwargs['momentum'])
        if 'momentum_decay' in keys:
            params['momentum'][1] = utils.to_float(kwargs['momentum_decay'])
        if 'momentum_max' in keys:
            params['momentum'][2] = utils.to_float(kwargs['momentum_max'])
        if 'weight_decay' in keys:
            params['weight_decay'][0] = utils.to_float(kwargs['weight_decay'])

        # need to initialize optimizer here as it is required to set the initial learning rate
        if isinstance(mixed, torch.optim.Optimizer):
            optimizer = mixed
        else:
            optimizer = torch.optim.SGD(mixed, lr=params['lr'][0])

        Scheduler.__init__(self, optimizer, params)


class ADAMScheduler(Scheduler):
    """
    Adam scheduler, initializes the SGD optimizer itself.
    """

    def __init__(self, mixed, **kwargs):
        """
        Constructor.

        :param parameters: model parameters or optimizer
        :type parameters: torch.autograd.Variable or torch.optim.Optimizer
        :param params: dictionary with parameters
        :type params: dict
        """

        params = {
            'lr': [0.01, 0.95, 1, 0.000000001],
            'weight_decay': [0, 1, 1, 0]
        }

        keys = kwargs.keys()
        if 'lr' in keys:
            params['lr'][0] = utils.to_float(kwargs['lr'])
        if 'lr_decay' in keys:
            params['lr'][1] = utils.to_float(kwargs['lr_decay'])
        if 'lr_min' in keys:
            params['lr'][3] = utils.to_float(kwargs['lr_min'])
        if 'weight_decay' in keys:
            params['weight_decay'][0] = utils.to_float(kwargs['weight_decay'])

        # need to initialize optimizer here as it is required to set the initial learning rate
        if isinstance(mixed, torch.optim.Optimizer):
            optimizer = mixed
        else:
            optimizer = torch.optim.Adam(mixed, lr=params['lr'][0], betas=(0.5, 0.9))

        Scheduler.__init__(self, optimizer, params)