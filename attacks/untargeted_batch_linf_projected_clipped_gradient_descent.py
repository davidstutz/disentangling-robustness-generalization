import torch
from .untargeted_batch_linf_gradient_descent import *
import common.torch


class UntargetedBatchLInfProjectedClippedGradientDescent(UntargetedBatchLInfGradientDescent):
    """
    Implementation of untargetetd PGD attack.
    """

    def __init__(self, model, images, classes=None, epsilon=0.5, c=0.05, base_lr=0.1, max_projections=5):
        """
        Constructor.

        :param model: model to attack
        :type model: torch.nn.Module
        :param images: image(s) to attack
        :type images: torch.autograd.Variable
        :param classes: true classes, if None, they will be deduced to avoid label leaking
        :type classes: torch.autograd.Variable
        :param epsilon: maximum strength of attack
        :type epsilon: float
        :param c: weight of objective relative to weight decay
        :type c: float
        :param base_lr: base learning rate
        :type base_lr: float
        :param max_projections: number of projections for alternating projection
        :type max_projections: int
        """

        super(UntargetedBatchLInfProjectedClippedGradientDescent, self).__init__(model, images, classes, epsilon, 0.0, c, base_lr)

        self.max_projections = max_projections
        """ (int) Maximum number of projections. """

    def set_max_projections(self, max_projections):
        """
        Set max projections
        :param max_projections: number of projections for alternating projection
        :type max_projections: int
        """

        self.max_projections = max_projections

    def norm_loss(self):
        """
        Norm loss.

        :return: loss based on norm/corresponding to norm constraint
        :rtype: torch.autograd.Variable
        """

        zeros = torch.zeros((self.perturbations.size(0)))
        if cuda.is_cuda(self.model):
            zeros = zeros.cuda()
        return zeros

    def bound_loss(self):
        """
        Bound loss.

        :return: loss to constrain [0,1]
        :rtype: torch.autograd.Variable
        """

        zeros = torch.zeros((self.perturbations.size(0)))
        if cuda.is_cuda(self.model):
            zeros = zeros.cuda()
        return zeros

    def project(self):
        """
        Project the perturbation.
        """

        for j in range(self.max_projections):
            # We assume that the auto encoder projection already takes care of clipping the
            # output to a valid range!
            # Also note that the order of projections is relevant!
            if self.auto_encoder is not None:
                self.perturbations.data = common.torch.project(self.perturbations.data, self.epsilon, float('inf'))
                self.perturbations.data = self.project_auto_encoder(self.perturbations.data)
            else:
                if self.max_bound is not None:
                    self.perturbations.data = torch.min(self.max_bound - self.images.data, self.perturbations.data)
                if self.min_bound is not None:
                    self.perturbations.data = torch.max(self.min_bound - self.images.data, self.perturbations.data)
                self.perturbations.data = common.torch.project(self.perturbations.data, self.epsilon, float('inf'))

