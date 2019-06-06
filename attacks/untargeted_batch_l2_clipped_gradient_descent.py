import torch
from .untargeted_batch_l2_gradient_descent import *


class UntargetedBatchL2ClippedGradientDescent(UntargetedBatchL2GradientDescent):
    """
    Implementation of untargetetd PGD attack.
    """

    def __init__(self, model, images, classes=None, epsilon=0.5, c=0.05, base_lr=0.1):
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
        """

        super(UntargetedBatchL2ClippedGradientDescent, self).__init__(model, images, classes, epsilon, 0.0, c, base_lr)

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
        Clip perturbations.
        """

        # We assume that the auto encoder projection already takes care of clipping the
        # output to a valid range!
        if self.auto_encoder is not None:
            self.perturbations.data = self.project_auto_encoder(self.perturbations.data)
        else:
            if self.max_bound is not None:
                self.perturbations.data = torch.min(self.max_bound - self.images.data, self.perturbations.data)
            if self.min_bound is not None:
                self.perturbations.data = torch.max(self.min_bound - self.images.data, self.perturbations.data)
