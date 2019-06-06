import torch
from .untargeted_batch_linf_gradient_descent import *


class UntargetedBatchLInfClippedGradientDescent(UntargetedBatchLInfGradientDescent):
    """
    Implementation of untargetetd PGD attack.
    """

    # For L_1 norm, the norm should be weighted smaller.
    # This means, weighting the objective and bound higher.
    def __init__(self, model, images, classes=None, epsilon=0.5, c_1=0.01, c_2=0.5, base_lr=0.1):
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
        :param c_1: weight of bound relative to weight decay
        :type c_1: float
        :param c_2: weight of objective relative to weight decay
        :type c_2: float
        :param base_lr: base learning rate
        :type base_lr: float
        """

        super(UntargetedBatchLInfClippedGradientDescent, self).__init__(model, images, classes, epsilon, c_1, c_2, base_lr)

        self.tau = self.epsilon
        """ (float) Used to approximate L_infinity norm as objective. """

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

    def norm_loss(self):
        """
        Norm loss.

        :return: loss based on norm/corresponding to norm constraint
        :rtype: torch.autograd.Variable
        """

        # As described by Carlini and Wagner, the norm is not implemented directly
        # as it is not differentiable.
        # Instead, it is approximate dusing a hinge loss.

        perturbations = self.perturbations
        images = self.images

        # important for the view calls, especially with some resnet architectures
        if not perturbations.is_contiguous():
            perturbations = perturbations.contiguous()
        if not images.is_contiguous():
            images = images.contiguous()

        perturbations = perturbations.view(perturbations.size()[0], -1) - images.view(images.size()[0], -1)

        # Done here because max iterations may change!
        self.tau_decay = math.pow(math.e, (1. / self.max_iterations) * math.log(1. / 10.)) # goal is to be at 1/5*epsilon after max_iterations
        """ (float) Decay of tau value. """

        self.tau *= self.tau_decay

        return torch.sum(torch.max(torch.abs(perturbations) - self.tau, torch.zeros_like(perturbations)), 1)