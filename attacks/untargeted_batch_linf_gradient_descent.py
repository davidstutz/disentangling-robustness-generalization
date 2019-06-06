import torch
from .untargeted_batch_gradient_descent import *


class UntargetedBatchLInfGradientDescent(UntargetedBatchGradientDescent):
    """
    Implementation of untargetetd PGD attack.
    """

    def __init__(self, model, images, classes=None, epsilon=0.5, c_1=0.001, c_2=0.05, base_lr=0.1):
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

        super(UntargetedBatchLInfGradientDescent, self).__init__(model, images, classes, epsilon, c_1, c_2, base_lr)

        self.tau = 0.5
        """ (float) Used to approximate L_infinity norm as objective. """

    def initialize_random(self):
        """
        Initialize the attack.
        """

        size = self.images.size()
        random = common.numpy.uniform_ball(size[0], numpy.prod(size[1:]), epsilon=self.epsilon, ord=float('inf'))
        self.perturbations = torch.from_numpy(random.reshape(size).astype(numpy.float32))
        self.perturbations = torch.autograd.Variable(self.perturbations, requires_grad=True)

    def norm(self):
        """
        Compute the norm to check.

        :return: norm of current perturbation
        :rtype: float
        """

        return torch.max(torch.abs(self.perturbations.view(self.perturbations.size()[0], -1)), 1)[0]

    def norm_loss(self):
        """
        Norm loss.

        :return: loss based on norm/corresponding to norm constraint
        :rtype: torch.autograd.Variable
        """

        # As described by Carlini and Wagner, the norm is not implemented directly
        # as it is not differentiable.
        # Instead, it is approximate dusing a hinge loss.
        max = torch.max(torch.abs(self.perturbations))
        if max.data <= self.tau:
            self.tau *= 0.9

        return torch.max(torch.abs(self.perturbations) - self.tau, torch.zeros_like(self.perturbations))[0]