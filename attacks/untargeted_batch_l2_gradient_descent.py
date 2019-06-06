import torch
from .untargeted_batch_gradient_descent import *


class UntargetedBatchL2GradientDescent(UntargetedBatchGradientDescent):
    """
    Implementation of untargetetd PGD attack.
    """

    def __init__(self, model, images, classes=None, epsilon=0.5, c_1=0.01, c_2=0.5, base_lr=0.01):
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

        super(UntargetedBatchL2GradientDescent, self).__init__(model, images, classes, epsilon, c_1, c_2, base_lr)

    def initialize_random(self):
        """
        Initialize the attack.
        """

        size = self.images.size()
        random = common.numpy.uniform_ball(size[0], numpy.prod(size[1:]), epsilon=self.epsilon, ord=2)
        self.perturbations = torch.from_numpy(random.reshape(size).astype(numpy.float32))
        self.perturbations = torch.autograd.Variable(self.perturbations, requires_grad=True)

    def norm(self):
        """
        Compute the norm to check.

        :return: norm of current perturbation
        :rtype: float
        """

        return torch.norm(self.perturbations.view(self.perturbations.size()[0], -1), 2, 1)

    def norm_loss(self):
        """
        Norm loss.

        :return: loss based on norm/corresponding to norm constraint
        :rtype: torch.autograd.Variable
        """

        return torch.norm(self.perturbations.view(self.perturbations.size()[0], -1), 2, 1)