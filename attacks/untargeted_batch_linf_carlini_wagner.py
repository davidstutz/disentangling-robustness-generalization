import torch
from .untargeted_batch_linf_reparameterized_gradient_descent import *


class UntargetedBatchLInfCarliniWagner(UntargetedBatchLInfReparameterizedGradientDescent):
    """
    Implementation of untargetetd PGD attack.
    """

    def __init__(self, model, images, classes=None, epsilon=0.5, c=0.3, base_lr=0.1):
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

        super(UntargetedBatchLInfCarliniWagner, self).__init__(model, images, classes, epsilon, c, base_lr)
