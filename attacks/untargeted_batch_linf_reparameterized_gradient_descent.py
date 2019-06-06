import torch
import math
from .untargeted_batch_reparameterized_gradient_descent import *


class UntargetedBatchLInfReparameterizedGradientDescent(UntargetedBatchReparameterizedGradientDescent):
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

        super(UntargetedBatchLInfReparameterizedGradientDescent, self).__init__(model, images, classes, epsilon, c, base_lr)

        self.EPS = 1e-6
        """ (float) For approximating the L1 norm. """

        self.tau = self.epsilon
        """ (float) Used to approximate L_infinity norm as objective. """

    def initialize_random(self):
        """
        Initialize the attack.
        """

        assert self.min_bound is not None, 'reparameterization only works with valid upper and lower bounds'
        assert self.max_bound is not None, 'reparameterization only works with valid upper and lower bounds'

        size = self.images.size()
        min_bound = self.min_bound.cpu().numpy()
        max_bound = self.max_bound.cpu().numpy()

        random = common.numpy.uniform_ball(size[0], numpy.prod(size[1:]), epsilon=self.epsilon, ord=float('inf'))
        random = self.images.data.cpu().numpy() + random.reshape(size)
        random = numpy.minimum(max_bound, random)
        random = numpy.maximum(min_bound, random)
        random = random.astype(numpy.float32)

        self.w = (2 - 2*self.EPS) * (random - min_bound) / (max_bound - min_bound) - 1 + self.EPS
        self.w = numpy.arctanh(self.w)

        self.w = torch.from_numpy(self.w)
        self.w = torch.autograd.Variable(self.w, requires_grad=True)

        if cuda.is_cuda(self.model):
            self.w = torch.autograd.Variable(self.w.cuda(), requires_grad=True)
        else:
            self.w = torch.autograd.Variable(self.w, requires_grad=True)

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
