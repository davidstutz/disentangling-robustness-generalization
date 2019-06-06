import torch
import numpy
from .untargeted_attack import *
from .untargeted_objectives import *
from common import cuda
from common.log import log
import common.torch


class UntargetedBatchGradientDescent(UntargetedAttack):
    """
    Implementation of untargetetd PGD attack.
    """

    def __init__(self, model, images, classes=None, epsilon=0.5, c_1=0.001, c_2=1, base_lr=0.1):
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

        super(UntargetedBatchGradientDescent, self).__init__(model, images, classes)

        self.perturbations = None
        """ (torch.autograd.Variable) Perturbation of attack. """

        self.output_probabilities = None
        """ (torch.autograd.Variable) Output probabilities of perturbations. """

        self.epsilon = epsilon
        """ (float) Strength of attack. """

        self.max_iterations = 1000
        """ (int) Maximum number of iterations. """

        self.optimizer = None
        """ (torch.optim.Optimizer) Optimizer for attack. """

        self.c_0 = 1
        """ (float) Weight of norm. """

        self.c_1 = c_1
        """ (float) Weight of bound. """

        self.c_2 = c_2
        """ (float) Weight of objective. """

        self.base_lr = base_lr
        """ (float) Base learning rate. """

        self.lr_decay = 1
        """ (float) Learning rate decay. """

        self.skip = 5
        """ (int) Verbosity skip. """

        assert self.max_iterations > 0
        assert self.base_lr > 0
        assert self.lr_decay > 0
        assert self.lr_decay <= 1

    def set_epsilon(self, epsilon):
        """
        Set epsilon.

        :param epsilon: maximum strength of attack
        :type epsilon: float
        """

        self.epsilon = epsilon

    def set_c_0(self, c_0):
        """
        Set c_0 (weight of norm loss).

        :param c_0: weight of norm loss
        :type c_0: float
        """

        self.c_0 = c_0

    def set_c_1(self, c_1):
        """
        Set c_1 (weight of bound loss).

        :param c_1: weight of bound relative to weight decay
        :type c_1: float
        """

        self.c_1 = c_1

    def set_c_2(self, c_2):
        """
        Set c_2 (weight of objective loss).

        :param c_2: weight of objective relative to weight decay
        :type c_2: float
        """

        self.c_2 = c_2

    def set_base_lr(self, base_lr):
        """
        Set base learning rate.

        :param base_lr: base learning rate
        :type base_lr: float
        """

        self.base_lr = base_lr

    def set_max_iterations(self, max_iterations):
        """
        Set max iterations.

        :param max_iterations: number of iterations
        :type max_iterations: int
        """

        self.max_iterations = max_iterations

    def initialize(self):
        """
        Initialize the attack.
        """

        self.initialize_zero()

    def initialize_random(self):
        """
        Initialize the attack.
        """

        raise NotImplementedError('initialize_random should be implemente dby child classes')

    def initialize_zero(self):
        """
        Initialize the attack.
        """

        random = numpy.zeros(self.images.size())
        self.perturbations = torch.from_numpy(random.astype(numpy.float32))
        self.perturbations = torch.autograd.Variable(self.perturbations, requires_grad=True)

    def initialize_optimizer(self):
        """
        Initalize optimizer and parameters to optimize.
        """

        if cuda.is_cuda(self.model):
            self.perturbations = self.perturbations.cuda()

        self.perturbations = torch.nn.Parameter(self.perturbations.data)
        self.optimizer = torch.optim.Adam([self.perturbations], lr=self.base_lr)

    def run(self, untargeted_objective, verbose=True):
        """
        Run the attack.

        :param untargeted_objective: untargeted objective
        :type untargeted_objective: UntargetedObjective
        :param verbose: output progress
        :type verbose: bool
        :return: success, perturbations, probabilities, norms, iteration
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, int
        """

        assert self.perturbations is not None, 'attack was not initialized properly'
        assert isinstance(untargeted_objective, UntargetedObjective), 'expected an objective of type UntargetedObjective, got %s instead' % untargeted_objective
        self.initialize_optimizer()

        # Will hold the individually best results (however, optimization
        # is run for all samples until all are successful or the maximum number
        # of iterations is reached.
        success = numpy.ones((self.perturbations.size()[0]), dtype=numpy.int32)*-1
        success_error = numpy.zeros((self.perturbations.size()[0]), dtype=numpy.float32)
        success_perturbations = numpy.zeros(self.perturbations.size(), dtype=numpy.float32)
        success_probabilities = numpy.zeros((self.perturbations.size()[0], self.logits.size()[1]), dtype=numpy.float32)
        success_norms = numpy.zeros((self.perturbations.size()[0]), dtype=numpy.float32)

        i = 0
        gradient = 0

        for i in range(self.max_iterations + 1):
            # MAIN LOOP OF ATTACK
            # ORDER IMPORTANT

            self.optimizer.zero_grad()

            # 0/
            # Projections if necessary.
            self.project()

            # 1/
            # Reset gradients and compute the logits for the current perturbations.
            # This is not applicable to batches, so check that the logits have been computed for one sample
            # only.
            output_logits = self.model.forward(self.images + self.perturbations)

            # 2/
            # Compute current probabilities and current class.
            # This is mainly used as return value and to check the break criterion, i.e.
            # if the class actually changes.
            output_probabilities = torch.nn.functional.softmax(output_logits, 1)
            _, other_classes = torch.max(output_probabilities, 1)

            # 3/
            # Compute the error components:
            # - norm of the perturbation to enforce the epsilon-ball constraint
            # - bound constraints to enforce image + perturbation in [0,1]
            # - the actual objective given the untargeted objective to use
            # The latter two components are weighted relative to the norm.
            # The weighting together with the used learning rate is actually quite important!
            norm = self.norm_loss() # Will be sum of norms!
            bound = self.bound_loss() + self.encoder_bound_loss()
            objective = untargeted_objective.f(output_logits, self.logits, self.classes)

            # Put together the error.
            error = self.c_0*norm + self.c_1*bound + self.c_2*objective

            # 4/
            # Logging and break condition.
            check_norm = self.norm() # Will be a vector of individual norms.
            batch_size = self.images.size()[0]

            for b in range(self.perturbations.size()[0]):
                # We explicitly do not check the norm here.
                # This allows to evaluate both success and average distance separately.
                if self.training_mode:
                    if error[b].item() < success_error[b] or success[b] < 0:
                        if other_classes.data[b] != self.classes.data[b] and success[b] < 0:
                            success[b] = i
                        success_error[b] = error[b].data.cpu()
                        success_perturbations[b] = numpy.copy(self.perturbations[b].data.cpu().numpy())
                        success_probabilities[b] = output_probabilities[b].data.cpu()
                        success_norms[b] = check_norm[b].data.cpu()
                else:
                    if other_classes.data[b] != self.classes.data[b] and success[b] < 0: # and check_norm.data[b] <= self.epsilon
                        success[b] = i
                        success_error[b] = error[b].data.cpu()
                        success_perturbations[b] = numpy.copy(self.perturbations[b].data.cpu().numpy())
                        success_probabilities[b] = output_probabilities[b].data.cpu()
                        success_norms[b] = check_norm[b].data.cpu()

            self.history.append({
                'iteration': i,
                'class': other_classes.cpu().numpy(),
                'error': error.detach().cpu().numpy(),
                'probabilities': output_probabilities.detach().cpu().numpy(),
                'norms': check_norm.detach().cpu().numpy()
            })

            common.torch.set_optimizer_parameter(self.optimizer, 'lr', self.base_lr * (self.lr_decay ** (1 + i / 100)))
            if verbose and i % self.skip == 0:
                log('[%s] %d: lr=%g objective=%g norm=%g bound=%g success=%g gradient=%g' % (self.__class__.__name__, i, self.base_lr * (self.lr_decay ** (1 + i / 100)), torch.sum(objective).data/batch_size, torch.sum(norm).data/batch_size, torch.sum(bound).data/batch_size, numpy.sum(success >= 0), gradient))

            # 5/
            # Break condition.
            if numpy.all(success >= 0) and not self.training_mode:
                if verbose:
                    log('[%s] %d: objective=%g norm=%g bound=%g success=%g gradient=%g' % (self.__class__.__name__, i, torch.sum(objective).data/batch_size, torch.sum(norm).data/batch_size, torch.sum(bound).data/batch_size, numpy.sum(success >= 0), gradient))
                break

            # Quick hack for handling the last iteration correctly.
            if i == self.max_iterations:
                if verbose:
                    log('[%s] %d: objective=%g norm=%g bound=%g success=%g gradient=%g' % (self.__class__.__name__, i, torch.sum(objective).data/batch_size, torch.sum(norm).data/batch_size, torch.sum(bound).data/batch_size, numpy.sum(success >= 0), gradient))
                break

            # 6/
            # Backprop error.
            error = torch.sum(error, 0)
            error.backward()
            self.optimizer.step()

            if verbose:
                gradient = torch.mean(torch.abs(self.perturbations.grad))

        for b in range(self.perturbations.size()[0]):
            # In any case, we return the current perturbations for non-successful attacks.
            if success[b] < 0:
                success_perturbations[b] = numpy.copy(self.perturbations[b].data.cpu().numpy())
                success_probabilities[b] = output_probabilities[b].data.cpu()
                success_norms[b] = check_norm[b].data.cpu()

        return success, success_perturbations, success_probabilities, success_norms, i

    def project(self):
        """
        Projection.
        """

        if self.auto_encoder is not None:
            self.perturbations.data = self.project_auto_encoder(self.perturbations.data)

    def norm(self):
        """
        Compute the norm to check.

        :return: norm of current perturbation
        :rtype: float
        """

        raise NotImplementedError()

    def norm_loss(self):
        """
        Norm loss.

        :return: loss based on norm/corresponding to norm constraint
        :rtype: torch.autograd.Variable
        """

        raise NotImplementedError()

    def bound_loss(self):
        """
        Bound loss.

        :return: loss to constrain [0,1]
        :rtype: torch.autograd.Variable
        """

        bound = torch.zeros_like(self.images)
        if self.max_bound is not None:
            bound = torch.max(torch.zeros_like(self.perturbations), self.perturbations + self.images - self.max_bound)
        if self.min_bound is not None:
            bound += torch.max(torch.zeros_like(self.perturbations), -self.perturbations - self.images + self.min_bound)
        return torch.sum(bound)

    def encoder_bound_loss(self):
        """
        Encoder bound loss.

        :return: loss for encode rbound constraint
        :rtype: torch.autograd.Variable
        """

        bound = torch.zeros_like(self.images)
        if self.encoder is not None:
            output = self.encoder.forward(self.images + self.perturbations)
            if self.encoder_max_bound is not None:
                bound = torch.max(torch.zeros_like(output), output - self.encoder_max_bound)
            if self.encoder_min_bound is not None:
                bound += torch.max(torch.zeros_like(output), -output + self.encoder_min_bound)
        return torch.sum(bound)