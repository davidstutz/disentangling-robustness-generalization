import torch
import numpy
from .untargeted_attack import *
from .untargeted_objectives import *
from common import cuda
from common.log import log, LogLevel


class UntargetedBatchNormalizedGradientMethod(UntargetedAttack):
    """
    Implementation of untargetetd PGD attack.
    """

    def __init__(self, model, images, classes=None, epsilon=0.5, base_lr=None, max_iterations=1, max_projections=5):
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
        :param base_lr: learning rate if more than one iteration
        :type base_lr: float
        :param max_iterations: maximum number of iterations
        :type max_iterations: int
        :param max_projections: number of projections for alternating projection
        :type max_projections: int
        """

        super(UntargetedBatchNormalizedGradientMethod, self).__init__(model, images, classes)

        self.perturbations = None
        """ (torch.autograd.Variable) Perturbation of attack. """

        self.gradients = None
        """ (torch.autograd.Variable) Will hold gradients. """

        self.output_probabilities = None
        """ (torch.autograd.Variable) Output probabilities of perturbations. """

        self.epsilon = epsilon
        """ (float) Strength of attack. """

        self.max_iterations = max_iterations
        """ (int) Maximum number of iterations. """

        self.max_projections = max_projections
        """ (int) Maximum number of projections. """

        self.lr_decay = 1
        """ (float) Learning rate decay. """

        self.skip = 5
        """ (int) Verbosity skip. """

        self.history = []
        """ ([dict] History. """

        if base_lr is not None:
            self.base_lr = base_lr
            """ (float) Learning rate if more than one iterations. """
        else:
            self.base_lr = epsilon
            if self.max_iterations > 1:
                self.max_iterations = 1
                log('[Warning] base_lr not given, but more than one iterations specified, only running one iteration now', LogLevel.WARNING)

    def set_epsilon(self, epsilon):
        """
        Set epsilon.

        :param epsilon: maximum strength of attack
        :type epsilon: float
        """

        self.epsilon = epsilon

    def set_base_lr(self, base_lr):
        """
        Set base_lr.

        :param base_lr: learning rate for multiple iterations
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

    def set_max_projections(self, max_projections):
        """
        Set max projections
        :param max_projections: number of projections for alternating projection
        :type max_projections: int
        """

        self.max_projections = max_projections

    def initialize(self):
        """
        Initialize the attack.
        """

        # Zero initialization is standard for FGSM.
        self.initialize_zero()

    def initialize_random(self):
        """
        Initialize the attack.
        """

        raise NotImplementedError('initialize_random should be implemented by child classes')

    def initialize_zero(self):
        """
        Initialize the attack.
        """

        random = numpy.zeros(self.images.size())
        self.perturbations = torch.from_numpy(random.astype(numpy.float32))
        if cuda.is_cuda(self.model):
            self.perturbations = self.perturbations.cuda()
        self.perturbations = torch.autograd.Variable(self.perturbations, requires_grad=True)

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

        assert isinstance(untargeted_objective, UntargetedObjective), 'expected an objective of type UntargetedObjective, got %s instead' % untargeted_objective

        # self.perturbations IS TRANSFERRED TO CUDA IN INITIALIZATION AS self.perturbations.grad
        # WILL NOT WORK ON NON-LEAF VARIABLES!

        self.gradients = torch.zeros_like(self.perturbations)
        #self.gradients = torch.autograd.Variable(self.gradients, requires_grad=False)
        if cuda.is_cuda(self.model):
            self.gradients = self.gradients.cuda()

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
            self.gradients.zero_()

            # 0/
            # Project current perturbation.
            self.project()

            # 1/
            # Reset gradients and compute the logits for the current perturbations.
            # This is not applicable to batches, so check that the logits have been computed for one sample
            # only.
            self.gradients.zero_()
            output_logits = self.model.forward(self.images + self.perturbations)

            # 2/
            # Compute current probabilities and current class.
            # This is mainly used as return value and to check the break criterion, i.e.
            # if the class actually changes.
            output_probabilities = torch.nn.functional.softmax(output_logits, 1)
            _, other_classes = torch.max(output_probabilities, 1)

            # 3/
            # Compute the objective to take gradients from.
            error = untargeted_objective.f(output_logits, self.logits, self.classes)

            # 4/
            # Logging and break condition.
            check_norm = self.norm()
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

            if verbose and i % self.skip == 0:
                log('[%s] %d: objective=%g success=%g gradient=%g' % (self.__class__.__name__, i, torch.sum(error.data)/batch_size, numpy.sum(success >= 0), gradient))

            # 5/
            # Break condition.
            if numpy.all(success >= 0) and not self.training_mode:
                if verbose:
                    log('[%s] %d: objective=%g success=%g gradient=%g' % (self.__class__.__name__, i, torch.sum(error.data)/batch_size, numpy.sum(success >= 0), gradient))
                break

            # Quick hack for handling the last iteration correctly.
            if i == self.max_iterations:
                if verbose:
                    log('[%s] %d: objective=%g success=%g gradient=%g' % (self.__class__.__name__, i, torch.sum(error.data)/batch_size, numpy.sum(success >= 0), gradient))
                break

            # 6/
            # Put together the error for differentiation and do backward pass.
            error = torch.sum(error, 0)
            error.backward()

            # 7/
            # Get the gradients and normalize.
            self.gradients = self.perturbations.grad.clone()
            self.normalize()

            # 8/
            # Update step according to learning rate.
            self.perturbations.data -= self.base_lr*self.gradients

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
        Clip perturbation.
        """

        raise NotImplementedError()

    def norm(self):
        """
        Norm.

        :return: norm of current perturbation
        :rtype: float
        """

        raise NotImplementedError()

    def normalize(self):
        """
        Normalize gradients.
        """

        raise NotImplementedError()