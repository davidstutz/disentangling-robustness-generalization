import torch
from .untargeted_batch_gradient_descent import *
from .untargeted_objectives import *
from common import cuda


class UntargetedBatchReparameterizedGradientDescent(UntargetedBatchGradientDescent):
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

        super(UntargetedBatchReparameterizedGradientDescent, self).__init__(model, images, classes, epsilon, 0.0, c, base_lr)

        # Reparameterization according to Carlini and Wagner:
        # perturbation = 0.5*(tanh(w) + 1) - image
        self.w = None
        """ (torch.autograd.Variable) This will be our reparameterized variables to optimize. """

        # Just to repeat:
        self.perturbations = None
        """ (torch.autograd.Variable) This is our actual perturbation. """

        self.EPS = 1e-5
        """ (float) For robustness of tanh / arctanh. """

    def set_auto_encoder(self, auto_encoder):
        """
        Not possible for the reparaetermization.
        """

        raise NotImplementedError()

    def initialize(self):
        """
        Initialize.
        """

        self.initialize_zero()

    def initialize_random(self):
        """
        Initialize the attack.
        """

        raise NotImplementedError('initialize_random needs to be implemented by child classes')

    def initialize_zero(self):
        """
        Initialize the attack.
        """

        assert self.min_bound is not None, 'reparameterization only works with valid upper and lower bounds'
        assert self.max_bound is not None, 'reparameterization only works with valid upper and lower bounds'
        
        self.w = numpy.arctanh(
            (2-self.EPS)
            * (self.images.data.cpu().numpy() - self.min_bound.cpu().numpy())
                / (self.max_bound.cpu().numpy() - self.min_bound.cpu().numpy())
            - 1 + self.EPS)
        self.w = torch.from_numpy(self.w)
        self.w = torch.autograd.Variable(self.w, requires_grad=True)

        if cuda.is_cuda(self.model):
            self.w = torch.autograd.Variable(self.w.cuda(), requires_grad=True)
        else:
            self.w = torch.autograd.Variable(self.w, requires_grad=True)

    def initialize_optimizer(self):
        """
        Initalize optimizer and parameters to optimize.
        """

        if cuda.is_cuda(self.model):
            self.w = self.w.cuda()

        # We directly optimize the reparameterized variables w!
        self.w = torch.nn.Parameter(self.w.data)
        self.optimizer = torch.optim.Adam([self.w], lr=self.base_lr)

    def project(self):
        """
        Reparameterization.
        """

        raise NotImplementedError('projection not necessary in Carlini+Wagner attacks')

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

        assert self.w is not None, 'attack was not initialized properly'
        assert isinstance(untargeted_objective, UntargetedObjective), 'expected an objective of type UntargetedObjective, got %s instead' % untargeted_objective
        self.initialize_optimizer()

        # Will hold the individually best results (however, optimization
        # is run for all samples until all are successful or the maximum number
        # of iterations is reached.
        success = numpy.ones((self.w.size()[0]), dtype=numpy.int32)*-1
        success_error = numpy.zeros((self.w.size()[0]), dtype=numpy.float32)
        success_perturbations = numpy.zeros(self.w.size(), dtype=numpy.float32)
        success_probabilities = numpy.zeros((self.w.size()[0], self.logits.size()[1]), dtype=numpy.float32)
        success_norms = numpy.zeros((self.w.size()[0]), dtype=numpy.float32)

        i = 0
        gradient = 0

        for i in range(self.max_iterations + 1):
            # MAIN LOOP OF ATTACK
            # ORDER IMPORTANT

            self.optimizer.zero_grad()

            self.perturbations = (0.5 + self.EPS) * (torch.nn.functional.tanh(self.w) + 1)
            self.perturbations = torch.mul(self.max_bound - self.min_bound, self.perturbations) + self.min_bound

            # 1/
            # Reset gradients and compute the logits for the current perturbations.
            # This is not applicable to batches, so check that the logits have been computed for one sample
            # only.
            output_logits = self.model.forward(self.perturbations)

            # 2/
            # Compute current probabilities and current class.
            # This is mainly used as return value and to check the break criterion, i.e.
            # if the class actually changes.
            output_probabilities = torch.nn.functional.softmax(output_logits, 1)
            _, other_classes = torch.max(output_probabilities, 1)

            # 3/
            # Compute the error components:
            # - norm of the perturbation to enforce the epsilon-ball constraint
            # - the actual objective given the untargeted objective to use
            # The latter two components are weighted relative to the norm.
            # The weighting together with the used learning rate is actually quite important!
            norm = self.norm_loss() # Will be sum of norms!
            objective = untargeted_objective.f(output_logits, self.logits, self.classes)

            # Put together the error.
            error = self.c_0*norm + self.c_2*objective

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
                        success_perturbations[b] = numpy.copy(self.perturbations[b].data.cpu().numpy() - self.images[b].data.cpu().numpy())
                        success_probabilities[b] = output_probabilities[b].data.cpu()
                        success_norms[b] = check_norm[b].data.cpu()
                else:
                    if other_classes.data[b] != self.classes.data[b] and success[b] < 0: # and check_norm.data[b] <= self.epsilon
                        success[b] = i
                        success_error[b] = error[b].data.cpu()
                        success_perturbations[b] = numpy.copy(self.perturbations[b].data.cpu().numpy() - self.images[b].data.cpu().numpy())
                        success_probabilities[b] = output_probabilities[b].data.cpu()
                        success_norms[b] = check_norm[b].data.cpu()

            self.history.append({
                'iteration': i,
                'class': other_classes.cpu().numpy(),
                'error': error.detach().cpu().numpy(),
                'objective': objective.detach().cpu().numpy(),
                'probabilities': output_probabilities.detach().cpu().numpy(),
                'norms': check_norm.detach().cpu().numpy()
            })

            common.torch.set_optimizer_parameter(self.optimizer, 'lr', self.base_lr * (self.lr_decay ** (1 + i / 100)))
            if verbose and i % self.skip == 0:
                log('[%s] %d: lr=%g objective=%g norm=%g success=%g gradient=%g' % (self.__class__.__name__, i, self.base_lr * (self.lr_decay ** (1 + i / 100)), torch.sum(objective).data/batch_size, torch.sum(norm).data/batch_size, numpy.sum(success >= 0), gradient))

            # 5/
            # Break condition.
            if numpy.all(success >= 0) and not self.training_mode:
                if verbose:
                    log('[%s] %d: objective=%g norm=%g success=%g gradient=%g' % (self.__class__.__name__, i, torch.sum(objective).data/batch_size, torch.sum(norm).data/batch_size, numpy.sum(success >= 0), gradient))
                break

            # Quick hack for handling the last iteration correctly.
            if i == self.max_iterations:
                if verbose:
                    log('[%s] %d: objective=%g norm=%g success=%g gradient=%g' % (self.__class__.__name__, i, torch.sum(objective).data/batch_size, torch.sum(norm).data/batch_size, numpy.sum(success >= 0), gradient))
                break

            # 6/
            # Backprop error.
            error = torch.sum(error, 0)
            error.backward()
            self.optimizer.step()

            if verbose:
                gradient = torch.mean(torch.abs(self.w.grad))

        for b in range(self.perturbations.size()[0]):
            # In any case, we return the current perturbations for non-successful attacks.
            if success[b] < 0:
                success_perturbations[b] = numpy.copy(self.perturbations[b].data.cpu().numpy() - self.images[b].data.cpu().numpy())
                success_probabilities[b] = output_probabilities[b].data.cpu()
                success_norms[b] = check_norm[b].data.cpu()

        return success, success_perturbations, success_probabilities, success_norms, i

    def bound_loss(self):
        """
        Bound loss.

        :return: loss to constrain [0,1]
        :rtype: torch.autograd.Variable
        """

        raise NotImplementedError()
