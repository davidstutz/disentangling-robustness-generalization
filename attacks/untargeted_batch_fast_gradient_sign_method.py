from .untargeted_batch_linf_normalized_gradient_method import *


class UntargetedBatchFastGradientSignMethod(UntargetedBatchLInfNormalizedGradientMethod):
    """
    Fast gradient sign method.
    """

    def __init__(self, model, images, classes=None, epsilon=0.5, base_lr=None, max_iterations=1):
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
        :param base_lr: learning rate
        :type base_lr: float
        """

        super(UntargetedBatchFastGradientSignMethod, self).__init__(model, images, classes, epsilon, base_lr, max_iterations)