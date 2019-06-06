import torch


class UntargetedObjective:
    """
    Untargeted attack objective.
    """

    def f(self, current_logits, reference_logits, classes):
        """
        Objective function.

        :param current_logits: logit output of the network
        :type current_logits: torch.autograd.Variable
        :param reference_logits: "true" logits
        :type reference_logits: torch.autograd.Variable
        :param classes: true classes
        :type classes: torch.autograd.Variable
        :return: error
        :rtype: torch.autograd.Variable
        """

        raise NotImplementedError()


class UntargetedF0(UntargetedObjective):
    """
    Untargeted attack objective.
    """

    def f(self, current_logits, reference_logits, classes):
        """
        Objective function.

        :param current_logits: logit output of the network
        :type current_logits: torch.autograd.Variable
        :param reference_logits: "true" logits
        :type reference_logits: torch.autograd.Variable
        :param classes: true classes
        :type classes: torch.autograd.Variable
        :return: error
        :rtype: torch.autograd.Variable
        """

        return -torch.nn.functional.cross_entropy(current_logits, classes, size_average=False, reduce=False)


class UntargetedF6(UntargetedObjective):
    """
    Untargeted attack objective.
    """

    def __init__(self, kappa=-3):
        """
        Constructor.

        :param kappa: confidence threshold
        :type kappa: float
        """

        self.kappa = kappa
        """ (float) Confidence threshold. """

    def f(self, current_logits, reference_logits, classes):
        """
        Objective function.

        :param current_logits: logit output of the network
        :type current_logits: torch.autograd.Variable
        :param reference_logits: "true" logits
        :type reference_logits: torch.autograd.Variable
        :param classes: true classes
        :type classes: torch.autograd.Variable
        :return: error
        :rtype: torch.autograd.Variable
        """

        other_logits = current_logits.clone()
        other_logits[torch.arange(0, current_logits.size()[0]).long(), classes.long()] = 0
        difference = current_logits[torch.arange(0, current_logits.size()[0]).long(), classes.long()] - torch.max(other_logits, 1)[0]
        return torch.max(torch.ones_like(difference)*self.kappa, difference)
