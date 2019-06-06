import torch
import numpy
from . import cuda


def one_hot(classes, C):
    """
    Convert class labels to one-hot vectors.

    :param classes: classes as B x 1
    :type classes: torch.autograd.Variable or torch.Tensor
    :param C: number of classes
    :type C: int
    :return: one hot vector as B x C
    :rtype: torch.autograd.Variable or torch.Tensor
    """

    assert isinstance(classes, torch.autograd.Variable) or isinstance(classes, torch.Tensor), 'classes needs to be torch.autograd.Variable or torch.Tensor'
    assert len(classes.size()) == 2 or len(classes.size()) == 1, 'classes needs to have rank 2 or 1'
    assert C > 0

    if len(classes.size()) < 2:
        classes = classes.view(-1, 1)

    one_hot = torch.Tensor(classes.size(0), C)
    if cuda.is_cuda(classes):
         one_hot = one_hot.cuda()

    if isinstance(classes, torch.autograd.Variable):
        one_hot = torch.autograd.Variable(one_hot)

    one_hot.zero_()
    one_hot.scatter_(1, classes, 1)

    return one_hot


def set_optimizer_parameter(optimizer, parameter, value):
    """
    Shorthand for setting an optimizer parameter.

    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param parameter: parameter
    :type parameter: str
    :param value: value
    :type value: mixed
    """

    for param_group in optimizer.param_groups:
        param_group[parameter] = value


def project(tensor, epsilon=1, ord=2):
    """
    Compute the orthogonal projection of the input tensor (as vector) onto the L_ord epsilon-ball.

    **Assumes the first dimension to be batch dimension, which is preserved.**

    :param tensor: variable or tensor
    :type tensor: torch.autograd.Variable or torch.Tensor
    :param epsilon: radius of ball.
    :type epsilon: float
    :param ord: order of norm
    :type ord: int
    :return: projected vector
    :rtype: torch.autograd.Variable or torch.Tensor
    """

    assert isinstance(tensor, torch.Tensor) or isinstance(tensor, torch.autograd.Variable), 'given tensor should be torch.Tensor or torch.autograd.Variable'

    if ord == 2:
        size = tensor.size()
        flattened_size = numpy.prod(numpy.array(size[1:]))

        tensor = tensor.view(-1, flattened_size)
        clamped = torch.clamp(epsilon/torch.norm(tensor, 2, dim=1), max=1)
        clamped = clamped.view(-1, 1)

        tensor = tensor * clamped
        if len(size) == 4:
            tensor = tensor.view(-1, size[1], size[2], size[3])
        elif len(size) == 2:
            tensor = tensor.view(-1, size[1])
    elif ord == float('inf'):
        tensor = torch.clamp(tensor, min=-epsilon, max=epsilon)
    else:
        raise NotImplementedError()

    return tensor


def tensor_or_value(mixed):
    """
    Get tensor or single value.

    :param mixed: variable, tensor or value
    :type mixed: mixed
    :return: tensor or value
    :rtype: torch.Tensor or value
    """

    if isinstance(mixed, torch.Tensor):
        if mixed.numel() > 1:
            return mixed
        else:
            return mixed.item()
    elif isinstance(mixed, torch.autograd.Variable):
        return tensor_or_value(mixed.cpu().data)
    else:
        return mixed


def as_variable(mixed, gpu=False, grads=False):
    """
    Get a tensor or numpy array as variable.

    :param mixed: input tensor
    :type mixed: torch.Tensor or numpy.ndarray
    :param gpu: gpu or not
    :type gpu: bool
    :param grads: gradients
    :type grads: bool
    :return: variable
    :rtype: torch.autograd.Variable
    """

    assert isinstance(mixed, numpy.ndarray) or isinstance(mixed, torch.Tensor), 'input needs to be numpy.ndarray or torch.Tensor'

    if isinstance(mixed, numpy.ndarray):
        mixed = torch.from_numpy(mixed)

    if gpu:
        mixed = mixed.cuda()
    return torch.autograd.Variable(mixed, grads)


def cross_entropy(pred, target):
    """
    "Proper" cross entropy between distributions.

    :param pred: predicted logits
    :type pred: torch.autograd.Variable
    :param target: target distributions
    :type target: torch.autograd.Variable
    :return: loss
    :rtype: torch.autograd.Variable
    """

    return torch.sum(- target * torch.nn.functional.log_softmax(pred, 1), 1)


class View(torch.nn.Module):
    """
    Simple view layer.
    """

    def __init__(self, *args):
        """
        Constructor.

        :param args: shape
        :type args: [int]
        """

        super(View, self).__init__()

        self.shape = args

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return input.view(self.shape)


class Clamp(torch.nn.Module):
    """
    Wrapper for clamp.
    """

    def __init__(self, min=0, max=1):
        """
        Constructor.
        """

        super(Clamp, self).__init__()

        self.min = min
        """ (float) Min value. """

        self.max = max
        """ (float) Max value. """

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return torch.clamp(torch.clamp(input, min=self.min), max=self.max)


class Scale(torch.nn.Module):
    """
    Simply scaling layer, mainly to allow simple saving and loading.
    """

    def __init__(self, shape):
        """
        Constructor.

        :param shape: shape
        :type shape: [int]
        """

        super(Scale, self).__init__()

        # https://github.com/pytorch/pytorch/issues/8104
        self.register_buffer('min', torch.Tensor(shape))
        self.register_buffer('max', torch.Tensor(shape))

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return self.min + torch.mul(self.max - self.min, input)

