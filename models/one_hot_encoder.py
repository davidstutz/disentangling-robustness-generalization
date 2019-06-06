import torch
import common.torch
from .encoder import Encoder


class OneHotEncoder(Encoder):
    """
    Encodes font and class separately in one-hot vectors.
    """

    def __init__(self, N_font, N_class, N_theta, theta_bounds, resolution=32, architecture='standard', **kwargs):
        """
        Initialize encoder.

        :param N_font: number of fonts
        :type N_font: int
        :param N_class: number of classes
        :type N_class: int
        :param N_theta: number of transformation codes
        :type N_theta: int
        :param theta_bounds: theta bounds (min and max)
        :type theta_bounds: (numpy.ndarray, numpy.ndarray)
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param architecture: architecture builder to use
        :type architecture: str
        """

        super(OneHotEncoder, self).__init__(N_font, N_class, N_theta, resolution, architecture, **kwargs)

        representation = getattr(self, 'view').shape[1]
        self.linear_code = torch.nn.Linear(representation, self.N_class*self.N_font)
        self.linear_theta = torch.nn.Linear(representation, self.N_theta)
        self.sigmoid_theta = torch.nn.Sigmoid()
        self.scale_theta = common.torch.Scale(N_theta)

        if theta_bounds is not None:
            self.scale_theta.min = torch.from_numpy(theta_bounds[0])
            self.scale_theta.max = torch.from_numpy(theta_bounds[1])

        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.kaiming_normal_(self.linear_code.weight, gain)
        torch.nn.init.constant_(self.linear_code.bias, 0)
        torch.nn.init.kaiming_normal_(self.linear_theta.weight, gain)
        torch.nn.init.constant_(self.linear_theta.bias, 0)

    def _forward(self, image):
        """
        Forward pass, takes a code(s) and generates the corresponding image(s).

        :param image: input image
        :type image: torch.autograd.Variable
        :return: output code and transformation
        :rtype: (torch.autograd.Variable, torch.autograd.Variable)
        """

        representation = image
        for name in self.layers:
            representation = getattr(self, name)(representation)

        code = self.linear_code(representation)
        theta = self.scale_theta(self.sigmoid_theta(self.linear_theta(representation)))
        #theta = self.min_theta + torch.mul(self.max_theta - self.min_theta, theta)

        return code, theta

