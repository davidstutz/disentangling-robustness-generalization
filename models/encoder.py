import torch
import common.torch


class Encoder(torch.nn.Module):
    """
    Encoder interface class.
    """

    def add_layer(self, name, layer):
        """
        Add a layer.
        :param name:
        :param layer:
        :return:
        """

        setattr(self, name, layer)
        self.layers.append(name)

    def standard(self, resolution, **kwargs):
        """
        Standard architeucture.

        :param resolution: input resolution (assumed square)
        :type resolution: int
        """

        #
        # All layer names will be saved in self.layers.
        # This allows the forward method to run the network without
        # knowing its exact structure.
        # The representation layer is assumed to be labeled accordingly.
        #

        layer = 0
        channels = []
        resolutions = []

        # Determine parameters of network
        activation = torch.nn.ReLU
        kwargs_activation = kwargs.get('activation', 'relu')
        gain = torch.nn.init.calculate_gain(kwargs_activation)
        kwargs_activation = kwargs_activation.lower()
        if kwargs_activation == 'sigmoid':
            activation = torch.nn.Sigmoid
        elif kwargs_activation == 'tanh':
            activation = torch.nn.Tanh
        batch_normalization = kwargs.get('batch_normalization', True)
        dropout = kwargs.get('dropout', False)
        start_channels = kwargs.get('start_channels', 16)

        while True:
            input_channels = 1 if layer == 0 else channels[layer - 1]
            output_channels = start_channels if layer == 0 else channels[layer - 1] * 2

            dim = torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
            conv = torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
            pool = torch.nn.MaxPool2d(2, stride=2)
            relu = activation(True)

            torch.nn.init.kaiming_normal_(dim.weight, gain)
            torch.nn.init.constant_(dim.bias, 0)
            torch.nn.init.kaiming_normal_(conv.weight, gain)
            torch.nn.init.constant_(conv.bias, 0)

            self.add_layer('dim%d' % layer, dim)
            self.add_layer('conv%d' % layer, conv)
            self.add_layer('pool%d' % layer, pool)
            self.add_layer('act%d' % layer, relu)

            if batch_normalization:
                bn = torch.nn.BatchNorm2d(output_channels)
                torch.nn.init.constant_(bn.weight, 1)
                torch.nn.init.constant_(bn.bias, 0)
                self.add_layer('bn%d' % layer, bn)

            channels.append(output_channels)
            resolutions.append(resolution // 2 if layer == 0 else resolutions[layer - 1] // 2)
            if resolutions[-1] // 2 < 2 or resolutions[-1] % 2 == 1:
                break;

            # Only add dropout if this was not the last layer.
            if dropout:
                drop = torch.nn.Dropout2d()
                self.add_layer('drop%d' % layer, drop)

            layer += 1

        representation = resolutions[-1] * resolutions[-1] * channels[-1]
        view = common.torch.View(-1, representation)
        self.add_layer('view', view)

    def __init__(self, N_font, N_class, N_theta, resolution=32, architecture='standard', **kwargs):
        """
        Initialize encoder.

        :param N_font: number of fonts
        :type N_font: int
        :param N_class: number of classes
        :type N_class: int
        :param N_theta: number of transformation codes
        :type N_theta: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param architecture: architecture builder to use
        :type architecture: str
        """

        super(Encoder, self).__init__()

        assert N_font > 0, 'positive N_font expected'
        assert N_class > 0, 'positive N_class expected'
        assert N_theta > 0, 'positive N_theta expected'
        assert resolution > 0

        self.N_font = N_font
        """ (int) Number of fonts. """

        self.N_class = N_class
        """ (int) Number of classes. """

        self.N_theta = N_theta
        """ (int) Number of transformation parameters. """

        self.layers = []
        """ ([str]) Will hold layer names. """

        self._theta = None
        """ (None or torch.autograd.Variable) Fixed theta if set. """

        self._code = None
        """ (None or torch.autograd.Variable) Fixed code if set. """

        if architecture == 'standard':
            self.standard(resolution, **kwargs)
        else:
            raise NotImplementedError()

    def forward(self, image):
        """
        Forward image, and output code and theta depending on what ahs been fixed.

        :param image: theta or None
        :type image: torch.autograd.Variable or None
        :return: output image
        :rtype: torch.autograd.Variable
        """

        code, theta = self._forward(image)
        if self._theta is not None:
            return code, self._theta
        elif self._code is not None:
            return self._code, theta
        else:
            return code, theta

    def set_theta(self, theta):
        """
        Set fixed theta to use in forward pass.

        :param theta: theta
        :type theta: torch.autograd.Variable
        """

        self._theta = theta

    def set_code(self, code):
        """
        Set fixed code to use in forward pass.

        :param code: code
        :type code: torch.autograd.Variable
        """

        self._code = code

    def _forward(self, code, theta):
        """
        Forward pass, takes a code(s) and generates the corresponding image(s).

        :param code: input code(s)
        :type code: torch.autograd.Variable
        :param theta: input transformation(s)
        :type theta: torch.autograd.Variable
        :return: output image
        :rtype: torch.autograd.Variable
        """

        raise NotImplementedError()

