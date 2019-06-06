import torch
import common.torch


class LearnedDecoder(torch.nn.Module):
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

        activation = None
        kwargs_activation = kwargs.get('activation', 'relu')
        kwargs_activation = kwargs_activation.lower()
        if kwargs_activation == 'relu':
            activation = torch.nn.ReLU
        elif kwargs_activation == 'sigmoid':
            activation = torch.nn.Sigmoid
        elif kwargs_activation == 'tanh':
            activation = torch.nn.Tanh
        elif kwargs_activation == 'leaky_relu':
            activation = torch.nn.LeakyReLU
        elif kwargs_activation == 'none':
            pass
        else:
            raise ValueError('Unsupported activation %s.' % kwargs_activation)

        gain = 1
        if activation:
            gain = torch.nn.init.calculate_gain(kwargs_activation)

        batch_normalization = kwargs.get('batch_normalization', True)
        start_channels = kwargs.get('start_channels', 16)

        layer = 0
        channels = []
        resolutions = []

        # This basically mimicks the encoder loop to get the same number of layers and everything.
        while True:
            channels.append(start_channels if layer == 0 else channels[layer - 1] * 2)
            resolutions.append([
                resolution[1] // 2 if layer == 0 else resolutions[layer - 1][0] // 2,
                resolution[2] // 2 if layer == 0 else resolutions[layer - 1][1] // 2,
            ])
            if resolutions[-1][0] // 2 < 3 or resolutions[-1][1] // 2 < 3:
                break;
            layer += 1

        representation = resolutions[-1][0] * resolutions[-1][1] * channels[-1]
        linear_code = torch.nn.Linear(self.N_latent, representation)
        self.add_layer('code', linear_code)
        #torch.nn.init.kaiming_normal_(linear_code.weight, gain)
        torch.nn.init.normal_(linear_code.weight, 0, 0.001)
        torch.nn.init.constant_(linear_code.bias, 0)

        view = common.torch.View(-1, channels[-1], resolutions[-1][0], resolutions[-1][1])
        self.add_layer('view0', view)

        layer = 1
        channels.insert(0, resolution[0])
        resolutions.insert(0, [resolution[1], resolution[2]])

        for i in range(len(channels) - 1, 0, -1):
            start_channels = channels[i]
            output_channels = channels[i - 1]

            output_padding = [0, 0]
            for j in range(2):
                if resolutions[i][j] == 3:
                    if resolutions[i][j]*2 != resolutions[i - 1][j]:
                        output_padding[j] = 1

            conv = torch.nn.ConvTranspose2d(start_channels, output_channels, 6, stride=2, padding=2, output_padding=output_padding)
            #torch.nn.init.kaiming_normal_(conv.weight, gain)
            torch.nn.init.normal_(conv.weight, 0, 0.001)
            torch.nn.init.constant_(conv.bias, 0)
            self.add_layer('conv%d' % layer, conv)

            if batch_normalization and i > 1:
                bn = torch.nn.BatchNorm2d(output_channels)
                torch.nn.init.constant_(bn.weight, 1)
                torch.nn.init.constant_(bn.bias, 0)
                self.add_layer('bn%d' % layer, bn)

            if i > 1:
                if activation:
                    act = activation(True)
                    self.add_layer('act%d' % layer, act)
            else:
                act = torch.nn.Sigmoid()
                self.add_layer('act%d' % layer, act)

            layer += 1

    def pool(self, resolution, **kwargs):
        """
        Standard architeucture.

        :param resolution: input resolution (assumed square)
        :type resolution: int
        """

        activation = None
        kwargs_activation = kwargs.get('activation', 'relu')
        kwargs_activation = kwargs_activation.lower()
        if kwargs_activation == 'relu':
            activation = torch.nn.ReLU
        elif kwargs_activation == 'sigmoid':
            activation = torch.nn.Sigmoid
        elif kwargs_activation == 'tanh':
            activation = torch.nn.Tanh
        elif kwargs_activation == 'leaky_relu':
            activation = torch.nn.LeakyReLU
        elif kwargs_activation == 'none':
            pass
        else:
            raise ValueError('Unsupported activation %s.' % kwargs_activation)

        gain = 1
        if activation:
            gain = torch.nn.init.calculate_gain(kwargs_activation)

        batch_normalization = kwargs.get('batch_normalization', True)
        start_channels = kwargs.get('start_channels', 16)

        layer = 0
        channels = []
        resolutions = []

        # This basically mimicks the encoder loop to get the same number of layers and everything.
        while True:
            channels.append(start_channels if layer == 0 else channels[layer - 1] * 2)
            resolutions.append([
                resolution[1] // 2 if layer == 0 else resolutions[layer - 1][0] // 2,
                resolution[2] // 2 if layer == 0 else resolutions[layer - 1][1] // 2,
            ])
            if resolutions[-1][0] // 2 < 3 or resolutions[-1][0]%2 == 1 or resolutions[-1][1] // 2 < 3 or resolutions[-1][1]%2 == 1:
                break;
            layer += 1

        representation = resolutions[-1][0] * resolutions[-1][1] * channels[-1]
        linear_code = torch.nn.Linear(self.N_latent, representation)
        self.add_layer('code', linear_code)
        #torch.nn.init.kaiming_normal_(linear_code.weight, gain)
        torch.nn.init.normal_(linear_code.weight, 0, 0.0005)
        torch.nn.init.constant_(linear_code.bias, 0)

        view = common.torch.View(-1, channels[-1], resolutions[-1][0], resolutions[-1][1])
        self.add_layer('view0', view)

        layer = 1
        channels.insert(0, resolution[0])
        resolutions.insert(0, [resolution[1], resolution[2]])

        for i in range(len(channels) - 1, 0, -1):
            start_channels = channels[i]
            output_channels = channels[i - 1]

            pool = torch.nn.UpsamplingNearest2d(scale_factor=2)
            conv = torch.nn.Conv2d(start_channels, output_channels, 5, stride=1, padding=2)
            #torch.nn.init.kaiming_normal_(conv.weight, gain)
            torch.nn.init.normal_(conv.weight, 0, 0.0005)
            torch.nn.init.constant_(conv.bias, 0)
            self.add_layer('pool%d' % layer, pool)
            self.add_layer('conv%d' % layer, conv)

            if batch_normalization and i > 1:
                bn = torch.nn.BatchNorm2d(output_channels)
                torch.nn.init.constant_(bn.weight, 1)
                torch.nn.init.constant_(bn.bias, 0)
                self.add_layer('bn%d' % layer, bn)

            if i > 1:
                if activation:
                    act = activation(True)
                    self.add_layer('act%d' % layer, act)
            else:
                act = torch.nn.Sigmoid()
                self.add_layer('act%d' % layer, act)

            layer += 1

    def mlp(self, resolution, **kwargs):
        """
        MLP architecture.

        :param resolution: input resolution (assumed square)
        :type resolution: int
        """

        activation = None
        kwargs_activation = kwargs.get('activation', 'relu')
        kwargs_activation = kwargs_activation.lower()
        if kwargs_activation == 'relu':
            activation = torch.nn.ReLU
        elif kwargs_activation == 'sigmoid':
            activation = torch.nn.Sigmoid
        elif kwargs_activation == 'tanh':
            activation = torch.nn.Tanh
        elif kwargs_activation == 'leaky_relu':
            activation = torch.nn.LeakyReLU
        elif kwargs_activation == 'none':
            pass
        else:
            raise ValueError('Unsupported activation %s.' % kwargs_activation)

        gain = 1
        if activation:
            gain = torch.nn.init.calculate_gain(kwargs_activation)

        batch_normalization = kwargs.get('batch_normalization', True)
        units = kwargs.get('units', [256, 256, 256])

        units = [resolution[0] * resolution[1] * resolution[2]] + units
        linear_code = torch.nn.Linear(self.N_latent, units[-1])
        self.add_layer('code', linear_code)
        #torch.nn.init.kaiming_normal_(linear_code.weight, gain)
        torch.nn.init.normal_(linear_code.weight, 0, 0.001)
        torch.nn.init.constant_(linear_code.bias, 0)

        layer = 1
        for i in range(len(units) - 1, 0, -1):
            in_features = units[i]
            out_features = units[i - 1]

            lin = torch.nn.Linear(in_features=in_features, out_features=out_features)
            #torch.nn.init.kaiming_normal_(lin.weight, gain)
            torch.nn.init.normal_(lin.weight, 0, 0.001)
            torch.nn.init.constant_(lin.bias, 0)
            self.add_layer('lin%d' % layer, lin)

            if i > 1:
                if activation:
                    act = activation(True)
                    self.add_layer('act%d' % layer, act)
            else:
                act = torch.nn.Sigmoid()
                self.add_layer('act%d' % layer, act)

            if batch_normalization and i > 1:
                bn = torch.nn.BatchNorm1d(out_features)
                torch.nn.init.constant_(bn.weight, 1)
                torch.nn.init.constant_(bn.bias, 0)
                self.add_layer('bn%d' % layer, bn)

            layer += 1

        view = common.torch.View(-1, resolution[0], resolution[1], resolution[2])
        self.add_layer('view', view)

    def dcgan(self, resolution, **kwargs):
        """
        Standard architeucture.

        :param resolution: input resolution (assumed square)
        :type resolution: int
        """

        gain = torch.nn.init.calculate_gain('relu')
        start_channels = kwargs.get('start_channels', 16)

        layer = 0
        channels = []
        resolutions = []

        # This basically mimicks the encoder loop to get the same number of layers and everything.
        while True:
            channels.append(start_channels if layer == 0 else channels[layer - 1] * 2)
            resolutions.append([
                resolution[1] // 2 if layer == 0 else resolutions[layer - 1][0] // 2,
                resolution[2] // 2 if layer == 0 else resolutions[layer - 1][1] // 2,
            ])
            if resolutions[-1][0] // 2 < 3 or resolutions[-1][1] // 2 < 3:
                break;
            layer += 1

        representation = channels[-1] * resolutions[-2][0] * resolutions[-2][1]
        linear_code = torch.nn.Linear(self.N_latent, representation)
        self.add_layer('code', linear_code)
        #torch.nn.init.kaiming_normal_(linear_code.weight, gain)
        torch.nn.init.normal_(linear_code.weight, 0, 0.001)
        torch.nn.init.constant_(linear_code.bias, 0)

        view = common.torch.View(-1, channels[-1], resolutions[-2][0], resolutions[-2][1])
        self.add_layer('view0', view)

        layer = 1
        channels.insert(0, resolution[0])
        resolutions.insert(0, [resolution[1], resolution[2]])

        for i in range(len(channels) - 1, 1, -1):
            start_channels = channels[i]
            output_channels = channels[i - 1]

            output_padding = 1 if resolutions[i - 1] == 3 or resolutions[i - 1] == 3 else 0
            conv = torch.nn.ConvTranspose2d(start_channels, output_channels, 4, stride=2, padding=1, output_padding=output_padding)
            #torch.nn.init.kaiming_normal_(conv.weight, gain)
            torch.nn.init.normal_(conv.weight, 0, 0.001)
            torch.nn.init.constant_(conv.bias, 0)
            self.add_layer('conv%d' % layer, conv)

            bn = torch.nn.BatchNorm2d(output_channels)
            torch.nn.init.constant_(bn.weight, 1)
            torch.nn.init.constant_(bn.bias, 0)
            self.add_layer('bn%d' % layer, bn)

            act = torch.nn.ReLU(True)
            self.add_layer('act%d' % layer, act)

            layer += 1

        # should be four, but that would screw up resolution
        conv = torch.nn.Conv2d(channels[1], channels[0], 3, stride=1, padding=1)
        #torch.nn.init.kaiming_normal_(conv.weight, gain)
        torch.nn.init.normal_(conv.weight, 0, 0.001)
        torch.nn.init.constant_(conv.bias, 0)
        self.add_layer('conv%d' % layer, conv)

        act = torch.nn.Sigmoid()
        self.add_layer('act%d' % layer, act)

    def __init__(self, N_latent, resolution=(1, 32, 32), architecture='standard', **kwargs):
        """
        Initialize encoder.

        :param N_latent: number of fonts
        :type N_latent: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param architecture: architecture builder to use
        :type architecture: str
        """

        super(LearnedDecoder, self).__init__()

        assert N_latent > 0, 'positive N_latent expected'
        assert len(resolution) == 3

        self.N_latent = N_latent
        """ (int) Number of fonts. """

        self.layers = []
        """ ([str]) Will hold layer names. """

        if architecture == 'standard':
            self.standard(resolution, **kwargs)
        elif architecture == 'mlp':
            self.mlp(resolution, **kwargs)
        elif architecture == 'dcgan':
            self.dcgan(resolution, **kwargs)
        elif architecture == 'pool':
            self.pool(resolution, **kwargs)
        else:
            raise NotImplementedError()

    def forward(self, codes):
        """
        Forward image, and output code and theta depending on what ahs been fixed.

        :param codes: codes
        :type codes: torch.autograd.Variable
        :return: output image
        :rtype: torch.autograd.Variable
        """

        output = codes
        for name in self.layers:
            output = getattr(self, name)(output)
        return output

    def __str__(self):
        """
        Print network.
        """

        string = ''
        for name in self.layers:
            string += '(' + name + ', ' + getattr(self, name).__class__.__name__ + ')\n'
        return string