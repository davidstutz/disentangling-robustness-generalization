import torch
import common.torch


class LearnedVariationalEncoder(torch.nn.Module):
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

        layer = 0
        channels = []
        resolutions = []

        # Determine parameters of network
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
        dropout = kwargs.get('dropout', False)
        start_channels = kwargs.get('start_channels', 16)

        while True:
            input_channels = resolution[0] if layer == 0 else channels[layer - 1]
            output_channels = start_channels if layer == 0 else channels[layer - 1] * 2

            # Large kernel size was result of poor discriminator;
            # generator did only produce very "thin" EMNIST digits.
            conv = torch.nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1)
            # torch.nn.init.kaiming_normal_(conv.weight, gain)
            torch.nn.init.normal_(conv.weight, 0, 0.001)
            torch.nn.init.constant_(conv.bias, 0)
            self.add_layer('conv%d' % layer, conv)

            if batch_normalization:
                bn = torch.nn.BatchNorm2d(output_channels)
                torch.nn.init.constant_(bn.weight, 1)
                torch.nn.init.constant_(bn.bias, 0)
                self.add_layer('bn%d' % layer, bn)

            if activation:
                relu = activation(True)
                self.add_layer('act%d' % layer, relu)

            channels.append(output_channels)
            resolutions.append([
                resolution[1] // 2 if layer == 0 else resolutions[layer - 1][0] // 2,
                resolution[2] // 2 if layer == 0 else resolutions[layer - 1][1] // 2
            ])
            if resolutions[-1][0] // 2 < 2 or resolutions[-1][1] < 2:
                break;

            layer += 1

        # if dropout:
        #    drop = torch.nn.Dropout2d()
        #    self.add_layer('drop%d' % layer, drop)

        # Having strange bug where torch complaints about numpy.in64 being passed to nn.Linear.
        representation = int(resolutions[-1][0] * resolutions[-1][1] * channels[-1])
        view = common.torch.View(-1, representation)
        self.add_layer('view%d' % layer, view)

        self.mu_code = torch.nn.Linear(representation, self.N_latent)
        torch.nn.init.kaiming_normal_(self.mu_code.weight, gain)
        torch.nn.init.constant_(self.mu_code.bias, 0)

        self.logvar_code = torch.nn.Linear(representation, self.N_latent)
        torch.nn.init.kaiming_normal_(self.logvar_code.weight, gain)
        torch.nn.init.constant_(self.logvar_code.bias, 0)

        if self.number_flows > 0:
            self.u_code = torch.nn.Linear(representation, self.N_latent * self.number_flows)
            torch.nn.init.kaiming_normal_(self.u_code.weight, gain)
            torch.nn.init.constant_(self.u_code.bias, 0)

            self.w_code = torch.nn.Linear(representation, self.N_latent * self.number_flows)
            torch.nn.init.kaiming_normal_(self.w_code.weight, gain)
            torch.nn.init.constant_(self.w_code.bias, 0)

            self.b_code = torch.nn.Linear(representation, self.number_flows)
            torch.nn.init.kaiming_normal_(self.b_code.weight, gain)
            torch.nn.init.constant_(self.b_code.bias, 0)

            self.u_view = common.torch.View(-1, self.number_flows, self.N_latent)
            self.w_view = common.torch.View(-1, self.number_flows, self.N_latent)

    def pool(self, resolution, **kwargs):
        """
        Standard architeucture.

        :param resolution: input resolution (assumed square)
        :type resolution: int
        """

        layer = 0
        channels = []
        resolutions = []

        # Determine parameters of network
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

        while True:
            input_channels = resolution[0] if layer == 0 else channels[layer - 1]
            output_channels = start_channels if layer == 0 else channels[layer - 1] * 2

            conv = torch.nn.Conv2d(input_channels, output_channels, kernel_size=5, stride=1, padding=2)
            # torch.nn.init.kaiming_normal_(conv.weight, gain)
            torch.nn.init.normal_(conv.weight, 0, 0.001)
            torch.nn.init.constant_(conv.bias, 0)
            self.add_layer('conv%d' % layer, conv)

            if batch_normalization:
                bn = torch.nn.BatchNorm2d(output_channels)
                torch.nn.init.constant_(bn.weight, 1)
                torch.nn.init.constant_(bn.bias, 0)
                self.add_layer('bn%d' % layer, bn)

            if activation:
                relu = activation(True)
                self.add_layer('act%d' % layer, relu)

            pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.add_layer('pool%d' % layer, pool)

            channels.append(output_channels)
            resolutions.append([
                resolution[1] // 2 if layer == 0 else resolutions[layer - 1][0] // 2,
                resolution[2] // 2 if layer == 0 else resolutions[layer - 1][1] // 2,
            ])
            if resolutions[-1][0] // 2 < 3 or resolutions[-1][0] % 2 == 1 or resolutions[-1][1] // 2 < 3 or resolutions[-1][1] % 2 == 1:
                break;

            layer += 1

        representation = int(resolutions[-1][0] * resolutions[-1][1] * channels[-1])
        view = common.torch.View(-1, representation)
        self.add_layer('view%d' % layer, view)

        self.mu_code = torch.nn.Linear(representation, self.N_latent)
        torch.nn.init.kaiming_normal_(self.mu_code.weight, gain)
        torch.nn.init.constant_(self.mu_code.bias, 0)

        self.logvar_code = torch.nn.Linear(representation, self.N_latent)
        torch.nn.init.kaiming_normal_(self.logvar_code.weight, gain)
        torch.nn.init.constant_(self.logvar_code.bias, 0)

        if self.number_flows > 0:
            self.u_code = torch.nn.Linear(representation, self.N_latent * self.number_flows)
            torch.nn.init.kaiming_normal_(self.u_code.weight, gain)
            torch.nn.init.constant_(self.u_code.bias, 0)

            self.w_code = torch.nn.Linear(representation, self.N_latent * self.number_flows)
            torch.nn.init.kaiming_normal_(self.w_code.weight, gain)
            torch.nn.init.constant_(self.w_code.bias, 0)

            self.b_code = torch.nn.Linear(representation, self.number_flows)
            torch.nn.init.kaiming_normal_(self.b_code.weight, gain)
            torch.nn.init.constant_(self.b_code.bias, 0)

            self.u_view = common.torch.View(-1, self.number_flows, self.N_latent)
            self.w_view = common.torch.View(-1, self.number_flows, self.N_latent)

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
        view = common.torch.View(-1, units[0])
        self.add_layer('view0', view)

        for layer in range(1, len(units)):
            in_features = units[layer - 1]
            out_features = units[layer]

            lin = torch.nn.Linear(in_features=in_features, out_features=out_features)
            # torch.nn.init.kaiming_normal_(lin.weight, gain)
            torch.nn.init.normal_(lin.weight, 0, 1. / in_features)
            torch.nn.init.constant_(lin.bias, 0)
            self.add_layer('lin%d' % layer, lin)

            if activation:
                act = activation(True)
                self.add_layer('act%d' % layer, act)

            if batch_normalization:
                bn = torch.nn.BatchNorm1d(out_features)
                torch.nn.init.constant_(bn.weight, 1)
                torch.nn.init.constant_(bn.bias, 0)
                self.add_layer('bn%d' % layer, bn)

        representation = units[-1]
        view = common.torch.View(-1, representation)
        self.add_layer('view%d' % len(units), view)

        self.mu_code = torch.nn.Linear(representation, self.N_latent)
        torch.nn.init.kaiming_normal_(self.mu_code.weight, gain)
        torch.nn.init.constant_(self.mu_code.bias, 0)

        self.logvar_code = torch.nn.Linear(representation, self.N_latent)
        torch.nn.init.kaiming_normal_(self.logvar_code.weight, gain)
        torch.nn.init.constant_(self.logvar_code.bias, 0)

        if self.number_flows > 0:
            self.u_code = torch.nn.Linear(representation, self.N_latent * self.number_flows)
            torch.nn.init.kaiming_normal_(self.u_code.weight, gain)
            torch.nn.init.constant_(self.u_code.bias, 0)

            self.w_code = torch.nn.Linear(representation, self.N_latent * self.number_flows)
            torch.nn.init.kaiming_normal_(self.w_code.weight, gain)
            torch.nn.init.constant_(self.w_code.bias, 0)

            self.b_code = torch.nn.Linear(representation, self.number_flows)
            torch.nn.init.kaiming_normal_(self.b_code.weight, gain)
            torch.nn.init.constant_(self.b_code.bias, 0)

            self.u_view = common.torch.View(-1, self.number_flows, self.N_latent)
            self.w_view = common.torch.View(-1, self.number_flows, self.N_latent)

    def dcgan(self, resolution, **kwargs):
        """
        Standard architeucture.

        :param resolution: input resolution (assumed square)
        :type resolution: int
        """

        layer = 0
        channels = []
        resolutions = []

        gain = torch.nn.init.calculate_gain('relu')
        start_channels = kwargs.get('start_channels', 16)

        while True:
            input_channels = resolution[0] if layer == 0 else channels[layer - 1]
            output_channels = start_channels if layer == 0 else channels[layer - 1] * 2

            conv = torch.nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1)
            # torch.nn.init.kaiming_normal_(conv.weight, gain)
            torch.nn.init.normal_(conv.weight, 0, 0.001)
            torch.nn.init.constant_(conv.bias, 0)
            self.add_layer('conv%d' % layer, conv)

            bn = torch.nn.BatchNorm2d(output_channels)
            torch.nn.init.constant_(bn.weight, 1)
            torch.nn.init.constant_(bn.bias, 0)
            self.add_layer('bn%d' % layer, bn)

            relu = torch.nn.ReLU(True)
            self.add_layer('act%d' % layer, relu)

            channels.append(output_channels)
            resolutions.append([
                resolution[1] // 2 if layer == 0 else resolutions[layer - 1][0] // 2,
                resolution[2] // 2 if layer == 0 else resolutions[layer - 1][1] // 2,
            ])
            if resolutions[-1][0] // 2 < 3 or resolutions[-1][1] // 2 < 3:
                break;

            layer += 1

        representation = int(resolutions[-1][0] * resolutions[-1][1] * channels[-1])
        view = common.torch.View(-1, representation)
        self.add_layer('view%d' % layer, view)

        self.mu_code = torch.nn.Linear(representation, self.N_latent)
        torch.nn.init.kaiming_normal_(self.mu_code.weight, gain)
        torch.nn.init.constant_(self.mu_code.bias, 0)

        self.logvar_code = torch.nn.Linear(representation, self.N_latent)
        torch.nn.init.kaiming_normal_(self.logvar_code.weight, gain)
        torch.nn.init.constant_(self.logvar_code.bias, 0)

        if self.number_flows > 0:
            self.u_code = torch.nn.Linear(representation, self.N_latent * self.number_flows)
            torch.nn.init.kaiming_normal_(self.u_code.weight, gain)
            torch.nn.init.constant_(self.u_code.bias, 0)

            self.w_code = torch.nn.Linear(representation, self.N_latent * self.number_flows)
            torch.nn.init.kaiming_normal_(self.w_code.weight, gain)
            torch.nn.init.constant_(self.w_code.bias, 0)

            self.b_code = torch.nn.Linear(representation, self.number_flows)
            torch.nn.init.kaiming_normal_(self.b_code.weight, gain)
            torch.nn.init.constant_(self.b_code.bias, 0)

            self.u_view = common.torch.View(-1, self.number_flows, self.N_latent)
            self.w_view = common.torch.View(-1, self.number_flows, self.N_latent)

    def __init__(self, N_latent, number_flows=0, resolution=(1, 32, 32), architecture='standard', **kwargs):
        """
        Initialize encoder.

        :param N_latent: number of latent codes
        :type N_latent: int
        :param number_flows: number of flows
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param architecture: architecture builder to use
        :type architecture: str
        """

        super(LearnedVariationalEncoder, self).__init__()

        assert N_latent > 0, 'positive N_latent expected'
        assert len(resolution) == 3

        self.N_latent = N_latent
        """ (int) Number of fonts. """

        self.layers = []
        """ ([str]) Will hold layer names. """

        self.number_flows = number_flows
        """ (int) Number of flows. """

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

    def forward(self, image):
        """
        Forward image, and output code and theta depending on what ahs been fixed.

        :param image: theta or None
        :type image: torch.autograd.Variable or None
        :return: output image
        :rtype: torch.autograd.Variable
        """

        output = image
        for name in self.layers:
            output = getattr(self, name)(output)

        if self.number_flows > 0:
            return self.mu_code(output), \
                   self.logvar_code(output), \
                   self.w_view(self.w_code(output)), \
                   self.u_view(self.u_code(output)), \
                   self.b_code(output)
        else:
            return self.mu_code(output), \
                   self.logvar_code(output)

    def __str__(self):
        """
        Print network.
        """

        string = ''
        for name in self.layers:
            string += '(' + name + ', ' + getattr(self, name).__class__.__name__ + ')\n'
        string += '(mu, ' + self.mu_code.__class__.__name__ + ')\n'
        string += '(logvar, ' + self.logvar_code.__class__.__name__ + ')\n'
        if self.number_flows > 0:
            string += '(u, ' + self.u_code.__class__.__name__ + ')\n'
            string += '(w, ' + self.w_code.__class__.__name__ + ')\n'
            string += '(b, ' + self.b_code.__class__.__name__ + ')\n'
        return string