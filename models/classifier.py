import torch
import common.torch


class _ResNetBlock(torch.nn.Module):
    """
    Taken from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        def conv3x3(in_planes, out_planes, stride=1):
            """
            3x3 convolution with padding.
            """

            return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        super(_ResNetBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.bn1 = torch.nn.BatchNorm2d(planes)
        torch.nn.init.constant_(self.bn1.weight, 1)
        torch.nn.init.constant_(self.bn1.bias, 0)

        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

        self.bn2 = torch.nn.BatchNorm2d(planes)
        torch.nn.init.constant_(self.bn2.weight, 1)
        torch.nn.init.constant_(self.bn2.bias, 0)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Classifier(torch.nn.Module):
    """
    Simple classifier.
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

    def resnet(self, resolution, **kwargs):
        """
        Resnet model.
        """

        layers = kwargs.get('units', [3, 3, 3]) # default resnet18

        start_channels = kwargs.get('start_channels', 64)
        self.inplanes = start_channels

        conv1 = torch.nn.Conv2d(resolution[0], self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
        self.add_layer('conv1', conv1)

        bn1 = torch.nn.BatchNorm2d(self.inplanes)
        torch.nn.init.constant_(bn1.weight, 1)
        torch.nn.init.constant_(bn1.bias, 0)
        self.add_layer('bn1', bn1)

        relu = torch.nn.ReLU(inplace=True)
        self.add_layer('relu1', relu)

        #pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.add_layer('pool1', pool)

        layer1 = self.resnet_block(start_channels, layers[0])
        self.add_layer('block1', layer1)

        layer2 = self.resnet_block(2*start_channels, layers[1], stride=2)
        self.add_layer('block2', layer2)

        layer3 = self.resnet_block(4*start_channels, layers[2], stride=2)
        self.add_layer('block3', layer3)

        #layer4 = self.resnet_block(8*start_channels, layers[3], stride=2)
        #self.add_layer('block4', layer4)

        representation = 4*start_channels
        pool = torch.nn.AvgPool2d((resolution[1]//4, resolution[2]//4), stride=1)
        self.add_layer('avgpool', pool)

        view = common.torch.View(-1, representation)
        self.add_layer('view', view)

        #representation = resolution[1] // 8 * resolution[2] // 8 * 8 * self.inplanes
        #view = common.torch.View(-1, representation)
        #self.add_layer('view', view)

        gain = torch.nn.init.calculate_gain('relu')
        logits = torch.nn.Linear(representation, self.N_class)
        torch.nn.init.kaiming_normal_(logits.weight, gain)
        torch.nn.init.constant_(logits.bias, 0)
        self.add_layer('logits', logits)

    def resnet_block(self, planes, blocks, stride=1):
        """
        Resnet block.
        """

        def conv1x1(in_planes, out_planes, stride=1):
            """
            1x1 convolution.
            """

            return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

        downsample = None
        if stride != 1 or self.inplanes != planes:
            conv = conv1x1(self.inplanes, planes, stride)
            torch.nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')

            bn = torch.nn.BatchNorm2d(planes)
            torch.nn.init.constant_(bn.weight, 1)
            torch.nn.init.constant_(bn.bias, 0)

            downsample = torch.nn.Sequential(*[conv, bn])

        layers = []
        layers.append(_ResNetBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_ResNetBlock(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def vgg(self, resolution, **kwargs):
        """
        VGG architecture.

        :param resolution: input resolution (assumed square)
        :type resolution: int
        """

        configurations = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

        dropout = kwargs.get('dropout', False)
        batch_normalization = kwargs.get('batch_normalization', True)
        channels = kwargs.get('units', [64, 128, 256, 256])

        in_channels = resolution[0]
        factor = 1

        for i in range(len(channels)):
            if i > 0 and channels[i] != channels[i - 1]:
                pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
                factor *= 2
                self.add_layer('pool%d' % i, pool)

            conv = torch.nn.Conv2d(in_channels, channels[i], kernel_size=3, padding=1)
            torch.nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(conv.bias, 0)
            self.add_layer('conv%d' % i, conv)

            if batch_normalization:
                bn = torch.nn.BatchNorm2d(channels[i])
                torch.nn.init.constant_(bn.weight, 1)
                torch.nn.init.constant_(bn.bias, 0)
                self.add_layer('bn%d' % i, bn)

            relu = torch.nn.ReLU(inplace=True)
            self.add_layer('relu%d' % i, relu)

            in_channels = channels[i]

        representation = channels[-1] * resolution[1]//factor* resolution[2]//factor
        units = 4*channels[-1]
        view = common.torch.View(-1, representation)
        self.add_layer('view%d' % i, view)

        linear = torch.nn.Linear(representation, units)
        torch.nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
        torch.nn.init.constant_(linear.bias, 0)
        self.add_layer('linear%d' % i, linear)

        relu = torch.nn.ReLU(True)
        self.add_layer('relu%d' % i, relu)
        if dropout:
            drop = torch.nn.Dropout()
            self.add_layer('drop%d' % i, drop)

        i += 1
        linear = torch.nn.Linear(units, units)
        torch.nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
        torch.nn.init.constant_(linear.bias, 0)
        self.add_layer('linear%d' % i, linear)

        relu = torch.nn.ReLU(True)
        self.add_layer('relu%d' % i, relu)
        if dropout:
            drop = torch.nn.Dropout()
            self.add_layer('drop%d' % i, drop)

        logits = torch.nn.Linear(units, self.N_class)
        torch.nn.init.kaiming_normal_(logits.weight, nonlinearity='relu')
        torch.nn.init.constant_(logits.bias, 0)
        self.add_layer('logits', logits)

    def simple(self, resolution, **kwargs):
        """
        Standard architeucture.

        :param resolution: input resolution (assumed square)
        :type resolution: int
        """

        assert len(resolution) == 3

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
        kernel_size = kwargs.get('kernel_size', 4)
        assert kernel_size%2 == 0

        while True:
            input_channels = resolution[0] if layer == 0 else channels[layer - 1]
            output_channels = start_channels if layer == 0 else channels[layer - 1]*2

            # Large kernel size was result of poor discriminator;
            # generator did only produce very "thin" EMNIST digits.
            conv = torch.nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=2, padding=kernel_size//2-1)
            torch.nn.init.kaiming_normal_(conv.weight, gain)
            #torch.nn.init.normal_(conv.weight, 0, 0.001)
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
                break

            layer += 1

        #if dropout:
        #    drop = torch.nn.Dropout2d()
        #    self.add_layer('drop%d' % layer, drop)

        # Having strange bug where torch complaints about numpy.in64 being passed to nn.Linear.
        representation = int(resolutions[-1][0]*resolutions[-1][1]*channels[-1])
        view = common.torch.View(-1, representation)
        self.add_layer('view', view)

        linear1 = torch.nn.Linear(representation, 10*self.N_class)
        bn1 = torch.nn.BatchNorm1d(10*self.N_class)
        relu1 = activation(True)
        linear2 = torch.nn.Linear(10*self.N_class, self.N_class)

        torch.nn.init.kaiming_normal_(linear1.weight, gain)
        #torch.nn.init.normal_(linear1.weight, 0, 0.001)
        torch.nn.init.constant_(linear1.bias, 0)
        torch.nn.init.kaiming_normal_(linear2.weight, gain)
        #torch.nn.init.normal_(linear2.weight, 0, 0.001)
        torch.nn.init.constant_(linear2.bias, 0)

        self.add_layer('representation1', linear1)
        self.add_layer('representation2', bn1)
        self.add_layer('representation', relu1)

        if dropout:
            drop = torch.nn.Dropout()
            self.add_layer('drop%d' % layer, drop)
        self.add_layer('logits', linear2)

    def standard(self, resolution, **kwargs):
        """
        Standard architeucture.

        :param resolution: input resolution (assumed square)
        :type resolution: int
        """

        assert len(resolution) == 3

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
        kernel_size = kwargs.get('kernel_size', 4)
        assert kernel_size%2 == 0

        while True:
            input_channels = resolution[0] if layer == 0 else channels[layer - 1]
            output_channels = start_channels if layer == 0 else channels[layer - 1]*2

            # Large kernel size was result of poor discriminator;
            # generator did only produce very "thin" EMNIST digits.
            conv = torch.nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=2, padding=kernel_size//2-1)
            torch.nn.init.kaiming_normal_(conv.weight, gain)
            #torch.nn.init.normal_(conv.weight, 0, 0.001)
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

        #if dropout:
        #    drop = torch.nn.Dropout2d()
        #    self.add_layer('drop%d' % layer, drop)

        # Having strange bug where torch complaints about numpy.in64 being passed to nn.Linear.
        representation = int(resolutions[-1][0]*resolutions[-1][1]*channels[-1])
        view = common.torch.View(-1, representation)
        self.add_layer('view', view)

        linear1 = torch.nn.Linear(representation, 10*self.N_class)
        linear2 = torch.nn.Linear(10*self.N_class, self.N_class)

        torch.nn.init.kaiming_normal_(linear1.weight, gain)
        #torch.nn.init.normal_(linear1.weight, 0, 0.001)
        torch.nn.init.constant_(linear1.bias, 0)
        torch.nn.init.kaiming_normal_(linear2.weight, gain)
        #torch.nn.init.normal_(linear2.weight, 0, 0.001)
        torch.nn.init.constant_(linear2.bias, 0)

        self.add_layer('representation', linear1)
        if dropout:
            drop = torch.nn.Dropout()
            self.add_layer('drop%d' % layer, drop)
        self.add_layer('logits', linear2)

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
        dropout = kwargs.get('dropout', False)
        units = kwargs.get('units', [256, 256, 256])

        from operator import mul
        from functools import reduce
        units = [reduce(mul, resolution, 1)] + units
        view = common.torch.View(-1, units[0])
        self.add_layer('view0', view)

        for layer in range(1, len(units)):
            in_features = units[layer - 1]
            out_features = units[layer]

            lin = torch.nn.Linear(in_features=in_features, out_features=out_features)
            torch.nn.init.kaiming_normal_(lin.weight, gain)
            #torch.nn.init.normal_(lin.weight, 0, 0.001)
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

        if dropout:
            drop = torch.nn.Dropout2d()
            self.add_layer('drop%d' % len(units), drop)

        logits = torch.nn.Linear(units[-1], self.N_class)
        torch.nn.init.kaiming_normal_(logits.weight, gain)
        #torch.nn.init.normal_(logits.weight, 0, 0.001)
        torch.nn.init.constant_(logits.bias, 0)
        self.add_layer('logits', logits)

    def linear(self, resolution, **kwargs):
        """
        MLP architecture.

        :param resolution: input resolution (assumed square)
        :type resolution: int
        """

        from operator import mul
        from functools import reduce
        dim = reduce(mul, resolution, 1)
        view = common.torch.View(-1, dim)
        self.add_layer('view0', view)
        logits = torch.nn.Linear(dim, self.N_class)
        torch.nn.init.kaiming_normal_(logits.weight)
        #torch.nn.init.normal_(logits.weight, 0, 0.001)
        torch.nn.init.constant_(logits.bias, 0)
        self.add_layer('logits', logits)

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
            torch.nn.init.kaiming_normal_(conv.weight, gain)
            #torch.nn.init.normal_(conv.weight, 0, 0.001)
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
        self.add_layer('view', view)

        logits = torch.nn.Linear(representation, self.N_class)
        torch.nn.init.kaiming_normal_(logits.weight, gain)
        #torch.nn.init.normal_(logits.weight, 0, 0.001)
        torch.nn.init.constant_(logits.bias, 0)
        self.add_layer('logits', logits)

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
            torch.nn.init.kaiming_normal_(conv.weight, gain)
            #torch.nn.init.normal_(conv.weight, 0, 0.001)
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
        self.add_layer('view', view)

        logits = torch.nn.Linear(representation, self.N_class)
        torch.nn.init.kaiming_normal_(logits.weight, gain)
        #torch.nn.init.normal_(logits.weight, 0, 0.001)
        torch.nn.init.constant_(logits.bias, 0)
        self.add_layer('logits', logits)

    def __init__(self, N_class, resolution=(1, 32, 32), architecture='standard', **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param architecture: architecture builder to use
        :type architecture: str
        """

        super(Classifier, self).__init__()

        assert N_class > 0, 'positive N_class expected'
        assert len(resolution) <= 3
        resolution = list(resolution)

        self.N_class = int(N_class) # Having strange bug where torch complaints about numpy.in64 being passed to nn.Linear.
        """ (int) Number of classes. """

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
        elif architecture == 'linear':
            self.linear(resolution, **kwargs)
        elif architecture == 'simple':
            self.simple(resolution, **kwargs)
        elif architecture == 'resnet':
            self.resnet(resolution, **kwargs)
        elif architecture == 'vgg':
            self.vgg(resolution, **kwargs)
        else:
            raise NotImplementedError()

    def forward(self, image, return_features=False):
        """
        Forward pass, takes an image and outputs the predictions.

        :param image: input image
        :type image: torch.autograd.Variable
        :param return_representation: whether to also return representation layer
        :type return_representation: bool
        :return: logits
        :rtype: torch.autograd.Variable
        """

        features = []
        output = image

        for name in self.layers:
            output = getattr(self, name)(output)
            features.append(output)
        if return_features:
            return output, features
        else:
            return output

    def __str__(self):
        """
        Print network.
        """

        string = ''
        for name in self.layers:
            string += '(' + name + ', ' + getattr(self, name).__class__.__name__ + ')\n'
            if type(getattr(self, name)).__class__.__name__ == 'Sequential':
                for module in getattr(self, name).modules():
                    string += '\t(' + module.__class__.__name__ + ')'
        return string

