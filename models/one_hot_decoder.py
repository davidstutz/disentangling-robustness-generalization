import torch
import common.cuda
from .decoder import Decoder


class OneHotDecoder(Decoder):
    """
    The decoder computes, given the font/letter combination (as one/hot vector) and affine transformation parameters,
    the resulting images bz first selecting the font/letter from a database and then applying the transformation
    through a spatial transformer network.
    """

    def __init__(self, database, N_theta, softmax=False, temperature=1):
        """
        Initialize decoder.

        :param database: database of prototype images
        :type database: torch.autograd.Variable
        :param N_theta: number of transformation parameters
        :type N_theta: int
        :param softmax: whether to apply softmax before working with code
        :type softmax: bool
        :param temperature: temperature of softmax
        :type temperature: float
        """

        super(OneHotDecoder, self).__init__()

        assert len(database.size()) == 3, 'database has to be of rank 3'

        self.database = database
        self.database.unsqueeze_(0)
        """ (torch.autograd.Variable) Database of protoype images. """

        self.N_theta = N_theta
        """ (int) Number of transformation parameters. """

        if softmax:
            self.softmax = torch.nn.Softmax(dim=1)
            """ (torch.nn.Softmax) Whether to apply softmax. """

            self.temperature = temperature
            """ (float) Termperature of softmax. """
        else:
            self.softmax = softmax
            """ (bool) Not to use softmax. """

        # https://github.com/pytorch/pytorch/issues/4632
        #torch.backends.cudnn.benchmark = False

    def stn(self, theta, input):
        """
        Apply spatial transformer network on image using affine transformation theta.

        :param theta: transformation parameters as 6-vector
        :type theta: torch.autograd.Variable
        :param input: image(s) to apply transformation to
        :type input: torch.autograd.Variable
        :return: transformed image(s)
        :rtype: torch.autograd.Variable
        """

        # theta is given as 6-vector
        theta = theta.view(-1, 2, 3)

        grid = torch.nn.functional.affine_grid(theta, input.size())
        output = torch.nn.functional.grid_sample(input, grid)

        return output

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

        # see broadcasting: http://pytorch.org/docs/master/notes/broadcasting.html
        # first computed weighted sum of database with relevant part of code
        code = code.view(code.size()[0], code.size()[1], 1, 1)

        if self.softmax:
            code = self.softmax(code/self.temperature)

        image = torch.sum(torch.mul(self.database, code), 1) # 1 x N x H x W and B x N x 1 x 1
        image.unsqueeze_(1)

        if self.N_theta > 0:
            translation_x = theta[:, 0]
        if self.N_theta > 1:
            translation_y = theta[:, 1]
        if self.N_theta > 2:
            shear_x = theta[:, 2]
        if self.N_theta > 3:
            shear_y = theta[:, 3]
        if self.N_theta > 4:
            scales = theta[:, 4]
        if self.N_theta > 5:
            rotation = theta[:, 5]

        transformation = torch.autograd.Variable(torch.FloatTensor(theta.size()[0], 6).fill_(0))
        if common.cuda.is_cuda(theta):
            transformation = transformation.cuda()

        # translation x
        if self.N_theta == 1:
            transformation[:, 0] = 1
            transformation[:, 4] = 1
            transformation[:, 2] = translation_x
        # translation y
        elif self.N_theta == 2:
            transformation[:, 0] = 1
            transformation[:, 4] = 1
            transformation[:, 2] = translation_x
            transformation[:, 5] = translation_y
        # shear x
        elif self.N_theta == 3:
            transformation[:, 0] = 1
            transformation[:, 1] = shear_x
            transformation[:, 2] = translation_x
            transformation[:, 4] = 1
            transformation[:, 5] = translation_y
        # shear y
        elif self.N_theta == 4:
            transformation[:, 0] = 1
            transformation[:, 1] = shear_x
            transformation[:, 2] = translation_x
            transformation[:, 3] = shear_y
            transformation[:, 4] = 1
            transformation[:, 5] = translation_y
        # scale
        elif self.N_theta == 5:
            transformation[:, 0] = scales
            transformation[:, 1] = scales * shear_x
            transformation[:, 2] = translation_x
            transformation[:, 3] = scales * shear_y
            transformation[:, 4] = scales
            transformation[:, 5] = translation_y
        # rotation
        elif self.N_theta >= 6 and self.N_theta <= 9:
            transformation[:, 0] = torch.cos(rotation) * scales - torch.sin(rotation) * scales * shear_x
            transformation[:, 1] = -torch.sin(rotation) * scales + torch.cos(rotation) * scales * shear_x
            transformation[:, 2] = translation_x
            transformation[:, 3] = torch.cos(rotation) * scales * shear_y + torch.sin(rotation) * scales
            transformation[:, 4] = -torch.sin(rotation) * scales * shear_y + torch.cos(rotation) * scales
            transformation[:, 5] = translation_y
        else:
            raise NotImplementedError()

        output = torch.clamp(torch.clamp(self.stn(transformation, image), min=0), max=1)

        if self.N_theta == 7:
            output = torch.mul(output, theta[:, 6].view(-1, 1, 1, 1))
        elif self.N_theta == 8:
            r = torch.mul(output, theta[:, 6].view(-1, 1, 1, 1))
            g = torch.mul(output, theta[:, 7].view(-1, 1, 1, 1))
            b = output
            output = 1 - torch.cat((
                r,
                g,
                b
            ), dim=1)
        elif self.N_theta == 9:
            # view is important as reshape does not allow grads in the cat case
            r = torch.mul(output, theta[:, 6].view(-1, 1, 1, 1))
            g = torch.mul(output, theta[:, 7].view(-1, 1, 1, 1))
            b = torch.mul(output, theta[:, 8].view(-1, 1, 1, 1))
            output = 1 - torch.cat((
                r,
                g,
                b
            ), dim=1)

        return output
