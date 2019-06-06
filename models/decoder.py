import torch


class Decoder(torch.nn.Module):
    """
    The decoder computes, given the font/letter combination (as one/hot vector) and affine transformation parameters,
    the resulting images bz first selecting the font/letter from a database and then applying the transformation
    through a spatial transformer network.
    """

    def __init__(self):
        """
        Initialize decoder.

        :param database: database of prototype images
        :type database: torch.autograd.Variable
        """

        super(Decoder, self).__init__()

        self._theta = None
        """ (None or torch.autograd.Variable) Fixed theta if set. """

        self._code = None
        """ (None or torch.autograd.Variable) Fixed code if set. """

    def forward(self, code_or_theta, theta=None):
        """
        Wrapper forward function that also allows to call forward on only the code or theta
        after setting theta or the code fixed.

        The fixed one should not allow gradients.

        :param code_or_theta: code or theta
        :type code_or_theta: torch.autograd.Variable
        :param theta: theta or None
        :type theta: torch.autograd.Variable or None
        :return: output image
        :rtype: torch.autograd.Variable
        """

        # Cases:
        # 1/ fixed theta
        # 2/ fixed code
        # 3/ both code and theta give
        assert theta is None and self.set_theta is not None \
            or theta is None and self.set_code is not None \
            or theta is not None

        if theta is None:
            if self._theta is not None:
                return self._forward(code_or_theta, self._theta)
            elif self._code is not None:
                return self._forward(self._code, code_or_theta)
        else:
            return self._forward(code_or_theta, theta)

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
