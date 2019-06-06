import torch
from .encoder import Encoder
from .decoder import Decoder


class AutoEncoder(torch.nn.Module):
    """
    Auto encoder.
    """

    def __init__(self, encoder, decoder):
        """
        Initialize encoder.

        :param N_font: length of first one-hot vector
        :type N_font: int
        :param N_class: length of second one-hot vector
        :type N_class: int
        """

        super(AutoEncoder, self).__init__()

        assert isinstance(encoder, Encoder), 'encoder has to be of class Encoder'
        assert isinstance(decoder, Decoder), 'decoder has to be of class Decoder'

        self.encoder = encoder
        """ (Encoder) Encoder. """

        self.decoder = decoder
        """ (Decoder) Decoder. """

    def forward(self, image):
        """
        Forward pass, takes a code(s) and generates the corresponding image(s).

        :param image: input image
        :type image: torch.autograd.Variable
        :return: output image
        :rtype: (torch.autograd.Variable, torch.autograd.Variable)
        """

        code, theta = self.encoder(image)
        reconstruction = self.decoder(code, theta)

        return reconstruction, code, theta

    def eval(self):
        """
        Just for saying that eval needs to be called on encoder and decoder explicitly.
        """

        raise NotImplementedError('eval needs to be called on encoder and decoder separately!')