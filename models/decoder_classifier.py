import torch
from common import cuda
from .decoder import Decoder
from .classifier import Classifier
from .learned_decoder import LearnedDecoder
from .selective_decoder import SelectiveDecoder
from .stn_decoder import STNDecoder


class DecoderClassifier(torch.nn.Module):
    """
    Combine decoder and classifier in single model for attacks.
    """

    def __init__(self, decoder, classifier):
        """
        Constructor.

        :param decoder: decoder
        :type decoder: torch.nn.Module
        :param classifier: classifier
        :type classifier: torch.nn.Module
        """

        assert isinstance(decoder, Decoder) or isinstance(decoder, LearnedDecoder) or isinstance(decoder, SelectiveDecoder) or isinstance(decoder, STNDecoder)
        assert isinstance(classifier, Classifier)

        super(DecoderClassifier, self).__init__()
        assert cuda.is_cuda(decoder) == cuda.is_cuda(classifier), 'decoder and classifier have to be both cuda or not'

        self.decoder = decoder
        """ (torch.nn.Module) Decoder. """

        self.classifier = classifier
        """ (torch.nn.Module) Classifier. """

    def forward(self, codes):
        """
        Forward pass through both models.

        :param codes: latent codes
        :type codes: torch.autograd.Variable
        :return: classification
        :rtype: torch.autograd.Variable
        """

        images = self.decoder.forward(codes)
        return self.classifier.forward(images)

    def eval(self):
        """
        Just for saying that eval needs to be called on encoder and decoder explicitly.
        """

        raise NotImplementedError('eval needs to be called on decoder and classifier separately!')