import torch
from . import learned_decoder
from common import cuda


class SelectiveDecoder(torch.nn.Module):
    """
    Decoder consisting of multiple decoders per class.
    """

    def __init__(self, decoders, resolution):
        """
        Initialize decoder.

        :param decoders: list of decoders
        :type decoders: [LearnedDecoder]
        """

        super(SelectiveDecoder, self).__init__()

        assert isinstance(decoders, list)
        for decoder in decoders:
            assert isinstance(decoder, learned_decoder.LearnedDecoder)

        self.decoders = decoders
        """ ([LearnedDecoder]) Set of decoders per class. """

        # Update training or eval state!
        self.training = False
        for decoder in self.decoders:
            self.training = self.training and decoder.training
        if not self.training:
            for decoder in self.decoders:
                decoder.eval()

        self.resolution = resolution
        """ (int) Resolution for output. """

        self._code = None
        """ (None or torch.autograd.Variable) Fixed code if set. """

    def forward(self, input):
        """
        Wrapper forward function that also allows to call forward on only the code or theta
        after setting theta or the code fixed.

        The fixed one should not allow gradients.

        :param input: code or theta
        :type input: torch.autograd.Variable
        :return: output image
        :rtype: torch.autograd.Variable
        """

        assert self._code is not None

        use_gpu = cuda.is_cuda(self.decoders[0])
        output = torch.zeros([self._code.size()[0], self.resolution[0], self.resolution[1], self.resolution[2]])
        if use_gpu:
            output = output.cuda()
        output = torch.autograd.Variable(output)

        for c in range(len(self.decoders)):
            if torch.sum(self._code == c).item() > 0:
                input_ = input[self._code == c]
                output_ = self.decoders[c].forward(input_)
                # workaround for in-place assignments ...
                output[self._code == c] = output_

        return output

    def set_code(self, code):
        """
        Set fixed code to use in forward pass.

        :param code: code
        :type code: torch.autograd.Variable
        """

        self._code = code
