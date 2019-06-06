import os
import sys

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
import models
from common.log import log, Log
from common.state import State
from common import cuda
from common import paths
import common.numpy
import math
import numpy
import argparse


class ComputeLearnedDecoderPerturbations:
    """
    Test a trained classifier.
    """

    def __init__(self, args=None):
        """
        Initialize.

        :param args: optional arguments if not to use sys.argv
        :type args: [str]
        """

        self.args = None
        """ Arguments of program. """

        parser = self.get_parser()
        if args is not None:
            self.args = parser.parse_args(args)
        else:
            self.args = parser.parse_args()

        self.test_codes = None
        """ (numpy.ndarray) Font classes. """

        self.model = None
        """ (encoder.Encoder) Model to train. """

        self.perturbations = None
        """ (numpy.ndarray) Perturbations per test image. """

        self.perturbation_images = None
        """ (numpy.ndarray) Perturbation images. """

        if self.args.log_file:
            utils.makedir(os.path.dirname(self.args.log_file))
            Log.get_instance().attach(open(self.args.log_file, 'w'))

        log('-- ' + self.__class__.__name__)
        for key in vars(self.args):
            log('[Testing] %s=%s' % (key, str(getattr(self.args, key))))

    def __del__(self):
        """
        Remove log file.
        """

        if self.args is not None:
            if self.args.log_file:
                Log.get_instance().detach(self.args.log_file)

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Attack decoder and classifier.')
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file with test images.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing test codes.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-decoder_files', default=paths.state_file('decoder'), help='Decoder state file.', type=str)
        parser.add_argument('-perturbations_file', default=paths.results_file('decoder/perturbations'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-perturbation_images_file', default=paths.results_file('decoder/perturbation_images'), help='HDF5 file for perturbation images.', type=str)
        parser.add_argument('-log_file', default=paths.log_file('decoder/attacks'), help='Log file.', type=str)
        parser.add_argument('-batch_size', default=128, help='Batch size of attack.', type=int)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')

        # Some decoder parameters.
        parser.add_argument('-latent_space_size', default=10, help='Size of latent space.', type=int)
        parser.add_argument('-decoder_architecture', default='standard', help='Architecture to use.', type=str)
        parser.add_argument('-decoder_activation', default='relu', help='Activation function to use.', type=str)
        parser.add_argument('-decoder_no_batch_normalization', default=False, help='Do not use batch normalization.', action='store_true')
        parser.add_argument('-decoder_channels', default=16, help='Channels of first convolutional layer, afterwards channels are doubled.', type=int)
        parser.add_argument('-decoder_units', default='1024,1024,1024,1024', help='Units for MLP.')

        return parser

    def test(self):
        """
        Test classifier to identify valid samples to attack.
        """

        num_batches = int(math.ceil(self.perturbations.shape[0] / self.args.batch_size))

        for b in range(num_batches):
            b_start = b * self.args.batch_size
            b_end = min((b + 1) * self.args.batch_size, self.perturbations.shape[0])
            batch_classes = common.torch.as_variable(self.test_codes[b_start: b_end], self.args.use_gpu)
            batch_inputs = common.torch.as_variable(self.perturbations[b_start: b_end], self.args.use_gpu)

            self.model.set_code(batch_classes)
            output_images = self.model(batch_inputs)

            output_images = numpy.squeeze(numpy.transpose(output_images.cpu().detach().numpy(), (0, 2, 3, 1)))
            self.perturbation_images = common.numpy.concatenate(self.perturbation_images, output_images)

            if b % 100 == 0:
                log('[Testing] computing perturbation images %d' % b)

        utils.makedir(os.path.dirname(self.args.perturbation_images_file))
        if len(self.perturbation_images.shape) > 3:
            self.perturbation_images = self.perturbation_images.reshape(self.N_samples, self.N_attempts, self.perturbation_images.shape[1], self.perturbation_images.shape[2], self.perturbation_images.shape[3])
        else:
            self.perturbation_images = self.perturbation_images.reshape(self.N_samples, self.N_attempts, self.perturbation_images.shape[1], self.perturbation_images.shape[2])
        self.perturbation_images = numpy.swapaxes(self.perturbation_images, 0, 1)
        utils.write_hdf5(self.args.perturbation_images_file, self.perturbation_images)
        log('[Testing] wrote %s' % self.args.perturbation_images_file)

    def load_model(self):
        """
        Load model.
        """

        assert self.args.decoder_files
        decoder_files = self.args.decoder_files.split(',')
        for decoder_file in decoder_files:
            assert os.path.exists(decoder_file), 'could not find %s' % decoder_file

        decoder_units = list(map(int, self.args.decoder_units.split(',')))
        log('[Testing] using %d input channels' % self.image_channels)

        if len(decoder_files) > 1:
            log('[Testing] loading multiple decoders')
            decoders = []
            for i in range(len(decoder_files)):
                decoder = models.LearnedDecoder(self.args.latent_space_size, resolution=(self.image_channels, self.resolution, self.resolution),
                                                architecture=self.args.decoder_architecture,
                                                start_channels=self.args.decoder_channels,
                                                activation=self.args.decoder_activation,
                                                batch_normalization=not self.args.decoder_no_batch_normalization,
                                                units=decoder_units)

                state = State.load(decoder_files[i])
                decoder.load_state_dict(state.model)
                if self.args.use_gpu and not cuda.is_cuda(decoder):
                    decoder = decoder.cuda()
                decoders.append(decoder)

                decoder.eval()
                log('[Testing] loaded %s' % decoder_files[i])
            decoder = models.SelectiveDecoder(decoders, resolution=(self.image_channels, self.resolution, self.resolution))
        else:
            log('[Testing] loading one decoder')
            decoder = models.LearnedDecoder(self.args.latent_space_size, resolution=(self.image_channels, self.resolution, self.resolution),
                                            architecture=self.args.decoder_architecture,
                                            start_channels=self.args.decoder_channels,
                                            activation=self.args.decoder_activation,
                                            batch_normalization=not self.args.decoder_no_batch_normalization,
                                            units=decoder_units)

            state = State.load(decoder_files[0])
            decoder.load_state_dict(state.model)
            if self.args.use_gpu and not cuda.is_cuda(decoder):
                decoder = decoder.cuda()
            decoder.eval()
            log('[Testing] read decoder')

        self.model = decoder

    def load_data(self):
        """
        Load data.
        """

        test_images = utils.read_hdf5(self.args.test_images_file)
        log('[Testing] read %s' % self.args.test_images_file)
        self.resolution = test_images.shape[1]
        self.image_channels = 1 if len(test_images.shape) == 3 else 3

        self.test_codes = utils.read_hdf5(self.args.test_codes_file)
        self.test_codes = self.test_codes[:, self.args.label_index]
        log('[Testing] read %s' % self.args.test_codes_file)

        self.perturbations = utils.read_hdf5(self.args.perturbations_file).astype(numpy.float32)
        self.N_attempts = self.perturbations.shape[0]
        self.N_samples = self.perturbations.shape[1]
        self.perturbations = numpy.swapaxes(self.perturbations, 0, 1)
        self.perturbations = self.perturbations.reshape((self.perturbations.shape[0] * self.perturbations.shape[1], -1))
        log('[Testing] read %s' % self.args.perturbations_file)

        self.test_codes = numpy.repeat(self.test_codes[:self.N_samples], self.N_attempts, axis=0)

    def main(self):
        """
        Main.
        """

        self.load_data()
        self.load_model()
        self.test()


if __name__ == '__main__':
    program = ComputeLearnedDecoderPerturbations()
    program.main()