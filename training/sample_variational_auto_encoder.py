import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
import models
from common.log import log, Log
from common.state import State
from common import cuda
from common import paths
import common.torch
import common.numpy
import numpy
import argparse
import math


class SampleVariationalAutoEncoder:
    """
    Sampled decoder.
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

        log('-- ' + self.__class__.__name__)
        for key in vars(self.args):
            log('[Sampling] %s=%s' % (key, str(getattr(self.args, key))))

        self.decoder = None
        """ (models.LearnedDecoder) Decoder. """

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Sample auto encoder.')
        parser.add_argument('-decoder_file', default=paths.state_file('decoder'), help='Snapshot state file.', type=str)
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing test images.', type=str)
        parser.add_argument('-images_file', default=paths.results_file('sampled_images'), help='HDF5 file for sampled test images.', type=str)
        parser.add_argument('-theta_file', default=paths.results_file('sampled_theta'), help='HDF5 file for sampled train theta.', type=str)
        parser.add_argument('-N_samples', default=40000, help='Number of samples.', type=int)
        parser.add_argument('-bound', default=2, help='Truncated normal bound.', type=float)
        parser.add_argument('-batch_size', default=128, help='Batch size.', type=int)
        parser.add_argument('-latent_space_size', default=10, help='Size of latent space.', type=int)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')

        # Some network parameters.
        parser.add_argument('-network_architecture', default='standard', help='Classifier architecture to use.', type=str)
        parser.add_argument('-network_activation', default='relu', help='Activation function to use.', type=str)
        parser.add_argument('-network_no_batch_normalization', default=False, help='Do not use batch normalization.', action='store_true')
        parser.add_argument('-network_channels', default=16, help='Channels of first convolutional layer, afterwards channels are doubled.', type=int)
        parser.add_argument('-network_units', default='1024,1024,1024,1024', help='Units for MLP.')

        return parser

    def sample(self):
        """
        Test the model.
        """

        assert self.decoder is not None

        self.decoder.eval()
        log('[Sampling] set decoder to eval')

        images = None

        theta = common.numpy.truncated_normal((self.args.N_samples, self.args.latent_space_size), lower=-self.args.bound, upper=self.args.bound).astype(numpy.float32)
        theta = theta.astype(numpy.float32)
        num_batches = int(math.ceil(theta.shape[0]/self.args.batch_size))
        
        for b in range(num_batches):
            b_start = b*self.args.batch_size
            b_end = min((b + 1)*self.args.batch_size, theta.shape[0])

            batch_theta = common.torch.as_variable(theta[b_start: b_end], self.args.use_gpu)

            # Important to get the correct codes!
            assert self.decoder.training is False
            output_images = self.decoder(batch_theta)

            output_images = numpy.squeeze(numpy.transpose(output_images.cpu().detach().numpy(), (0, 2, 3, 1)))
            images = common.numpy.concatenate(images, output_images)

            if b%100 == 50:
                log('[Sampling] %d' % b)

        if self.args.images_file:
            utils.write_hdf5(self.args.images_file, images)
            log('[Sampling] wrote %s' % self.args.images_file)

        if self.args.theta_file:
            utils.write_hdf5(self.args.theta_file, theta)
            log('[Sampling] wrote %s' % self.args.theta_file)

    def main(self):
        """
        Main which should be overwritten.
        """

        test_images = utils.read_hdf5(self.args.test_images_file)
        log('[Sampling] read %s' % self.args.test_images_file)

        if len(test_images.shape) < 4:
            test_images = numpy.expand_dims(test_images, axis=3)

        network_units = list(map(int, self.args.network_units.split(',')))
        self.decoder = models.LearnedDecoder(self.args.latent_space_size, resolution=(test_images.shape[3], test_images.shape[1], test_images.shape[2]),
                                             architecture=self.args.network_architecture,
                                             start_channels=self.args.network_channels,
                                             activation=self.args.network_activation,
                                             batch_normalization=not self.args.network_no_batch_normalization,
                                             units=network_units)
        log(self.decoder)

        assert os.path.exists(self.args.decoder_file)
        state = State.load(self.args.decoder_file)
        log('[Sampling] loaded %s' % self.args.decoder_file)

        self.decoder.load_state_dict(state.model)
        log('[Sampling] loaded decoder')

        if self.args.use_gpu and not cuda.is_cuda(self.decoder):
            self.decoder = self.decoder.cuda()

        log('[Sampling] model needs %gMiB' % ((cuda.estimate_size(self.decoder))/ (1024 * 1024)))
        self.sample()


if __name__ == '__main__':
    program = SampleVariationalAutoEncoder()
    program.main()