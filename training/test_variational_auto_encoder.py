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
import scipy.interpolate
import torch
import numpy
import argparse
import math
if utils.display():
    from common import plot


class TestVariationalAutoEncoder:
    """
    Train an encoder on transformed letters.
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

        self.train_images = None
        """ (numpy.ndarray) Images to train on. """

        self.test_images = None
        """ (numpy.ndarray) Images to test on. """

        self.train_codes = None
        """ (numpy.ndarray) Labels to train on. """

        self.test_codes = None
        """ (numpy.ndarray) Labels to test on. """

        self.resolution = None
        """ (int) Resolution. """

        self.encoder = None
        """ (models.LearnedEncoder) Encoder. """

        self.decoder = None
        """ (models.LearnedDecoder) Decoder. """

        self.reconstruction_error = 0
        """ (int) Reconstruction error. """

        self.code_mean = 0
        """ (int) Reconstruction error. """

        self.code_var = 0
        """ (int) Reconstruction error. """

        self.pred_images = None
        """ (numpy.ndarray) Test images reconstructed. """

        self.pred_codes = None
        """ (numpy.ndarray) Test latent codes. """

        self.results = dict()
        """ (dict) Results. """

        utils.makedir(os.path.dirname(self.args.log_file))
        if self.args.log_file:
            Log.get_instance().attach(open(self.args.log_file, 'w'))

        log('-- ' + self.__class__.__name__)
        for key in vars(self.args):
            log('[Testing] %s=%s' % (key, str(getattr(self.args, key))))

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Test auto encoder.')
        parser.add_argument('-train_images_file', default=paths.train_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-train_codes_file', default=paths.train_codes_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-train_theta_file', default=paths.results_file('train_theta'), help='HDF5 file for codes.', type=str)
        parser.add_argument('-test_theta_file', default=paths.results_file('test_theta'), help='HDF5 file for codes.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-label', default=-1, help='Label to constrain to.', type=int)
        parser.add_argument('-encoder_file', default=paths.state_file('encoder'), help='Snapshot state file.', type=str)
        parser.add_argument('-decoder_file', default=paths.state_file('decoder'), help='Snapshot state file.', type=str)
        parser.add_argument('-reconstruction_file', default=paths.results_file('reconstructions'), help='Reconstructions file.', type=str)
        parser.add_argument('-train_reconstruction_file', default='', help='Reconstructions file.', type=str)
        parser.add_argument('-random_file', default=paths.results_file('random'), help='Reconstructions file.', type=str)
        parser.add_argument('-interpolation_file', default=paths.results_file('interpolation'), help='Interpolations file.', type=str)
        parser.add_argument('-batch_size', default=64, help='Batch size.', type=int)
        parser.add_argument('-latent_space_size', default=10, help='Size of latent space.', type=int)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')
        parser.add_argument('-results_file', default='', help='Results file for evaluation.', type=str)
        parser.add_argument('-output_directory', default='', help='Output directory for plots.', type=str)
        parser.add_argument('-log_file', default=paths.log_file('test_auto_encoder'), help='Log file.', type=str)

        # Some network parameters.
        parser.add_argument('-network_architecture', default='standard', help='Architecture type.')
        parser.add_argument('-network_activation', default='relu', help='Activation function to use.', type=str)
        parser.add_argument('-network_no_batch_normalization', default=False, help='Do not use batch normalization.', action='store_true')
        parser.add_argument('-network_channels', default=16, help='Channels of first convolutional layer, afterwards channels are doubled.', type=int)
        parser.add_argument('-network_dropout', default=False, action='store_true', help='Whether to use dropout.')
        parser.add_argument('-network_units', default='1024,1024,1024,1024', help='Units for MLP.')

        return parser

    def reconstruction_loss(self, batch_images, output_images):
        """
        Reconstruction loss.

        :param batch_images: target images
        :type batch_images: torch.autograd.Variable
        :param output_images: predicted images
        :type output_images: torch.autograd.Variable
        :return: error
        :rtype: torch.autograd.Variable
        """

        return torch.mean(torch.mul(batch_images - output_images, batch_images - output_images))

    def test_train(self):
        """
        Test on training set.
        """

        pred_codes = None
        pred_images = None
        num_batches = int(math.ceil(self.train_images.shape[0] / self.args.batch_size))

        for b in range(num_batches):
            b_start = b * self.args.batch_size
            b_end = min((b + 1) * self.args.batch_size, self.train_images.shape[0])
            batch_images = common.torch.as_variable(self.train_images[b_start: b_end], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            # Important to get the correct codes!
            output_mu, output_logvar = self.encoder(batch_images)
            output_images = self.decoder(output_mu)

            output_mu = output_mu.cpu().detach().numpy()
            pred_codes = common.numpy.concatenate(pred_codes, output_mu)

            output_images = numpy.squeeze(numpy.transpose(output_images.cpu().detach().numpy(), (0, 2, 3, 1)))
            pred_images = common.numpy.concatenate(pred_images, output_images)

            if b % 100 == 50:
                log('[Testing] %d' % b)

        assert pred_codes.shape[0] == self.train_images.shape[0], 'computed invalid number of train codes'
        if self.args.train_theta_file:
            utils.write_hdf5(self.args.train_theta_file, pred_codes)
            log('[Testing] wrote %s' % self.args.train_theta_file)

        assert pred_images.shape[0] == self.train_images.shape[0], 'computed invalid number of test images'
        if self.args.train_reconstruction_file:
            utils.write_hdf5(self.args.train_reconstruction_file, pred_images)
            log('[Testing] wrote %s' % self.args.train_reconstruction_file)

        threshold = 0.9
        percentage = 0
        # values = numpy.linalg.norm(pred_codes, ord=2, axis=1)
        values = numpy.max(numpy.abs(pred_codes), axis=1)

        while percentage < 0.9:
            threshold += 0.1
            percentage = numpy.sum(values <= threshold) / float(values.shape[0])
            log('[Testing] threshold %g percentage %g' % (threshold, percentage))
        log('[Testing] taking threshold %g with percentage %g' % (threshold, percentage))

        if self.args.output_directory and utils.display():
            # fit = 10
            # plot_file = os.path.join(self.args.output_directory, 'train_codes_tsne')
            # plot.manifold(plot_file, pred_codes[::fit], None, None, 'tsne', None, title='t-SNE of Training Codes')
            # log('[Testing] wrote %s' % plot_file)

            for d in range(1, pred_codes.shape[1]):
                plot_file = os.path.join(self.args.output_directory, 'train_codes_%s' % d)
                plot.scatter(plot_file, pred_codes[:, 0], pred_codes[:, d], (values <= threshold).astype(int),
                             ['greater %g' % threshold, 'smaller %g' % threshold], title='Dimensions 0 and %d of Training Codes' % d)
                log('[Testing] wrote %s' % plot_file)

    def test_test(self):
        """
        Test on testing set.
        """

        num_batches = int(math.ceil(self.test_images.shape[0] / self.args.batch_size))

        for b in range(num_batches):
            b_start = b * self.args.batch_size
            b_end = min((b + 1) * self.args.batch_size, self.test_images.shape[0])

            batch_images = common.torch.as_variable(self.test_images[b_start: b_end], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            # Important to get the correct codes!
            output_codes, output_logvar = self.encoder(batch_images)
            output_images = self.decoder(output_codes)
            e = self.reconstruction_loss(batch_images, output_images)
            self.reconstruction_error += e.data

            self.code_mean += torch.mean(output_codes).item()
            self.code_var += torch.var(output_codes).item()

            output_images = numpy.squeeze(numpy.transpose(output_images.cpu().detach().numpy(), (0, 2, 3, 1)))
            self.pred_images = common.numpy.concatenate(self.pred_images, output_images)

            output_codes = output_codes.cpu().detach().numpy()
            self.pred_codes = common.numpy.concatenate(self.pred_codes, output_codes)

            if b % 100 == 50:
                log('[Testing] %d' % b)

        assert self.pred_images.shape[0] == self.test_images.shape[0], 'computed invalid number of test images'
        if self.args.reconstruction_file:
            utils.write_hdf5(self.args.reconstruction_file, self.pred_images)
            log('[Testing] wrote %s' % self.args.reconstruction_file)

        if self.args.test_theta_file:
            assert self.pred_codes.shape[0] == self.test_images.shape[0], 'computed invalid number of test codes'
            utils.write_hdf5(self.args.test_theta_file, self.pred_codes)
            log('[Testing] wrote %s' % self.args.test_theta_file)

        threshold = 0.9
        percentage = 0
        # values = numpy.linalg.norm(pred_codes, ord=2, axis=1)
        values = numpy.max(numpy.abs(self.pred_codes), axis=1)

        while percentage < 0.9:
            threshold += 0.1
            percentage = numpy.sum(values <= threshold) / float(values.shape[0])
            log('[Testing] threshold %g percentage %g' % (threshold, percentage))
        log('[Testing] taking threshold %g with percentage %g' % (threshold, percentage))

        if self.args.output_directory and utils.display():
            # fit = 10
            # plot_file = os.path.join(self.args.output_directory, 'test_codes')
            # plot.manifold(plot_file, pred_codes[::fit], None, None, 'tsne', None, title='t-SNE of Test Codes')
            # log('[Testing] wrote %s' % plot_file)

            for d in range(1, self.pred_codes.shape[1]):
                plot_file = os.path.join(self.args.output_directory, 'test_codes_%s' % d)
                plot.scatter(plot_file, self.pred_codes[:, 0], self.pred_codes[:, d], (values <= threshold).astype(int),
                             ['greater %g' % threshold, 'smaller %g' % threshold], title='Dimensions 0 and %d of Test Codes' % d)
                log('[Testing] wrote %s' % plot_file)

        self.reconstruction_error /= num_batches
        log('[Testing] reconstruction error %g' % self.reconstruction_error)

    def test_random(self):
        """
        Test random.
        """

        pred_images = None
        codes = numpy.random.normal(0, 1, (1000, self.args.latent_space_size)).astype(numpy.float32)
        num_batches = int(math.ceil(codes.shape[0] / self.args.batch_size))

        for b in range(num_batches):
            b_start = b * self.args.batch_size
            b_end = min((b + 1) * self.args.batch_size, self.test_images.shape[0])
            batch_codes = common.torch.as_variable(codes[b_start: b_end], self.args.use_gpu)

            # To get the correct images!
            output_images = self.decoder(batch_codes)

            output_images = numpy.squeeze(numpy.transpose(output_images.cpu().detach().numpy(), (0, 2, 3, 1)))
            pred_images = common.numpy.concatenate(pred_images, output_images)

            if b % 100 == 50:
                log('[Testing] %d' % b)

        utils.write_hdf5(self.args.random_file, pred_images)
        log('[Testing] wrote %s' % self.args.random_file)

    def test_interpolation(self):
        """
        Test interpolation.
        """

        interpolations = None
        perm = numpy.random.permutation(numpy.array(range(self.pred_codes.shape[0])))

        for i in range(50):
            first = self.pred_codes[i]
            second = self.pred_codes[perm[i]]
            linfit = scipy.interpolate.interp1d([0, 1], numpy.vstack([first, second]), axis=0)
            interpolations = common.numpy.concatenate(interpolations, linfit(numpy.linspace(0, 1, 10)))

        pred_images = None
        num_batches = int(math.ceil(interpolations.shape[0] / self.args.batch_size))
        interpolations = interpolations.astype(numpy.float32)

        for b in range(num_batches):
            b_start = b * self.args.batch_size
            b_end = min((b + 1) * self.args.batch_size, self.test_images.shape[0])
            batch_codes = common.torch.as_variable(interpolations[b_start: b_end], self.args.use_gpu)

            # To get the correct images!
            output_images = self.decoder(batch_codes)

            output_images = numpy.squeeze(numpy.transpose(output_images.cpu().detach().numpy(), (0, 2, 3, 1)))
            pred_images = common.numpy.concatenate(pred_images, output_images)

            if b % 100 == 50:
                log('[Testing] %d' % b)

        utils.write_hdf5(self.args.interpolation_file, pred_images)
        log('[Testing] wrote %s' % self.args.interpolation_file)

    def test(self):
        """
        Test the model.
        """

        assert self.encoder is not None and self.decoder is not None

        self.encoder.eval()
        log('[Testing] set encoder to eval')
        self.decoder.eval()
        log('[Testing] set decoder to eval')

        if self.args.train_theta_file or self.train_reconstruction_file:
            self.test_train()
        self.test_test()
        if self.args.random_file:
            self.test_random()
        if self.args.interpolation_file:
            self.test_interpolation()

        self.results = {
            'reconstruction_error': self.reconstruction_error,
            'code_mean': self.code_mean,
            'code_var': self.code_var,
        }
        if self.args.results_file:
            utils.write_pickle(self.args.results_file, self.results)
            log('[Testing] wrote %s' % self.args.results_file)

    def main(self):
        """
        Main which should be overwritten.
        """

        self.train_images = utils.read_hdf5(self.args.train_images_file).astype(numpy.float32)
        log('[Testing] read %s' % self.args.train_images_file)

        self.test_images = utils.read_hdf5(self.args.test_images_file).astype(numpy.float32)
        log('[Testing] read %s' % self.args.test_images_file)

        # For handling both color and gray images.
        if len(self.train_images.shape) < 4:
            self.train_images = numpy.expand_dims(self.train_images, axis=3)
            self.test_images = numpy.expand_dims(self.test_images, axis=3)
            log('[Testing] no color images, adjusted size')
        self.resolution = self.train_images.shape[2]
        log('[Testing] resolution %d' % self.resolution)

        self.train_codes = utils.read_hdf5(self.args.train_codes_file).astype(numpy.float32)
        log('[Testing] read %s' % self.args.train_codes_file)

        self.test_codes = utils.read_hdf5(self.args.test_codes_file).astype(numpy.float32)
        log('[Testing] read %s' % self.args.test_codes_file)

        self.train_codes = self.train_codes[:, self.args.label_index]
        self.test_codes = self.test_codes[:, self.args.label_index]

        if self.args.label >= 0:
            self.train_images = self.train_images[self.train_codes == self.args.label]
            self.test_images = self.test_images[self.test_codes == self.args.label]

        log('[Testing] using %d input channels' % self.test_images.shape[3])
        network_units = list(map(int, self.args.network_units.split(',')))
        self.encoder = models.LearnedVariationalEncoder(self.args.latent_space_size, 0, resolution=(self.train_images.shape[3], self.train_images.shape[1], self.train_images.shape[2]),
                                                        architecture=self.args.network_architecture,
                                                        start_channels=self.args.network_channels,
                                                        activation=self.args.network_activation,
                                                        batch_normalization=not self.args.network_no_batch_normalization,
                                                        units=network_units)
        self.decoder = models.LearnedDecoder(self.args.latent_space_size, resolution=(self.train_images.shape[3], self.train_images.shape[1], self.train_images.shape[2]),
                                             architecture=self.args.network_architecture,
                                             start_channels=self.args.network_channels,
                                             activation=self.args.network_activation,
                                             batch_normalization=not self.args.network_no_batch_normalization,
                                             units=network_units)
        log(self.encoder)
        log(self.decoder)

        assert os.path.exists(self.args.encoder_file) and os.path.exists(self.args.decoder_file)
        state = State.load(self.args.encoder_file)
        log('[Testing] loaded %s' % self.args.encoder_file)

        self.encoder.load_state_dict(state.model)
        log('[Testing] loaded encoder')

        state = State.load(self.args.decoder_file)
        log('[Testing] loaded %s' % self.args.decoder_file)

        self.decoder.load_state_dict(state.model)
        log('[Testing] loaded decoder')

        if self.args.use_gpu and not cuda.is_cuda(self.encoder):
            self.encoder = self.encoder.cuda()
        if self.args.use_gpu and not cuda.is_cuda(self.decoder):
            self.decoder = self.decoder.cuda()

        log('[Testing] model needs %gMiB' % ((cuda.estimate_size(self.encoder) + cuda.estimate_size(self.decoder))/ (1024 * 1024)))
        self.test()


if __name__ == '__main__':
    program = TestVariationalAutoEncoder()
    program.main()