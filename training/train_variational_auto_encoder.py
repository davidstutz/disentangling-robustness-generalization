import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
import models
from common.log import log, Log
from common.scheduler import ADAMScheduler
from common.state import State
from common import cuda
from common.timer import elapsed
from common import paths
import common.torch
import common.numpy
import scipy.interpolate
import torch
import numpy
import argparse
import math
import functools


if utils.display():
    from common import plot
    from common import vis


class TrainVariationalAutoEncoder:
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

        if self.args.log_file:
            utils.makedir(os.path.dirname(self.args.log_file))
            Log.get_instance().attach(open(self.args.log_file, 'w'))

        log('-- ' + self.__class__.__name__)
        for key in vars(self.args):
            log('[Training] %s=%s' % (key, str(getattr(self.args, key))))

        utils.makedir(os.path.dirname(self.args.encoder_file))
        utils.makedir(os.path.dirname(self.args.decoder_file))
        utils.makedir(os.path.dirname(self.args.log_file))

        self.resolution = None
        """ (int) Resolution. """

        self.encoder = None
        """ (models.LearnedVariationalEncoder) Encoder. """

        self.decoder = None
        """ (models.LearnedDecoder) Decoder. """

        self.auto_encoder = None
        """ (models.LearnedVariationalAutoEncoder) Auto encoder. """

        self.train_statistics = numpy.zeros((0, 10))
        """ (numpy.ndarray) Will hold training statistics. """

        self.test_statistics = numpy.zeros((0, 10))
        """ (numpy.ndarray) Will hold testing statistics. """

        self.results = dict()
        """ (dict) Results. """

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

        parser = argparse.ArgumentParser(description='Train auto encoder.')
        parser.add_argument('-train_images_file', default=paths.train_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-train_codes_file', default=paths.train_codes_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-label', default=-1, help='Label to constrain to.', type=int)
        parser.add_argument('-encoder_file', default=paths.state_file('encoder'), help='Snapshot state file.', type=str)
        parser.add_argument('-decoder_file', default=paths.state_file('decoder'), help='Snapshot state file.', type=str)
        parser.add_argument('-reconstruction_file', default=paths.results_file('reconstructions'), help='Reconstructions file.', type=str)
        parser.add_argument('-interpolation_file', default=paths.results_file('interpolations'), help='Interpolation file.', type=str)
        parser.add_argument('-random_file', default=paths.results_file('random'), help='Reconstructions file.', type=str)
        parser.add_argument('-log_file', default=paths.log_file('auto_encoder'), help='Log file.', type=str)
        parser.add_argument('-batch_size', default=64, help='Batch size.', type=int)
        parser.add_argument('-latent_space_size', default=10, help='Size of latent space.', type=int)
        parser.add_argument('-epochs', default=20, help='Number of epochs.', type=int)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')
        parser.add_argument('-base_lr', default=0.01, type=float, help='Base learning rate.')
        parser.add_argument('-base_lr_decay', default=0.9, type=float, help='Base learning rate.')
        parser.add_argument('-results_file', default='', help='Results file for evaluation.', type=str)
        parser.add_argument('-training_file', default=paths.results_file('auto_encoder_training'), help='Training statistics file.', type=str)
        parser.add_argument('-testing_file', default=paths.results_file('auto_encoder_testing'), help='Testing statistics file.', type=str)
        parser.add_argument('-error_file', default=paths.image_file('auto_encoder_error'), help='Error plot file.', type=str)
        parser.add_argument('-beta', default=1, help='Weight of KLD.', type=float)
        parser.add_argument('-weight_decay', default=0.0001, help='Weight decay importance.', type=float)
        parser.add_argument('-absolute_error', default=False, action='store_true', help='Use absolute loss.')

        # Some network parameters.
        parser.add_argument('-network_architecture', default='standard', help='Classifier architecture to use.', type=str)
        parser.add_argument('-network_activation', default='relu', help='Activation function to use.', type=str)
        parser.add_argument('-network_no_batch_normalization', default=False, help='Do not use batch normalization.', action='store_true')
        parser.add_argument('-network_channels', default=16, help='Channels of first convolutional layer, afterwards channels are doubled.', type=int)
        parser.add_argument('-network_dropout', default=False, action='store_true', help='Whether to use dropout.')
        parser.add_argument('-network_units', default='1024,1024,1024,1024', help='Units for MLP.')

        return parser

    def reconstruction_loss(self, batch_images, output_images):
        """
        Reconstruction loss.

        :param batch_images: original images
        :type batch_images: torch.autograd.Variable
        :param output_images: output images
        :type output_images: torch.autograd.Variable
        :return: error
        :rtype: torch.autograd.Variable
        """

        if self.args.absolute_error:
            return torch.sum(torch.abs(batch_images - output_images))
        else:
            return torch.sum(torch.mul(batch_images - output_images, batch_images - output_images))

    def reconstruction_error(self, batch_images, output_images):
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

    def latent_loss(self, output_mu, output_logvar):
        """
        Latent KLD loss.

        :param output_mu: target images
        :type output_mu: torch.autograd.Variable
        :param output_logvar: predicted images
        :type output_logvar: torch.autograd.Variable
        :return: error
        :rtype: torch.autograd.Variable
        """

        return -0.5 * torch.sum(1 + output_logvar - output_mu.pow(2) - output_logvar.exp())

    def train(self, epoch):
        """
        Train for one epoch.

        :param epoch: current epoch
        :type epoch: int
        """

        assert self.encoder is not None and self.decoder is not None
        assert self.scheduler is not None

        self.auto_encoder.train()
        log('[Training] %d set auto encoder to train' % epoch)
        self.encoder.train()
        log('[Training] %d set encoder to train' % epoch)
        self.decoder.train()
        log('[Training] %d set decoder to train' % epoch)

        num_batches = int(math.ceil(self.train_images.shape[0]/self.args.batch_size))
        assert self.encoder.training is True

        permutation = numpy.random.permutation(self.train_images.shape[0])
        permutation = numpy.concatenate((permutation, permutation[:self.args.batch_size]), axis=0)

        for b in range(num_batches):
            self.scheduler.update(epoch, float(b)/num_batches)

            perm = permutation[b * self.args.batch_size: (b + 1) * self.args.batch_size]
            batch_images = common.torch.as_variable(self.train_images[perm], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            output_images, output_mu, output_logvar = self.auto_encoder(batch_images)
            reconstruction_loss = self.reconstruction_loss(batch_images, output_images)

            self.scheduler.optimizer.zero_grad()
            latent_loss = self.latent_loss(output_mu, output_logvar)
            loss = self.args.beta*reconstruction_loss + latent_loss
            loss.backward()
            self.scheduler.optimizer.step()
            reconstruction_loss = reconstruction_loss.item()
            latent_loss = latent_loss.item()

            reconstruction_error = self.reconstruction_error(batch_images, output_images)
            reconstruction_error = reconstruction_error.item()

            iteration = epoch*num_batches + b + 1
            self.train_statistics = numpy.vstack((self.train_statistics, numpy.array([
                iteration,
                iteration * self.args.batch_size,
                min(num_batches, iteration),
                min(num_batches, iteration) * self.args.batch_size,
                reconstruction_loss,
                reconstruction_error,
                latent_loss,
                torch.mean(output_mu).item(),
                torch.var(output_mu).item(),
                torch.mean(output_logvar).item(),
            ])))

            skip = 10
            if b%skip == skip//2:
                log('[Training] %d | %d: %g (%g) %g %g %g %g' % (
                    epoch,
                    b,
                    numpy.mean(self.train_statistics[max(0, iteration-skip):iteration, 4]),
                    numpy.mean(self.train_statistics[max(0, iteration-skip):iteration, 5]),
                    numpy.mean(self.train_statistics[max(0, iteration-skip):iteration, 6]),
                    numpy.mean(self.train_statistics[max(0, iteration-skip):iteration, 7]),
                    numpy.mean(self.train_statistics[max(0, iteration-skip):iteration, 8]),
                    numpy.mean(self.train_statistics[max(0, iteration-skip):iteration, 9]),
                ))

    def test(self, epoch):
        """
        Test the model.

        :param epoch: current epoch
        :type epoch: int
        """

        assert self.encoder is not None and self.decoder is not None
        assert self.scheduler is not None

        self.auto_encoder.eval()
        log('[Training] %d set auto encoder to eval' % epoch)
        self.encoder.eval()
        log('[Training] %d set encoder to eval' % epoch)
        self.decoder.eval()
        log('[Training] %d set decoder to eval' % epoch)

        reconstruction_loss = 0
        reconstruction_error = 0
        latent_loss = 0
        mean = 0
        var = 0
        logvar = 0
        pred_images = None
        pred_codes = None
        num_batches = int(math.ceil(self.test_images.shape[0]/self.args.batch_size))
        assert self.encoder.training is False

        for b in range(num_batches):
            batch_images = common.torch.as_variable(self.test_images[b*self.args.batch_size: min((b + 1)*self.args.batch_size, self.test_images.shape[0])], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            output_images, output_mu, output_logvar = self.auto_encoder(batch_images)
            e = self.reconstruction_loss(batch_images, output_images)
            reconstruction_loss += e.item()
            e = self.reconstruction_error(batch_images, output_images)
            reconstruction_error += e.item()
            e = self.latent_loss(output_mu, output_logvar)
            latent_loss += e.item()

            mean += torch.mean(output_mu).item()
            var += torch.var(output_mu).item()
            logvar += torch.mean(output_logvar).item()

            output_images = numpy.squeeze(numpy.transpose(output_images.cpu().detach().numpy(), (0, 2, 3, 1)))
            pred_images = common.numpy.concatenate(pred_images, output_images)
            output_codes = output_mu.cpu().detach().numpy()
            pred_codes = common.numpy.concatenate(pred_codes, output_codes)

        utils.write_hdf5(self.args.reconstruction_file, pred_images)
        log('[Training] %d: wrote %s' % (epoch, self.args.reconstruction_file))

        if utils.display():
            png_file = self.args.reconstruction_file + '.%d.png' % epoch
            if epoch == 0:
                vis.mosaic(png_file, self.test_images[:225], 15, 5, 'gray', 0, 1)
            else:
                vis.mosaic(png_file, pred_images[:225], 15, 5, 'gray', 0, 1)
            log('[Training] %d: wrote %s' % (epoch, png_file))

        reconstruction_loss /= num_batches
        reconstruction_error /= num_batches
        latent_loss /= num_batches
        mean /= num_batches
        var /= num_batches
        logvar /= num_batches
        log('[Training] %d: test %g (%g) %g %g %g %g' % (epoch, reconstruction_loss, reconstruction_error, latent_loss, mean, var, logvar))

        num_batches = int(math.ceil(self.train_images.shape[0] / self.args.batch_size))
        iteration = epoch * num_batches
        self.test_statistics = numpy.vstack((self.test_statistics, numpy.array([
            iteration,
            iteration * self.args.batch_size,
            min(num_batches, iteration),
            min(num_batches, iteration) * self.args.batch_size,
            reconstruction_loss,
            reconstruction_error,
            latent_loss,
            mean,
            var,
            logvar
        ])))

        pred_images = None
        codes = numpy.random.normal(0, 1, (1000, self.args.latent_space_size)).astype(numpy.float32)
        num_batches = int(math.ceil(codes.shape[0] / self.args.batch_size))

        for b in range(num_batches):
            batch_codes = common.torch.as_variable(codes[b * self.args.batch_size: min((b + 1) * self.args.batch_size, codes.shape[0])], self.args.use_gpu)
            output_images = self.auto_encoder.decoder(batch_codes)

            output_images = numpy.squeeze(numpy.transpose(output_images.cpu().detach().numpy(), (0, 2, 3, 1)))
            pred_images = common.numpy.concatenate(pred_images, output_images)

        utils.write_hdf5(self.args.random_file, pred_images)
        log('[Training] %d: wrote %s' % (epoch, self.args.random_file))

        if utils.display() and epoch > 0:
            png_file = self.args.random_file + '.%d.png' % epoch
            vis.mosaic(png_file, pred_images[:225], 15, 5, 'gray', 0, 1)
            log('[Training] %d: wrote %s' % (epoch, png_file))

        interpolations = None
        perm = numpy.random.permutation(numpy.array(range(pred_codes.shape[0])))

        for i in range(50):
            first = pred_codes[i]
            second = pred_codes[perm[i]]
            linfit = scipy.interpolate.interp1d([0, 1], numpy.vstack([first, second]), axis=0)
            interpolations = common.numpy.concatenate(interpolations, linfit(numpy.linspace(0, 1, 10)))

        pred_images = None
        num_batches = int(math.ceil(interpolations.shape[0] / self.args.batch_size))
        interpolations = interpolations.astype(numpy.float32)

        for b in range(num_batches):
            b_start = b * self.args.batch_size
            b_end = min((b + 1) * self.args.batch_size, self.test_images.shape[0])
            if b_start >= b_end: break

            batch_codes = common.torch.as_variable(interpolations[b_start: b_end], self.args.use_gpu)
            output_images = self.decoder(batch_codes)
            output_images = numpy.squeeze(numpy.transpose(output_images.cpu().detach().numpy(), (0, 2, 3, 1)))
            pred_images = common.numpy.concatenate(pred_images, output_images)

            if b % 100 == 50:
                log('[Testing] %d' % b)

        utils.write_hdf5(self.args.interpolation_file, pred_images)
        log('[Testing] wrote %s' % self.args.interpolation_file)

        if utils.display() and epoch > 0:
            png_file = self.args.interpolation_file + '.%d.png' % epoch
            vis.mosaic(png_file, pred_images[:100], 10, 5, 'gray', 0, 1)
            log('[Training] %d: wrote %s' % (epoch, png_file))

    def plot(self):
        """
        Plot error and accuracy.
        """

        if self.args.error_file:
            plot.line(self.args.error_file, [
                self.train_statistics[:, 0],
                self.test_statistics[:, 0],
                self.train_statistics[:, 0],
                self.test_statistics[:, 0]
            ], [
                self.train_statistics[:, 4],
                self.test_statistics[:, 4],
                self.train_statistics[:, 5],
                self.test_statistics[:, 5]
            ],
            ['Training Reconstruction Error', 'Testing Reconstruction Error', 'Training Latent Error', 'Testing Latent Error'], title='Training and Testing Error during Training',
            xlabel='\#Samples Seen', ylabel='Error')

    def loop(self):
        """
        Main loop for training and testing, saving ...
        """

        params = {
            'lr': self.args.base_lr,
            'lr_decay': self.args.base_lr_decay,
            'lr_min': 0.00000001,
            'weight_decay': self.args.weight_decay
        }

        e = 0
        if os.path.exists(self.args.encoder_file) and os.path.exists(self.args.decoder_file):
            state = State.load(self.args.encoder_file)
            log('[Training] loaded %s' % self.args.encoder_file)

            self.encoder.load_state_dict(state.model)
            log('[Training] loaded encoder')

            state = State.load(self.args.decoder_file)
            log('[Training] loaded %s' % self.args.decoder_file)

            self.decoder.load_state_dict(state.model)
            log('[Training] loaded decoder')

            if self.args.use_gpu and not cuda.is_cuda(self.encoder):
                self.encoder = self.encoder.cuda()
            if self.args.use_gpu and not cuda.is_cuda(self.decoder):
                self.decoder = self.decoder.cuda()

            optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), params['lr'])
            optimizer.load_state_dict(state.optimizer)
            self.scheduler = ADAMScheduler(optimizer, **params)

            e = state.epoch + 1
            self.scheduler.update(e)
        else:
            if self.args.use_gpu and not cuda.is_cuda(self.encoder):
                self.encoder = self.encoder.cuda()
            if self.args.use_gpu and not cuda.is_cuda(self.decoder):
                self.decoder = self.decoder.cuda()

            self.scheduler = ADAMScheduler(list(self.encoder.parameters()) + list(self.decoder.parameters()), **params)
            self.scheduler.initialize() # !

        log('[Training] model needs %gMiB' % (cuda.estimate_size(self.encoder)/(1024*1024)))

        while e < self.args.epochs:
            log('[Training] %s' % self.scheduler.report())

            testing = elapsed(functools.partial(self.test, e))
            training = elapsed(functools.partial(self.train, e))
            log('[Training] %gs training, %gs testing' % (training, testing))

            #utils.remove(self.args.encoder_file + '.%d' % (e - 1))
            #utils.remove(self.args.decoder_file + '.%d' % (e - 1))
            State.checkpoint(self.encoder, self.scheduler.optimizer, e, self.args.encoder_file + '.%d' % e)
            State.checkpoint(self.decoder, self.scheduler.optimizer, e, self.args.decoder_file + '.%d' % e)
            log('[Training] %d: checkpoint' % e)
            torch.cuda.empty_cache() # necessary?

            # Save statistics and plots.
            if self.args.training_file:
                utils.write_hdf5(self.args.training_file, self.train_statistics)
                log('[Training] %d: wrote %s' % (e, self.args.training_file))
            if self.args.testing_file:
                utils.write_hdf5(self.args.testing_file, self.test_statistics)
                log('[Training] %d: wrote %s' % (e, self.args.testing_file))

            if utils.display():
                self.plot()

            e += 1 # !

        testing = elapsed(functools.partial(self.test, e))
        log('[Training] %gs testing' % (testing))

        #utils.remove(self.args.encoder_file + '.%d' % (e - 1))
        #utils.remove(self.args.decoder_file + '.%d' % (e - 1))
        State.checkpoint(self.encoder, self.scheduler.optimizer, e, self.args.encoder_file)
        State.checkpoint(self.decoder, self.scheduler.optimizer, e, self.args.decoder_file)

        self.results = {
            'training_statistics': self.train_statistics,
            'testing_statistics': self.test_statistics,
        }
        if self.args.results_file:
            utils.write_pickle(self.args.results_file, self.results)
            log('[Training] wrote %s' % self.args.results_file)

    def load_data(self):
        """
        Load data.
        """

        self.train_images = utils.read_hdf5(self.args.train_images_file).astype(numpy.float32)
        log('[Training] read %s' % self.args.train_images_file)

        self.test_images = utils.read_hdf5(self.args.test_images_file).astype(numpy.float32)
        log('[Training] read %s' % self.args.test_images_file)

        # For handling both color and gray images.
        if len(self.train_images.shape) < 4:
            self.train_images = numpy.expand_dims(self.train_images, axis=3)
            self.test_images = numpy.expand_dims(self.test_images, axis=3)
            log('[Training] no color images, adjusted size')
        self.resolution = self.train_images.shape[2]
        log('[Training] resolution %d' % self.resolution)

        self.train_codes = utils.read_hdf5(self.args.train_codes_file).astype(numpy.float32)
        log('[Training] read %s' % self.args.train_codes_file)

        self.test_codes = utils.read_hdf5(self.args.test_codes_file).astype(numpy.float32)
        log('[Training] read %s' % self.args.test_codes_file)

        self.train_codes = self.train_codes[:, self.args.label_index]
        self.test_codes = self.test_codes[:, self.args.label_index]

        if self.args.label >= 0:
            self.train_images = self.train_images[self.train_codes == self.args.label]
            self.test_images = self.test_images[self.test_codes == self.args.label]

    def load_models(self):
        """
        Init models.
        """

        log('[Trainin] using %d input channels' % self.train_images.shape[3])
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
        self.auto_encoder = models.LearnedVariationalAutoEncoder(self.encoder, self.decoder)
        log(self.encoder)
        log(self.decoder)

    def main(self):
        """
        Main which should be overwritten.
        """

        self.load_data()
        self.load_models()
        self.loop()


if __name__ == '__main__':
    program = TrainVariationalAutoEncoder()
    program.main()