import os
import sys

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
from common.log import log, logw
from common import paths
from common.state import State
from common import cuda
import common.torch
import common.numpy
import models

import math
import torch
import numpy
import argparse
import sklearn.decomposition
import sklearn.neighbors
from training import detect_attack_classifier_nn


class DetectAttackLearnedDecoderClassifierNN(detect_attack_classifier_nn.DetectAttackClassifierNN):
    """
    Test a trained classifier.
    """

    def __init__(self, args=None):
        """
        Initialize.

        :param args: optional arguments if not to use sys.argv
        :type args: [str]
        """

        super(DetectAttackLearnedDecoderClassifierNN, self).__init__(args)

        self.perturbation_images = None
        """ (numpy.ndarray) Perturbation images. """

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Detect attacks on classifier.')
        parser.add_argument('-mode', default='svd', help='Mode.', type=str)
        parser.add_argument('-train_images_file', default=paths.train_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-train_codes_file', default=paths.train_codes_file(), help='HDF5 file codes dataset.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file codes dataset.', type=str)
        parser.add_argument('-test_theta_file', default='', help='HDF5 file codes dataset.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-perturbations_file', default=paths.results_file('decoder/perturbations'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-success_file', default=paths.results_file('decoder/success'), help='HDF5 file containing success indicators.', type=str)
        parser.add_argument('-accuracy_file', default=paths.results_file('decoder/accuracy'), help='HDF5 file containing accuracy indicators.', type=str)
        parser.add_argument('-batch_size', default=64, help='Batch size.', type=int)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')
        parser.add_argument('-pre_pca', default=20, help='PCA dimensionality reduction ebfore NN.', type=int)
        parser.add_argument('-n_nearest_neighbors', default=50, help='Number of NNs to consider.', type=int)
        parser.add_argument('-n_pca', default=10, help='Number of NNs to consider.', type=int)
        parser.add_argument('-n_fit', default=100000, help='Training images to fit.', type=int)
        parser.add_argument('-plot_directory', default=paths.experiment_dir('decoder/detection'), help='Plot directory.', type=str)
        parser.add_argument('-max_samples', default=1000, help='Number of samples.', type=int)

        # Some decoder parameters.
        parser.add_argument('-decoder_files', default=paths.state_file('decoder'), help='Decoder files.', type=str)
        parser.add_argument('-latent_space_size', default=10, help='Size of latent space.', type=int)
        parser.add_argument('-decoder_architecture', default='standard', help='Architecture to use.', type=str)
        parser.add_argument('-decoder_activation', default='relu', help='Activation function to use.', type=str)
        parser.add_argument('-decoder_no_batch_normalization', default=False, help='Do not use batch normalization.', action='store_true')
        parser.add_argument('-decoder_channels', default=16, help='Channels of first convolutional layer, afterwards channels are doubled.', type=int)
        parser.add_argument('-decoder_dropout', default=False, action='store_true', help='Whether to use dropout.')
        parser.add_argument('-decoder_units', default='1024,1024,1024,1024', help='Units for MLP.')

        return parser

    def load_data(self):
        """
        Load data and model.
        """

        with logw('[Detection] read %s' % self.args.train_images_file):
            self.nearest_neighbor_images = utils.read_hdf5(self.args.train_images_file)
            assert len(self.nearest_neighbor_images.shape) == 3

        with logw('[Detection] read %s' % self.args.test_images_file):
            self.test_images = utils.read_hdf5(self.args.test_images_file)
            if len(self.test_images.shape) < 4:
                self.test_images = numpy.expand_dims(self.test_images, axis=3)

        with logw('[Detection] read %s' % self.args.train_codes_file):
            self.train_codes = utils.read_hdf5(self.args.train_codes_file)
            self.train_codes = self.train_codes[:, self.args.label_index]

        with logw('[Detection] read %s' % self.args.test_codes_file):
            self.test_codes = utils.read_hdf5(self.args.test_codes_file)
            self.test_codes = self.test_codes[:, self.args.label_index]

        with logw('[Detection] read %s' % self.args.test_theta_file):
            self.test_theta = utils.read_hdf5(self.args.test_theta_file)

        with logw('[Detection] read %s' % self.args.perturbations_file):
            self.perturbations = utils.read_hdf5(self.args.perturbations_file)
            assert len(self.perturbations.shape) == 3

        with logw('[Detection] read %s' % self.args.success_file):
            self.success = utils.read_hdf5(self.args.success_file)

        with logw('[Detection] read %s' % self.args.accuracy_file):
            self.accuracy = utils.read_hdf5(self.args.accuracy_file)

        self.perturbations = numpy.swapaxes(self.perturbations, 0, 1)
        num_attempts = self.perturbations.shape[1]
        self.test_images = self.test_images[:self.perturbations.shape[0]]
        self.train_images = self.nearest_neighbor_images[:self.perturbations.shape[0]]
        self.test_codes = self.test_codes[:self.perturbations.shape[0]]
        self.accuracy = self.accuracy[:self.perturbations.shape[0]]
        self.test_theta = self.test_theta[:self.perturbations.shape[0]]

        self.perturbations = self.perturbations.reshape((self.perturbations.shape[0]*self.perturbations.shape[1], self.perturbations.shape[2]))
        self.success = numpy.swapaxes(self.success, 0, 1)
        self.success = self.success.reshape((self.success.shape[0]*self.success.shape[1]))

        self.accuracy = numpy.repeat(self.accuracy, num_attempts, axis=0)
        self.test_images = numpy.repeat(self.test_images, num_attempts, axis=0)
        self.train_images = numpy.repeat(self.train_images, num_attempts, axis=0)
        self.test_codes = numpy.repeat(self.test_codes, num_attempts, axis=0)
        self.test_theta = numpy.repeat(self.test_theta, num_attempts, axis=0)

        max_samples = self.args.max_samples
        self.success = self.success[:max_samples]
        self.accuracy = self.accuracy[:max_samples]
        self.perturbations = self.perturbations[:max_samples]
        self.test_images = self.test_images[:max_samples]
        self.train_images = self.train_images[:max_samples]
        self.test_codes = self.test_codes[:max_samples]
        self.test_theta = self.test_theta[:max_samples]

        assert self.args.decoder_files
        decoder_files = self.args.decoder_files.split(',')
        for decoder_file in decoder_files:
            assert os.path.exists(decoder_file), 'could not find %s' % decoder_file

        resolution = [1 if len(self.test_images.shape) <= 3 else self.test_images.shape[3], self.test_images.shape[1], self.test_images.shape[2]]
        decoder_units = list(map(int, self.args.decoder_units.split(',')))

        if len(decoder_files) > 1:
            log('[Detection] loading multiple decoders')
            decoders = []
            for i in range(len(decoder_files)):
                decoder = models.LearnedDecoder(self.args.latent_space_size,
                                                resolution=resolution,
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
                log('[Detection] loaded %s' % decoder_files[i])
            self.model = models.SelectiveDecoder(decoders, resolution=resolution)
        else:
            log('[Detection] loading one decoder')
            decoder = models.LearnedDecoder(self.args.latent_space_size,
                                            resolution=resolution,
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
            log('[Detection] read decoder')

            self.model = decoder

        self.compute_images()

    def compute_images(self):
        """
        Compute images.
        """

        assert self.test_codes is not None

        num_batches = int(math.ceil(self.perturbations.shape[0] / self.args.batch_size))
        for b in range(num_batches):
            b_start = b * self.args.batch_size
            b_end = min((b + 1) * self.args.batch_size, self.perturbations.shape[0])

            batch_codes = common.torch.as_variable(self.test_codes[b_start: b_end], self.args.use_gpu)
            if isinstance(self.model, models.SelectiveDecoder):
                self.model.set_code(batch_codes)
            batch_perturbation = common.torch.as_variable(self.perturbations[b_start: b_end].astype(numpy.float32), self.args.use_gpu)
            perturbation_images = self.model(batch_perturbation)

            if b % 100:
                log('[Detection] %d' % b)

            perturbation_images = numpy.squeeze(perturbation_images.cpu().detach().numpy())
            self.perturbation_images = common.numpy.concatenate(self.perturbation_images, perturbation_images)

        # Trick to perform analysis on actual images of adversarial examples.
        self.perturbations = self.perturbation_images


if __name__ == '__main__':
    program = DetectAttackLearnedDecoderClassifierNN()
    program.main()
