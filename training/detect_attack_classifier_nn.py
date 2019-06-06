import os
import sys

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
from common.log import log, logw, LogLevel
from common import paths
from common.state import State
from common import cuda
from common.scheduler import ADAMScheduler
import common.torch
import common.numpy
from common.ppca import PPCA
import models
import math
import torch
import numpy
import argparse
import sklearn.decomposition
import sklearn.neighbors
if utils.display():
    from common import plot
    from common import latex
    from common import vis


class DetectAttackClassifierNN:
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

        self.train_images = None
        """ (numpy.ndarray) Images to train on. """

        self.test_images = None
        """ (numpy.ndarray) Images to test. """

        self.train_codes = None
        """ (numpy.ndarray) Codes to train on. """

        self.test_codes = None
        """ (numpy.ndarray) Codes to test. """

        self.test_theta = None
        """ (str) Test theta. """

        self.perturbations = None
        """ (numpy.ndarray) Perturbations per test image. """

        self.success = None
        """ (numpy.ndarray) Detection indicator. """

        self.accuracy = None
        """ (numpy.ndarray) Accuracy indicator. """

        self.pca = None
        """ (sklearn.decompositions.IncrementalPCA) PCA model. """

        self.neighbors = None
        """ (sklearn.neighbors.NearestNeighbors) NN model. """

        self.angles = dict()
        """ (dict) Angles. """

        self.distances = dict()
        """ (dict) Distances. """

        self.probabilities = dict()
        """ (dict) Probabilities. """

        self.projected_perturbations = None
        """ (numpy.ndarray) Projected perturbations. """

        self.projected_test_images = None
        """ (numpy.ndarray) Projected test images. """

        log('-- ' + self.__class__.__name__)
        for key in vars(self.args):
            log('[Detection] %s=%s' % (key, str(getattr(self.args, key))))

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Detect attacks on classifier.')
        parser.add_argument('-mode', default='svd', help='Mode.', type=str)
        parser.add_argument('-database_file', default=paths.database_file(), help='HDF5 file containing font prototype images.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file codes dataset.', type=str)
        parser.add_argument('-test_theta_file', default=paths.test_theta_file(), help='HDF5 file containing transformations.', type=str)
        parser.add_argument('-train_images_file', default=paths.train_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-perturbations_file', default=paths.results_file('classifier/perturbations'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-success_file', default=paths.results_file('classifier/success'), help='HDF5 file containing success indicators.', type=str)
        parser.add_argument('-accuracy_file', default=paths.results_file('classifier/accuracy'), help='HDF5 file containing accuracy indicators.', type=str)
        parser.add_argument('-batch_size', default=64, help='Batch size.', type=int)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')
        parser.add_argument('-pre_pca', default=20, help='PCA dimensionality reduction ebfore NN.', type=int)
        parser.add_argument('-n_nearest_neighbors', default=50, help='Number of NNs to consider.', type=int)
        parser.add_argument('-n_pca', default=10, help='Number of NNs to consider.', type=int)
        parser.add_argument('-n_fit', default=100000, help='Training images to fit.', type=int)
        parser.add_argument('-plot_directory', default=paths.experiment_dir('classifier/detection'), help='Plot directory.', type=str)
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

        with logw('[Detection] read %s' % self.args.perturbations_file):
            self.perturbations = utils.read_hdf5(self.args.perturbations_file)
            assert len(self.perturbations.shape) == 4

        with logw('[Detection] read %s' % self.args.success_file):
            self.success = utils.read_hdf5(self.args.success_file)

        with logw('[Detection] read %s' % self.args.accuracy_file):
            self.accuracy = utils.read_hdf5(self.args.accuracy_file)

        self.perturbations = numpy.swapaxes(self.perturbations, 0, 1)
        num_attempts = self.perturbations.shape[1]
        self.test_images = self.test_images[:self.perturbations.shape[0]]
        self.train_images = self.nearest_neighbor_images[:self.perturbations.shape[0]]
        self.accuracy = self.accuracy[:self.perturbations.shape[0]]

        self.perturbations = self.perturbations.reshape((self.perturbations.shape[0]*self.perturbations.shape[1], self.perturbations.shape[2], self.perturbations.shape[3]))
        self.success = numpy.swapaxes(self.success, 0, 1)
        self.success = self.success.reshape((self.success.shape[0]*self.success.shape[1]))

        self.accuracy = numpy.repeat(self.accuracy, num_attempts, axis=0)
        self.test_images = numpy.repeat(self.test_images, num_attempts, axis=0)
        self.train_images = numpy.repeat(self.train_images, num_attempts, axis=0)

        max_samples = self.args.max_samples
        self.success = self.success[:max_samples]
        self.accuracy = self.accuracy[:max_samples]
        self.perturbations = self.perturbations[:max_samples]
        self.test_images = self.test_images[:max_samples]
        self.train_images = self.train_images[:max_samples]

        if self.args.mode == 'true':
            assert self.args.database_file
            assert self.args.test_codes_file
            assert self.args.test_theta_file

            self.test_codes = utils.read_hdf5(self.args.test_codes_file)
            log('[Detection] read %s' % self.args.test_codes_file)

            self.test_theta = utils.read_hdf5(self.args.test_theta_file)
            log('[Detection] read %s' % self.args.test_theta_file)

            self.test_codes = self.test_codes[:self.perturbations.shape[0]]
            self.test_theta = self.test_theta[:self.perturbations.shape[0]]

            self.test_codes = numpy.repeat(self.test_codes, num_attempts, axis=0)
            self.test_theta = numpy.repeat(self.test_theta, num_attempts, axis=0)

            self.test_codes = self.test_codes[:max_samples]
            self.test_theta = self.test_theta[:max_samples]

            database = utils.read_hdf5(self.args.database_file)
            log('[Detection] read %s' % self.args.database_file)

            self.N_font = database.shape[0]
            self.N_class = database.shape[1]
            self.N_theta = self.test_theta.shape[1]

            database = database.reshape((database.shape[0]*database.shape[1], database.shape[2], database.shape[3]))
            database = torch.from_numpy(database)
            if self.args.use_gpu:
                database = database.cuda()
            database = torch.autograd.Variable(database, False)

            self.model = models.AlternativeOneHotDecoder(database, self.N_font, self.N_class, self.N_theta)
            self.model.eval()
            log('[Detection] initialized decoder')
        elif self.args.mode == 'appr':
            assert self.args.decoder_files
            assert self.args.test_codes_file
            assert self.args.test_theta_file

            self.test_codes = utils.read_hdf5(self.args.test_codes_file)
            log('[Detection] read %s' % self.args.test_codes_file)

            self.test_theta = utils.read_hdf5(self.args.test_theta_file)
            log('[Detection] read %s' % self.args.test_theta_file)

            self.test_codes = self.test_codes[:self.perturbations.shape[0]]
            self.test_theta = self.test_theta[:self.perturbations.shape[0]]

            self.test_codes = numpy.repeat(self.test_codes, num_attempts, axis=0)
            self.test_theta = numpy.repeat(self.test_theta, num_attempts, axis=0)

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

    def compute_nearest_neighbors(self, images):
        """
        Compute distances in image and latent space.

        :param images: images to get nearest neighbors for
        :type images: numpy.ndarray
        :param n_nearest_neighbors: number of nearest neighbors
        :type n_nearest_neighbors: int
        """

        fit = self.args.n_fit
        if self.pca is None:
            nearest_neighbor_images = self.nearest_neighbor_images[:fit]
            nearest_neighbor_images = nearest_neighbor_images.reshape((nearest_neighbor_images.shape[0], -1))
            self.pca = sklearn.decomposition.IncrementalPCA(n_components=self.args.pre_pca)
            self.pca.fit(nearest_neighbor_images)
            log('[Detection] fitted PCA')
        if self.neighbors is None:
            nearest_neighbor_images = self.nearest_neighbor_images[:fit]
            nearest_neighbor_images = nearest_neighbor_images.reshape((nearest_neighbor_images.shape[0], -1))
            data = self.pca.transform(nearest_neighbor_images)
            self.neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=self.args.n_nearest_neighbors + 1, algorithm='kd_tree')
            self.neighbors.fit(data)
            log('[Detection] fitted nearest neighbor')

        data = self.pca.transform(images)
        _, indices = self.neighbors.kneighbors(data)
        return indices

    def compute_true(self):
        """
        Compute true.
        """

        assert self.test_codes is not None
        num_batches = int(math.ceil(self.perturbations.shape[0] / self.args.batch_size))

        params = {
            'lr': 0.09,
            'lr_decay': 0.95,
            'lr_min': 0.0000001,
            'weight_decay': 0,
        }

        for b in range(num_batches):
            b_start = b * self.args.batch_size
            b_end = min((b + 1) * self.args.batch_size, self.perturbations.shape[0])

            batch_fonts = self.test_codes[b_start: b_end, 1]
            batch_classes = self.test_codes[b_start: b_end, 2]
            batch_code = numpy.concatenate((common.numpy.one_hot(batch_fonts, self.N_font), common.numpy.one_hot(batch_classes, self.N_class)), axis=1).astype( numpy.float32)
            batch_code = common.torch.as_variable(batch_code, self.args.use_gpu)

            batch_images = common.torch.as_variable(self.test_images[b_start: b_end], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            batch_theta = common.torch.as_variable(self.test_theta[b_start: b_end].astype(numpy.float32), self.args.use_gpu, True)
            batch_perturbation = common.torch.as_variable(self.perturbations[b_start: b_end].astype(numpy.float32), self.args.use_gpu)

            self.model.set_code(batch_code)

            #output_images = self.model.forward(batch_theta)
            #test_error = torch.mean(torch.mul(output_images - batch_images, output_images - batch_images))
            #print(test_error.item())
            #vis.mosaic('true.png', batch_images.cpu().detach().numpy()[:, 0, :, :])
            #vis.mosaic('output.png', output_images.cpu().detach().numpy()[:, 0, :, :])
            # print(batch_images.cpu().detach().numpy()[0])
            # print(output_images.cpu().detach().numpy()[0, 0])

            #_batch_images = batch_images.cpu().detach().numpy()
            #_output_images = output_images.cpu().detach().numpy()[:, 0, :, :]
            #test_error = numpy.max(numpy.abs(_batch_images.reshape(_batch_images.shape[0], -1) - _output_images.reshape(_output_images.shape[0], -1)), axis=1)
            #print(test_error)
            #test_error = numpy.mean(numpy.multiply(_batch_images - _output_images, _batch_images - _output_images), axis=1)
            #print(test_error)

            batch_theta = torch.nn.Parameter(batch_theta)
            scheduler = ADAMScheduler([batch_theta], **params)

            log('[Detection] %d: start' % b)
            for t in range(100):
                scheduler.update(t//10, float(t)/10)
                scheduler.optimizer.zero_grad()
                output_perturbation = self.model.forward(batch_theta)
                error = torch.mean(torch.mul(output_perturbation - batch_perturbation, output_perturbation - batch_perturbation))
                test_error = torch.mean(torch.mul(output_perturbation - batch_images, output_perturbation - batch_images))
                #error.backward()
                #scheduler.optimizer.step()

                log('[Detection] %d: %d = %g, %g' % (b, t, error.item(), test_error.item()))

                output_perturbation = numpy.squeeze(numpy.transpose(output_perturbation.cpu().detach().numpy(), (0, 2, 3, 1)))
            self.projected_perturbations = common.numpy.concatenate(self.projected_perturbations, output_perturbation)

        projected_perturbations = self.projected_perturbations.reshape((self.projected_perturbations.shape[0], -1))
        perturbations = self.perturbations.reshape((self.perturbations.shape[0], -1))

        success = numpy.logical_and(self.success >= 0, self.accuracy)
        log('[Detection] %d valid attacked samples' % numpy.sum(success))

        self.distances['true'] = numpy.linalg.norm(perturbations - projected_perturbations, ord=2, axis=1)
        self.angles['true'] = numpy.rad2deg(common.numpy.angles(perturbations.T, projected_perturbations.T))

        self.distances['true'] = self.distances['true'][success]
        self.angles['true'] = self.angles['true'][success]

        self.distances['test'] = numpy.zeros((numpy.sum(success)))
        self.angles['test'] = numpy.zeros((numpy.sum(success)))

    def compute_appr(self):
        """
        Compute approximate.
        """

        assert self.test_codes is not None
        num_batches = int(math.ceil(self.perturbations.shape[0] / self.args.batch_size))

        for b in range(num_batches):
            b_start = b * self.args.batch_size
            b_end = min((b + 1) * self.args.batch_size, self.perturbations.shape[0])

            batch_classes = common.torch.as_variable(self.test_codes[b_start: b_end], self.args.use_gpu)
            batch_theta = common.torch.as_variable(self.test_theta[b_start: b_end].astype(numpy.float32), self.args.use_gpu, True)
            batch_perturbation = common.torch.as_variable(self.perturbations[b_start: b_end].astype(numpy.float32), self.args.use_gpu)

            if isinstance(self.model, models.SelectiveDecoder):
                self.model.set_code(batch_classes)
            batch_theta = torch.nn.Parameter(batch_theta)
            optimizer = torch.optim.Adam([batch_theta], lr=0.1)

            log('[Detection] %d: start' % b)
            for t in range(100):
                optimizer.zero_grad()
                output_perturbation = self.model.forward(batch_theta)
                error = torch.mean(torch.mul(output_perturbation - batch_perturbation, output_perturbation - batch_perturbation))
                error.backward()
                optimizer.step()

                log('[Detection] %d: %d = %g' % (b, t, error.item()))

            output_perturbation = numpy.squeeze(output_perturbation.cpu().detach().numpy())
            self.projected_perturbations = common.numpy.concatenate(self.projected_perturbations, output_perturbation)

            batch_theta = common.torch.as_variable(self.test_theta[b_start: b_end].astype(numpy.float32), self.args.use_gpu, True)
            batch_images = common.torch.as_variable(self.test_images[b_start: b_end].astype(numpy.float32), self.args.use_gpu)

            batch_theta = torch.nn.Parameter(batch_theta)
            optimizer = torch.optim.Adam([batch_theta], lr=0.5)

            log('[Detection] %d: start' % b)
            for t in range(100):
                optimizer.zero_grad()
                output_images = self.model.forward(batch_theta)
                error = torch.mean(torch.mul(output_images - batch_images, output_images - batch_images))
                error.backward()
                optimizer.step()

                log('[Detection] %d: %d = %g' % (b, t, error.item()))

            output_images = numpy.squeeze(output_images.cpu().detach().numpy())
            self.projected_test_images = common.numpy.concatenate(self.projected_test_images, output_images)

        projected_perturbations = self.projected_perturbations.reshape((self.projected_perturbations.shape[0], -1))
        projected_test_images = self.projected_test_images.reshape((self.projected_test_images.shape[0], -1))

        perturbations = self.perturbations.reshape((self.perturbations.shape[0], -1))
        test_images = self.test_images.reshape((self.test_images.shape[0], -1))

        success = numpy.logical_and(self.success >= 0, self.accuracy)
        log('[Detection] %d valid attacked samples' % numpy.sum(success))

        self.distances['true'] = numpy.linalg.norm(perturbations - projected_perturbations, ord=2, axis=1)
        self.angles['true'] = numpy.rad2deg(common.numpy.angles(perturbations.T, projected_perturbations.T))

        self.distances['true'] = self.distances['true'][success]
        self.angles['true'] = self.angles['true'][success]

        self.distances['test'] = numpy.linalg.norm(test_images - projected_test_images, ord=2, axis=1)
        self.angles['test'] = numpy.rad2deg(common.numpy.angles(test_images.T, projected_test_images.T))

        self.distances['test'] = self.distances['test'][success]
        self.angles['test'] = self.angles['test'][success]

    def compute_pca(self):
        """
        Compute PCA.
        """

        success = numpy.logical_and(self.success >= 0, self.accuracy)
        log('[Detection] %d valid attacked samples' % numpy.sum(success))

        nearest_neighbor_images = self.nearest_neighbor_images.reshape(self.nearest_neighbor_images.shape[0], -1)
        nearest_neighbor_images = nearest_neighbor_images[:self.args.n_fit]
        
        perturbations = self.perturbations.reshape(self.perturbations.shape[0], -1)
        test_images = self.test_images.reshape(self.test_images.shape[0], -1)
        pure_perturbations = perturbations - test_images

        pca = sklearn.decomposition.IncrementalPCA(n_components=self.args.n_pca)
        pca.fit(nearest_neighbor_images)

        reconstructed_test_images = pca.inverse_transform(pca.transform(test_images))
        reconstructed_perturbations = pca.inverse_transform(pca.transform(perturbations))
        reconstructed_pure_perturbations = pca.inverse_transform(pca.transform(pure_perturbations))

        self.distances['test'] = numpy.average(numpy.multiply(reconstructed_test_images - test_images, reconstructed_test_images - test_images), axis=1)
        self.distances['perturbation'] = numpy.average(numpy.multiply(reconstructed_perturbations - perturbations, reconstructed_perturbations - perturbations), axis=1)
        self.distances['true'] = numpy.average(numpy.multiply(reconstructed_pure_perturbations - pure_perturbations, reconstructed_pure_perturbations - pure_perturbations), axis=1)

        self.distances['l2'] = numpy.average(numpy.multiply(pure_perturbations, pure_perturbations), axis=1)
        self.distances['projection'] = numpy.average(numpy.multiply(pure_perturbations, pure_perturbations), axis=1)

        self.angles['test'] = numpy.rad2deg(common.numpy.angles(reconstructed_test_images.T, test_images.T))
        self.angles['perturbation'] = numpy.rad2deg(common.numpy.angles(reconstructed_perturbations.T, perturbations.T))
        self.angles['true'] = numpy.rad2deg(common.numpy.angles(reconstructed_pure_perturbations.T, pure_perturbations.T))

        self.distances['test'] = self.distances['test'][success]
        self.distances['perturbation'] = self.distances['perturbation'][success]
        self.distances['true'] = self.distances['true'][success]
        self.distances['l2'] = self.distances['l2'][success]

    def compute_local_pca(self):
        """
        Compute PCA.
        """

        success = numpy.logical_and(self.success >= 0, self.accuracy)
        log('[Detection] %d valid attacked samples' % numpy.sum(success))

        nearest_neighbor_images = self.nearest_neighbor_images.reshape(self.nearest_neighbor_images.shape[0], -1)
        nearest_neighbor_images = nearest_neighbor_images[:self.args.n_fit]

        perturbations = self.perturbations.reshape(self.perturbations.shape[0], -1)
        test_images = self.test_images.reshape(self.test_images.shape[0], -1)
        pure_perturbations = perturbations - test_images

        nearest_neighbors_indices = self.compute_nearest_neighbors(perturbations)

        self.distances['true'] = numpy.zeros((success.shape[0]))
        self.distances['test'] = numpy.zeros((success.shape[0]))
        self.distances['perturbation'] = numpy.zeros((success.shape[0]))

        self.angles['true'] = numpy.zeros((success.shape[0]))
        self.angles['test'] = numpy.zeros((success.shape[0]))
        self.angles['perturbation'] = numpy.zeros((success.shape[0]))

        for n in range(pure_perturbations.shape[0]):
            if success[n]:
                nearest_neighbors = nearest_neighbor_images[nearest_neighbors_indices[n, :]]
                nearest_neighbors = numpy.concatenate((nearest_neighbors, test_images[n].reshape(1, -1)), axis=0)

                pca = sklearn.decomposition.IncrementalPCA(n_components=self.args.n_pca)
                pca.fit(nearest_neighbors)

                reconstructed_test_images = pca.inverse_transform(pca.transform(test_images[n].reshape(1, -1)))
                reconstructed_perturbations = pca.inverse_transform(pca.transform(perturbations[n].reshape(1, -1)))
                reconstructed_pure_perturbations = pca.inverse_transform(pca.transform(pure_perturbations[n].reshape(1, -1)))

                self.distances['test'][n] = numpy.average(numpy.multiply(reconstructed_test_images - test_images[n], reconstructed_test_images - test_images[n]), axis=1)
                self.distances['perturbation'][n] = numpy.average(numpy.multiply(reconstructed_perturbations - perturbations[n], reconstructed_perturbations - perturbations[n]), axis=1)
                self.distances['true'][n] = numpy.average(numpy.multiply(reconstructed_pure_perturbations - pure_perturbations[n], reconstructed_pure_perturbations - pure_perturbations[n]), axis=1)

                self.angles['test'][n] = numpy.rad2deg(common.numpy.angles(reconstructed_test_images.T, test_images[n].T))
                self.angles['perturbation'][n] = numpy.rad2deg(common.numpy.angles(reconstructed_perturbations.T, perturbations[n].T))
                self.angles['true'][n] = numpy.rad2deg(common.numpy.angles(reconstructed_pure_perturbations.T, pure_perturbations[n].T))

                log('[Detection] %d: true distance=%g angle=%g' % (n, self.distances['true'][n], self.angles['true'][n]))
                log('[Detection] %d: perturbation distance=%g angle=%g' % (n, self.distances['perturbation'][n], self.angles['perturbation'][n]))
                log('[Detection] %d: test distance=%g angle=%g' % (n, self.distances['test'][n], self.angles['test'][n]))

        self.distances['test'] = self.distances['test'][success]
        self.distances['perturbation'] = self.distances['perturbation'][success]
        self.distances['true'] = self.distances['true'][success]

    def compute_normalized_pca(self):
        """
        Compute PCA.
        """

        nearest_neighbor_images = self.nearest_neighbor_images.reshape(self.nearest_neighbor_images.shape[0], -1)
        nearest_neighbor_images = nearest_neighbor_images[:self.args.n_fit]

        perturbations = self.perturbations.reshape(self.perturbations.shape[0], -1)
        test_images = self.test_images.reshape(self.test_images.shape[0], -1)
        pure_perturbations = perturbations - test_images

        nearest_neighbor_images_norms = numpy.linalg.norm(nearest_neighbor_images, ord=2, axis=1)
        perturbations_norms = numpy.linalg.norm(perturbations, ord=2, axis=1)
        test_images_norms = numpy.linalg.norm(test_images, ord=2, axis=1)
        pure_perturbations_norms = numpy.linalg.norm(pure_perturbations, ord=2, axis=1)

        success = numpy.logical_and(numpy.logical_and(self.success >= 0, self.accuracy), pure_perturbations_norms > 1e-4)
        log('[Detection] %d valid attacked samples' % numpy.sum(success))

        perturbations_norms = perturbations_norms[success]
        test_images_norms = test_images_norms[success]
        pure_perturbations_norms = pure_perturbations_norms[success]

        perturbations = perturbations[success]
        test_images = test_images[success]
        pure_perturbations = pure_perturbations[success]

        nearest_neighbor_images /= numpy.repeat(nearest_neighbor_images_norms.reshape(-1, 1), nearest_neighbor_images.shape[1], axis=1)
        perturbations /= numpy.repeat(perturbations_norms.reshape(-1, 1), perturbations.shape[1], axis=1)
        test_images /= numpy.repeat(test_images_norms.reshape(-1, 1), test_images.shape[1], axis=1)
        pure_perturbations /= numpy.repeat(pure_perturbations_norms.reshape(-1, 1), pure_perturbations.shape[1], axis=1)

        assert not numpy.any(nearest_neighbor_images != nearest_neighbor_images)
        assert not numpy.any(perturbations != perturbations)
        assert not numpy.any(test_images != test_images)
        assert not numpy.any(pure_perturbations != pure_perturbations)

        pca = sklearn.decomposition.IncrementalPCA(n_components=self.args.n_pca)
        pca.fit(nearest_neighbor_images)

        reconstructed_test_images = pca.inverse_transform(pca.transform(test_images))
        reconstructed_perturbations = pca.inverse_transform(pca.transform(perturbations))
        reconstructed_pure_perturbations = pca.inverse_transform(pca.transform(pure_perturbations))

        self.distances['test'] = numpy.average(numpy.multiply(reconstructed_test_images - test_images, reconstructed_test_images - test_images), axis=1)
        self.distances['perturbation'] = numpy.average(numpy.multiply(reconstructed_perturbations - perturbations, reconstructed_perturbations - perturbations), axis=1)
        self.distances['true'] = numpy.average(numpy.multiply(reconstructed_pure_perturbations - pure_perturbations, reconstructed_pure_perturbations - pure_perturbations), axis=1)

        self.angles['test'] = numpy.rad2deg(common.numpy.angles(reconstructed_test_images.T, test_images.T))
        self.angles['perturbation'] = numpy.rad2deg(common.numpy.angles(reconstructed_perturbations.T, perturbations.T))
        self.angles['true'] = numpy.rad2deg(common.numpy.angles(reconstructed_pure_perturbations.T, pure_perturbations.T))

    def compute_local_normalized_pca(self, inclusive):
        """
        Compute PCA.
        """

        success = numpy.logical_and(self.success >= 0, self.accuracy)
        log('[Detection] %d valid attacked samples' % numpy.sum(success))

        nearest_neighbor_images = self.nearest_neighbor_images.reshape(self.nearest_neighbor_images.shape[0], -1)
        nearest_neighbor_images = nearest_neighbor_images[:self.args.n_fit]

        perturbations = self.perturbations.reshape(self.perturbations.shape[0], -1)
        test_images = self.test_images.reshape(self.test_images.shape[0], -1)
        pure_perturbations = perturbations - test_images

        nearest_neighbor_images_norms = numpy.linalg.norm(nearest_neighbor_images, ord=2, axis=1)
        perturbations_norms = numpy.linalg.norm(perturbations, ord=2, axis=1)
        test_images_norms = numpy.linalg.norm(test_images, ord=2, axis=1)
        pure_perturbations_norms = numpy.linalg.norm(pure_perturbations, ord=2, axis=1)

        success = numpy.logical_and(numpy.logical_and(self.success >= 0, self.accuracy), pure_perturbations_norms > 1e-4)
        log('[Detection] %d valid attacked samples' % numpy.sum(success))

        perturbations_norms = perturbations_norms[success]
        test_images_norms = test_images_norms[success]
        pure_perturbations_norms = pure_perturbations_norms[success]

        perturbations = perturbations[success]
        test_images = test_images[success]
        pure_perturbations = pure_perturbations[success]

        nearest_neighbor_images /= numpy.repeat(nearest_neighbor_images_norms.reshape(-1, 1), nearest_neighbor_images.shape[1], axis=1)
        perturbations /= numpy.repeat(perturbations_norms.reshape(-1, 1), perturbations.shape[1], axis=1)
        test_images /= numpy.repeat(test_images_norms.reshape(-1, 1), test_images.shape[1], axis=1)
        pure_perturbations /= numpy.repeat(pure_perturbations_norms.reshape(-1, 1), pure_perturbations.shape[1], axis=1)

        nearest_neighbors_indices = self.compute_nearest_neighbors(perturbations)

        self.distances['true'] = numpy.zeros((numpy.sum(success)))
        self.distances['test'] = numpy.zeros((numpy.sum(success)))
        self.distances['perturbation'] = numpy.zeros((numpy.sum(success)))

        self.angles['true'] = numpy.zeros((numpy.sum(success)))
        self.angles['test'] = numpy.zeros((numpy.sum(success)))
        self.angles['perturbation'] = numpy.zeros((numpy.sum(success)))

        for n in range(pure_perturbations.shape[0]):
            nearest_neighbors = nearest_neighbor_images[nearest_neighbors_indices[n, :]]

            if inclusive:
                nearest_neighbors = numpy.concatenate((nearest_neighbors, test_images[n].reshape(1, -1)), axis=0)

            pca = sklearn.decomposition.IncrementalPCA(n_components=self.args.n_pca)
            pca.fit(nearest_neighbors)

            reconstructed_test_images = pca.inverse_transform(pca.transform(test_images[n].reshape(1, -1)))
            reconstructed_perturbations = pca.inverse_transform(pca.transform(perturbations[n].reshape(1, -1)))
            reconstructed_pure_perturbations = pca.inverse_transform(pca.transform(pure_perturbations[n].reshape(1, -1)))

            self.distances['test'][n] = numpy.average(numpy.multiply(reconstructed_test_images - test_images[n], reconstructed_test_images - test_images[n]), axis=1)
            self.distances['perturbation'][n] = numpy.average(numpy.multiply(reconstructed_perturbations - perturbations[n], reconstructed_perturbations - perturbations[n]), axis=1)
            self.distances['true'][n] = numpy.average(numpy.multiply(reconstructed_pure_perturbations - pure_perturbations[n], reconstructed_pure_perturbations - pure_perturbations[n]), axis=1)

            self.angles['test'][n] = numpy.rad2deg(common.numpy.angles(reconstructed_test_images.T, test_images[n].T))
            self.angles['perturbation'][n] = numpy.rad2deg(common.numpy.angles(reconstructed_perturbations.T, perturbations[n].T))
            self.angles['true'][n] = numpy.rad2deg(common.numpy.angles(reconstructed_pure_perturbations.T, pure_perturbations[n].T))

            log('[Detection] %d: true distance=%g angle=%g' % (n, self.distances['true'][n], self.angles['true'][n]))
            log('[Detection] %d: perturbation distance=%g angle=%g' % (n, self.distances['perturbation'][n], self.angles['perturbation'][n]))
            log('[Detection] %d: test distance=%g angle=%g' % (n, self.distances['test'][n], self.angles['test'][n]))

    def compute_ppca(self):
        """
        Compute PPCA.
        """

        success = numpy.logical_and(self.success >= 0, self.accuracy)
        log('[Detection] %d valid attacked samples' % numpy.sum(success))

        nearest_neighbor_images = self.nearest_neighbor_images.reshape(self.nearest_neighbor_images.shape[0], -1)
        nearest_neighbor_images = nearest_neighbor_images[:self.args.n_fit]
        
        perturbations = self.perturbations.reshape(self.perturbations.shape[0], -1)
        test_images = self.test_images.reshape(self.test_images.shape[0], -1)
        pure_perturbations = perturbations - test_images

        ppca = PPCA(n_components=self.args.n_pca)
        ppca.fit(nearest_neighbor_images)
        log('[Experiment] computed PPCA on nearest neighbor images')

        reconstructed_test_images = ppca.inverse_transform(ppca.transform(test_images))
        reconstructed_perturbations = ppca.inverse_transform(ppca.transform(perturbations))
        reconstructed_pure_perturbations = ppca.inverse_transform(ppca.transform(pure_perturbations))

        self.distances['test'] = numpy.average(numpy.multiply(reconstructed_test_images - test_images, reconstructed_test_images - test_images), axis=1)
        self.distances['perturbation'] = numpy.average(numpy.multiply(reconstructed_perturbations - perturbations, reconstructed_perturbations - perturbations), axis=1)
        self.distances['true'] = numpy.average(numpy.multiply(reconstructed_pure_perturbations - pure_perturbations, reconstructed_pure_perturbations - pure_perturbations), axis=1)

        self.angles['test'] = numpy.rad2deg(common.numpy.angles(test_images.T, reconstructed_test_images.T))
        self.angles['perturbation'] = numpy.rad2deg(common.numpy.angles(reconstructed_perturbations.T, perturbations.T))
        self.angles['true'] = numpy.rad2deg(common.numpy.angles(reconstructed_pure_perturbations.T, pure_perturbations.T))

        self.distances['test'] = self.distances['test'][success]
        self.distances['perturbation'] = self.distances['perturbation'][success]
        self.distances['true'] = self.distances['true'][success]

    def compute_normalized_ppca(self):
        """
        Compute PPCA.
        """

        nearest_neighbor_images = self.nearest_neighbor_images.reshape(self.nearest_neighbor_images.shape[0], -1)
        nearest_neighbor_images = nearest_neighbor_images[:self.args.n_fit]

        perturbations = self.perturbations.reshape(self.perturbations.shape[0], -1)
        test_images = self.test_images.reshape(self.test_images.shape[0], -1)
        pure_perturbations = perturbations - test_images

        nearest_neighbor_images_norms = numpy.linalg.norm(nearest_neighbor_images, ord=2, axis=1)
        perturbations_norms = numpy.linalg.norm(perturbations, ord=2, axis=1)
        test_images_norms = numpy.linalg.norm(test_images, ord=2, axis=1)
        pure_perturbations_norms = numpy.linalg.norm(pure_perturbations, ord=2, axis=1)

        success = numpy.logical_and(numpy.logical_and(self.success >= 0, self.accuracy), pure_perturbations_norms > 1e-4)
        log('[Detection] %d valid attacked samples' % numpy.sum(success))

        perturbations_norms = perturbations_norms[success]
        test_images_norms = test_images_norms[success]
        pure_perturbations_norms = pure_perturbations_norms[success]

        perturbations = perturbations[success]
        test_images = test_images[success]
        pure_perturbations = pure_perturbations[success]

        nearest_neighbor_images /= numpy.repeat(nearest_neighbor_images_norms.reshape(-1, 1), nearest_neighbor_images.shape[1], axis=1)
        perturbations /= numpy.repeat(perturbations_norms.reshape(-1, 1), perturbations.shape[1], axis=1)
        test_images /= numpy.repeat(test_images_norms.reshape(-1, 1), test_images.shape[1], axis=1)
        pure_perturbations /= numpy.repeat(pure_perturbations_norms.reshape(-1, 1), pure_perturbations.shape[1], axis=1)

        assert not numpy.any(nearest_neighbor_images != nearest_neighbor_images)
        assert not numpy.any(perturbations != perturbations)
        assert not numpy.any(test_images != test_images)
        assert not numpy.any(pure_perturbations != pure_perturbations)

        ppca = PPCA(n_components=self.args.n_pca)
        ppca.fit(nearest_neighbor_images)
        log('[Experiment] computed PPCA on nearest neighbor images')

        reconstructed_test_images = ppca.inverse_transform(ppca.transform(test_images))
        reconstructed_perturbations = ppca.inverse_transform(ppca.transform(perturbations))
        reconstructed_pure_perturbations = ppca.inverse_transform(ppca.transform(pure_perturbations))
        
        #self.probabilities['test'] = ppca.marginal(test_images)
        #self.probabilities['perturbation'] = ppca.marginal(perturbations)
        #self.probabilities['true'] = ppca.marginal(pure_perturbations)

        self.distances['test'] = numpy.average(numpy.multiply(reconstructed_test_images - test_images, reconstructed_test_images - test_images), axis=1)
        self.distances['perturbation'] = numpy.average(numpy.multiply(reconstructed_perturbations - perturbations, reconstructed_perturbations - perturbations), axis=1)
        self.distances['true'] = numpy.average(numpy.multiply(reconstructed_pure_perturbations - pure_perturbations, reconstructed_pure_perturbations - pure_perturbations), axis=1)

        self.angles['test'] = numpy.rad2deg(common.numpy.angles(test_images.T, reconstructed_test_images.T))
        self.angles['perturbation'] = numpy.rad2deg(common.numpy.angles(reconstructed_perturbations.T, perturbations.T))
        self.angles['true'] = numpy.rad2deg(common.numpy.angles(reconstructed_pure_perturbations.T, pure_perturbations.T))

    def compute_nn(self, inclusive=False):
        """
        Test detector.
        """

        success = numpy.logical_and(self.success >= 0, self.accuracy)
        log('[Detection] %d valid attacked samples' % numpy.sum(success))

        nearest_neighbor_images = self.nearest_neighbor_images.reshape(self.nearest_neighbor_images.shape[0], -1)
        perturbations = self.perturbations.reshape(self.perturbations.shape[0], -1)
        test_images = self.test_images.reshape(self.test_images.shape[0], -1)

        nearest_neighbors_indices = self.compute_nearest_neighbors(perturbations)
        pure_perturbations = perturbations - test_images
        log('[Detection] computed nearest neighbors for perturbations')

        self.distances['true'] = numpy.zeros((success.shape[0]))
        self.distances['test'] = numpy.zeros((success.shape[0]))
        self.distances['perturbation'] = numpy.zeros((success.shape[0]))

        self.angles['true'] = numpy.zeros((success.shape[0]))
        self.angles['test'] = numpy.zeros((success.shape[0]))
        self.angles['perturbation'] = numpy.zeros((success.shape[0]))

        for n in range(pure_perturbations.shape[0]):
            if success[n]:
                nearest_neighbors = nearest_neighbor_images[nearest_neighbors_indices[n, :]]

                if inclusive:
                    nearest_neighbors = numpy.concatenate((nearest_neighbors, test_images[n].reshape(1, -1)), axis=0)
                    nearest_neighbor_mean = test_images[n]
                else:
                    nearest_neighbor_mean = numpy.average(nearest_neighbors, axis=0)

                nearest_neighbor_basis = nearest_neighbors - nearest_neighbor_mean

                relative_perturbation = perturbations[n] - nearest_neighbor_mean
                relative_test_image = test_images[n] - nearest_neighbor_mean

                if inclusive:
                    assert numpy.allclose(relative_test_image, nearest_neighbor_basis[-1])

                nearest_neighbor_vectors = numpy.stack((
                    pure_perturbations[n],
                    relative_perturbation,
                    relative_test_image
                ), axis=1)

                nearest_neighbor_projections = common.numpy.project_orthogonal(nearest_neighbor_basis.T, nearest_neighbor_vectors)
                assert nearest_neighbor_vectors.shape[0] == nearest_neighbor_projections.shape[0]
                assert nearest_neighbor_vectors.shape[1] == nearest_neighbor_projections.shape[1]

                angles = numpy.rad2deg(common.numpy.angles(nearest_neighbor_vectors, nearest_neighbor_projections))
                distances = numpy.linalg.norm(nearest_neighbor_vectors - nearest_neighbor_projections, ord=2, axis=0)

                assert distances.shape[0] == 3
                assert angles.shape[0] == 3

                self.distances['true'][n] = distances[0]
                self.distances['perturbation'][n] = distances[1]
                self.distances['test'][n] = distances[2]

                self.angles['true'][n] = angles[0]
                self.angles['perturbation'][n] = angles[1]
                self.angles['test'][n] = angles[2]

                log('[Detection] %d: true distance=%g angle=%g' % (n, self.distances['true'][n], self.angles['true'][n]))
                log('[Detection] %d: perturbation distance=%g angle=%g' % (n, self.distances['perturbation'][n], self.angles['perturbation'][n]))
                log('[Detection] %d: test distance=%g angle=%g' % (n, self.distances['test'][n], self.angles['test'][n]))

        self.distances['true'] = self.distances['true'][success]
        self.distances['test'] = self.distances['test'][success]
        self.distances['perturbation'] = self.distances['perturbation'][success]

        self.angles['true'] = self.angles['true'][success]
        self.angles['test'] = self.angles['test'][success]
        self.angles['perturbation'] = self.angles['perturbation'][success]

        if inclusive:
            self.distances['test'][:] = 0
            self.angles['test'][:] = 0

    def plot_histogram(self, plot_file, values):
        """
        Plot distances.
        """

        try:
            plot.histogram(plot_file, values, 100)
            log('[Detection] wrote %s' % plot_file)
        except ValueError as e:
            log('[Detection] plotting not possible: %s' % str(e), LogLevel.WARNING)

        latex.histogram(plot_file + '.tex', values, 100)
        log('[Detection] wrote %s.tex' % plot_file)

        latex.histogram(plot_file + '.normalized.tex', values, 100, None, True)
        log('[Detection] wrote %s.normalized.tex' % plot_file)

    def main(self):
        """
        Main.
        """

        self.load_data()
        if self.args.mode == 'true':
            self.compute_true()
        elif self.args.mode == 'appr':
            self.compute_appr()
        elif self.args.mode == 'nn':
            self.compute_nn(False)
        elif self.args.mode == 'inclusive_nn':
            self.compute_nn(True)
        elif self.args.mode == 'pca':
            self.compute_pca()
        elif self.args.mode == 'local_pca':
            self.compute_local_pca()
        elif self.args.mode == 'normalized_pca':
            self.compute_normalized_pca()
        elif self.args.mode == 'local_normalized_pca':
            self.compute_local_normalized_pca(False)
        elif self.args.mode == 'local_normalized_inclusive_pca':
            self.compute_local_normalized_pca(True)
        elif self.args.mode == 'ppca':
            self.compute_ppca()
        elif self.args.mode == 'normalized_ppca':
            self.compute_normalized_ppca()

        self.plot_histogram(os.path.join(self.args.plot_directory, '%s_%d_%d/true_test_distances.png' % (self.args.mode, self.args.n_nearest_neighbors, self.args.n_pca)), self.distances['true'] - self.distances['test'])
        if 'perturbation' in self.distances.keys():
            self.plot_histogram(os.path.join(self.args.plot_directory, '%s_%d_%d/perturbation_test_distances.png' % (self.args.mode, self.args.n_nearest_neighbors, self.args.n_pca)), self.distances['perturbation'] - self.distances['test'])
        for key in self.distances.keys():
            self.plot_histogram(os.path.join(self.args.plot_directory, '%s_%d_%d/%s_distances.png' % (self.args.mode, self.args.n_nearest_neighbors, self.args.n_pca, key)), self.distances[key])
        for key in self.angles.keys():
            self.plot_histogram(os.path.join(self.args.plot_directory, '%s_%d_%d/%s_angles.png' % (self.args.mode, self.args.n_nearest_neighbors, self.args.n_pca, key)), self.angles[key])
        for key in self.probabilities.keys():
            self.plot_histogram(os.path.join(self.args.plot_directory, '%s_%d_%d/%s_probabilities.png' % (self.args.mode, self.args.n_nearest_neighbors, self.args.n_pca, key)), self.probabilities[key])


if __name__ == '__main__':
    program = DetectAttackClassifierNN()
    program.main()
