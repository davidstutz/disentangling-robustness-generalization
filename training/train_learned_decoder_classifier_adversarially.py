import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
import models
from common.log import log, LogLevel
from common.state import State
from common import cuda
from common import paths
import common.torch
import common.numpy
from training import train_classifier_adversarially
import torch
import numpy
import argparse
import math


if utils.display():
    from common import plot


class TrainLearnedDecoderClassifierAdversarially(train_classifier_adversarially.TrainClassifierAdversarially):
    """
    Train a classifier.

    :param args: arguments
    :type args: list
    """

    def __init__(self, args=None):
        """
        Initialize.

        :param args: optional arguments if not to use sys.argv
        :type args: [str]
        """

        super(TrainLearnedDecoderClassifierAdversarially, self).__init__(args)

        self.train_statistics = numpy.zeros((0, 11))
        self.test_statistics = numpy.zeros((0, 10))

        self.train_theta = None
        """ (numpy.ndarray) Training transformation parameters. """

        self.test_theta = None
        """ (numpy.ndarray) Testing transformation parameters. """

        self.decoder = None
        """ (Decoder) Decoder. """

        self.decoder_classifier = None
        """ (DecoderClassifier) Model to attack. """

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Train classifier.')
        parser.add_argument('-train_images_file', default=paths.train_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-train_codes_file', default=paths.train_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-train_theta_file', default=paths.results_file('train_theta'), help='HDF5 file containing transformations.', type=str)
        parser.add_argument('-test_theta_file', default=paths.results_file('test_theta'), help='HDF5 file containing transformations.', type=str)
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-decoder_files', default=paths.state_file('decoder'), help='Decoder files.', type=str)
        parser.add_argument('-latent_space_size', default=10, help='Size of latent space.', type=int)
        parser.add_argument('-label_index', default=2, help='Column index in label file.', type=int)
        parser.add_argument('-state_file', default=paths.state_file('robust_manifold_classifier'), help='Snapshot state file.', type=str)
        parser.add_argument('-log_file', default=paths.log_file('robust_manifold_classifier'), help='Log file.', type=str)
        parser.add_argument('-training_file', default=paths.results_file('robust_manifold_training'), help='Training statistics file.', type=str)
        parser.add_argument('-testing_file', default=paths.results_file('robust_manifold_testing'), help='Testing statistics file.', type=str)
        parser.add_argument('-loss_file', default=paths.image_file('loss'), help='Loss plot file.', type=str)
        parser.add_argument('-error_file', default=paths.image_file('error'), help='Error plot file.', type=str)
        parser.add_argument('-success_file', default=paths.image_file('robust_success'), help='Success rate plot file.', type=str)
        parser.add_argument('-gradient_file', default='', help='Gradient plot file.', type=str)
        parser.add_argument('-random_samples', default=False, action='store_true', help='Randomize the subsampling of the training set.')
        parser.add_argument('-training_samples', default=-1, help='Number of samples used for training.', type=int)
        parser.add_argument('-test_samples', default=-1, help='Number of samples for validation.', type=int)
        parser.add_argument('-validation_samples', default=0, help='Number of samples for validation.', type=int)
        parser.add_argument('-early_stopping', default=False, action='store_true', help='Use early stopping.')
        parser.add_argument('-attack_samples', default=1000, help='Samples to attack.', type=int)
        parser.add_argument('-batch_size', default=64, help='Batch size.', type=int)
        parser.add_argument('-epochs', default=10, help='Number of epochs.', type=int)
        parser.add_argument('-weight_decay', default=0.0001, help='Weight decay importance.', type=float)
        parser.add_argument('-logit_decay', default=0, help='Logit decay importance.', type=float)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')
        parser.add_argument('-skip', default=5, help='Verbosity in iterations.', type=int)
        parser.add_argument('-lr', default=0.005, type=float, help='Base learning rate.')
        parser.add_argument('-lr_decay', default=0.9, type=float, help='Learning rate decay.')
        parser.add_argument('-results_file', default='', help='Results file for evaluation.', type=str)
        parser.add_argument('-bound', default=2, help='Bound used to define "safe" latent codes to compute adversarial examples on.', type=float)
        parser.add_argument('-debug_directory', default='', help='Debug directory.', type=str)

        # Some network parameters.
        parser.add_argument('-network_architecture', default='standard', help='Classifier architecture to use.', type=str)
        parser.add_argument('-network_activation', default='relu', help='Activation function to use.', type=str)
        parser.add_argument('-network_no_batch_normalization', default=False, help='Do not use batch normalization.', action='store_true')
        parser.add_argument('-network_channels', default=16, help='Channels of first convolutional layer, afterwards channels are doubled.', type=int)
        parser.add_argument('-network_dropout', default=False, action='store_true', help='Whether to use dropout.')
        parser.add_argument('-network_units', default='1024,1024,1024,1024', help='Units for MLP.')

        # Decoder parameters.
        parser.add_argument('-decoder_architecture', default='standard', help='Architecture to use.', type=str)
        parser.add_argument('-decoder_activation', default='relu', help='Activation function to use.', type=str)
        parser.add_argument('-decoder_no_batch_normalization', default=False, help='Do not use batch normalization.', action='store_true')
        parser.add_argument('-decoder_channels', default=16, help='Channels of first convolutional layer, afterwards channels are doubled.', type=int)
        parser.add_argument('-decoder_dropout', default=False, action='store_true', help='Whether to use dropout.')
        parser.add_argument('-decoder_units', default='1024,1024,1024,1024', help='Units for MLP.')

        # Attack parameters.
        parser.add_argument('-attack', default='UntargetedBatchL2ClippedGradientDescent', help='Attack to try.', type=str)
        parser.add_argument('-objective', default='UntargetedF6', help='Objective to use.', type=str)
        parser.add_argument('-epsilon', default=1, help='Epsilon allowed for attacks.', type=float)
        parser.add_argument('-c_0', default=0., help='Weight of norm.', type=float)
        parser.add_argument('-c_1', default=0.1, help='Weight of bound, if not enforced through clipping or reparameterization.', type=float)
        parser.add_argument('-c_2', default=0.5, help='Weight of objective.', type=float)
        parser.add_argument('-max_iterations', default=10, help='Number of iterations for attack.', type=int)
        parser.add_argument('-max_projections', default=5, help='Number of projections for alternating projection.', type=int)
        parser.add_argument('-base_lr', default=0.005, help='Learning rate for attack.', type=float)
        parser.add_argument('-verbose', action='store_true', default=False, help='Verbose attacks.')
        parser.add_argument('-anneal_epochs', default=0, help='Anneal iterations in the first epochs.', type=int)

        # Variants.
        parser.add_argument('-full_variant', default=False, action='store_true', help='100% variant.')
        parser.add_argument('-safe', default=False, action='store_true', help='Save variant.')
        parser.add_argument('-safe_bn', default=False, action='store_true', help='Save batch normalization variant.')
        parser.add_argument('-training_mode', default=False, action='store_true', help='Training mode variant for attack.')

        return parser

    def train(self):
        """
        Train adversarially.
        """

        split = self.args.batch_size // 2
        num_batches = int(math.ceil(self.train_images.shape[0] / self.args.batch_size))
        permutation = numpy.random.permutation(self.train_images.shape[0])
        perturbation_permutation = numpy.random.permutation(self.train_images.shape[0])
        if self.args.safe:
            perturbation_permutation = perturbation_permutation[self.train_valid == 1]
        else:
            perturbation_permuation = permutation

        for b in range(num_batches):
            self.scheduler.update(self.epoch, float(b) / num_batches)

            self.model.eval()
            assert self.model.training is False
            objective = self.objective_class()

            if self.args.full_variant:
                perm = numpy.take(perturbation_permutation, range(b*self.args.batch_size, (b+1)*self.args.batch_size), mode='wrap')
                batch_images = common.torch.as_variable(self.train_images[perm], self.args.use_gpu)
                batch_classes = common.torch.as_variable(self.train_codes[perm], self.args.use_gpu)
                batch_theta = common.torch.as_variable(self.train_theta[perm], self.args.use_gpu)
                batch_images = batch_images.permute(0, 3, 1, 2)

                if isinstance(self.decoder, models.SelectiveDecoder):
                    self.decoder.set_code(batch_classes)
                attack = self.setup_attack(self.decoder_classifier, batch_theta, batch_classes)
                attack.set_bound(torch.from_numpy(self.min_bound), torch.from_numpy(self.max_bound))
                success, perturbations, probabilities, norm, _ = attack.run(objective, self.args.verbose)

                perturbations = common.torch.as_variable(perturbations, self.args.use_gpu)
                batch_perturbed_theta = batch_theta + perturbations
                batch_perturbed_images = self.decoder(batch_perturbed_theta)
                batch_perturbations = batch_perturbed_images - batch_images

                self.model.train()
                assert self.model.training is True

                output_classes = self.model(batch_perturbed_images)

                self.scheduler.optimizer.zero_grad()
                loss = self.loss(batch_classes, output_classes)
                loss.backward()
                self.scheduler.optimizer.step()
                perturbation_loss = loss = loss.item()

                gradient = torch.mean(torch.abs(list(self.model.parameters())[0].grad))
                gradient = gradient.item()

                error = self.error(batch_classes, output_classes)
                perturbation_error = error = error.item()
            else:
                perm = numpy.concatenate((
                    numpy.take(permutation, range(b * self.args.batch_size, b * self.args.batch_size + split), mode='wrap'),
                    numpy.take(perturbation_permutation, range(b * self.args.batch_size + split, (b + 1) * self.args.batch_size), mode='wrap')
                ), axis=0)
                batch_images = common.torch.as_variable(self.train_images[perm], self.args.use_gpu)
                batch_classes = common.torch.as_variable(self.train_codes[perm], self.args.use_gpu)
                batch_theta = common.torch.as_variable(self.train_theta[perm], self.args.use_gpu)
                batch_images = batch_images.permute(0, 3, 1, 2)

                if isinstance(self.decoder, models.SelectiveDecoder):
                    self.decoder.set_code(batch_classes[split:])
                attack = self.setup_attack(self.decoder_classifier, batch_theta[split:], batch_classes[split:])
                attack.set_bound(torch.from_numpy(self.min_bound), torch.from_numpy(self.max_bound))
                success, perturbations, probabilities, norm, _ = attack.run(objective, self.args.verbose)

                perturbations = common.torch.as_variable(perturbations, self.args.use_gpu)
                batch_perturbed_theta = batch_theta[split:] + perturbations
                batch_perturbed_images = self.decoder(batch_perturbed_theta)
                batch_perturbations = batch_perturbed_images - batch_images[split:]

                self.model.train()
                assert self.model.training is True

                if self.args.safe_bn:
                    output_classes = self.model(batch_images[:split])

                    self.scheduler.optimizer.zero_grad()
                    loss = self.loss(batch_classes[:split], output_classes[:split])
                    loss.backward()
                    self.scheduler.optimizer.step()
                    loss = loss.item()

                    error = self.error(batch_classes[:split], output_classes)
                    error = error.item()

                    output_classes = self.model(batch_perturbed_images)

                    self.scheduler.optimizer.zero_grad()
                    perturbation_loss = self.loss(batch_classes[split:], output_classes)
                    perturbation_loss.backward()
                    self.scheduler.optimizer.step()
                    perturbation_loss = perturbation_loss.item()

                    gradient = torch.mean(torch.abs(list(self.model.parameters())[0].grad))
                    gradient = gradient.item()

                    perturbation_error = self.error(batch_classes[split:], output_classes)
                    perturbation_error = perturbation_error.item()
                else:
                    batch_input_images = torch.cat((batch_images[:split], batch_perturbed_images), dim=0)
                    output_classes = self.model(batch_input_images)

                    self.scheduler.optimizer.zero_grad()
                    loss = self.loss(batch_classes[:split], output_classes[:split])
                    perturbation_loss = self.loss(batch_classes[split:], output_classes[split:])
                    l = (loss + perturbation_loss) / 2
                    l.backward()
                    self.scheduler.optimizer.step()
                    loss = loss.item()
                    perturbation_loss = perturbation_loss.item()

                    gradient = torch.mean(torch.abs(list(self.model.parameters())[0].grad))
                    gradient = gradient.item()

                    error = self.error(batch_classes[:split], output_classes[:split])
                    error = error.item()

                    perturbation_error = self.error(batch_classes[split:], output_classes[split:])
                    perturbation_error = perturbation_error.item()

            iterations = numpy.mean(success[success >= 0]) if numpy.sum(success >= 0) > 0 else -1
            norm = numpy.mean(numpy.linalg.norm(perturbations.reshape(perturbations.shape[0], -1), axis=1, ord=self.norm))
            success = numpy.sum(success >= 0) / (self.args.batch_size//2)

            iteration = self.epoch * num_batches + b + 1
            self.train_statistics = numpy.vstack((self.train_statistics, numpy.array([[
                iteration,  # iterations
                iteration * (1 + self.args.max_iterations) * self.args.batch_size,  # samples seen
                min(num_batches, iteration) * self.args.batch_size + iteration * self.args.max_iterations * self.args.batch_size,  # unique samples seen
                loss,
                error,
                perturbation_loss,
                perturbation_error,
                success,
                iterations,
                norm,
                gradient
            ]])))

            skip = 20
            if b % skip == skip // 2:
                log('[Training] %d | %d: %g (%g) %g (%g) [%g]' % (
                    self.epoch,
                    b,
                    numpy.mean(self.train_statistics[max(0, iteration - skip):iteration, 3]),
                    numpy.mean(self.train_statistics[max(0, iteration - skip):iteration, 4]),
                    numpy.mean(self.train_statistics[max(0, iteration - skip):iteration, 5]),
                    numpy.mean(self.train_statistics[max(0, iteration - skip):iteration, 6]),
                    numpy.mean(self.train_statistics[max(0, iteration - skip):iteration, -1]),
                ))
                log('[Training] %d | %d: %g (%g, %g)' % (
                    self.epoch,
                    b,
                    numpy.mean(self.train_statistics[max(0, iteration - skip):iteration, 7]),
                    numpy.mean(self.train_statistics[max(0, iteration - skip):iteration, 8]),
                    numpy.mean(self.train_statistics[max(0, iteration - skip):iteration, 9]),
                ))

        self.debug('clean.%d.png' % self.epoch, batch_images.permute(0, 2, 3, 1))
        self.debug('perturbed.%d.png' % self.epoch, batch_perturbed_images.permute(0, 2, 3, 1))
        self.debug('perturbation.%d.png' % self.epoch, batch_perturbations.permute(0, 2, 3, 1), cmap='seismic')

    def test(self):
        """
        Test the model.
        """

        self.model.eval()
        assert self.model.training is False
        log('[Training] %d set classifier to eval' % self.epoch)

        loss = error = perturbation_loss = perturbation_error = success = iterations = norm = 0
        num_batches = int(math.ceil(self.args.test_samples / self.args.batch_size))

        for b in range(num_batches):
            perm = numpy.take(range(self.args.test_samples), range(b*self.args.batch_size, (b+1)*self.args.batch_size), mode='clip')
            batch_images = common.torch.as_variable(self.test_images[perm], self.args.use_gpu)
            batch_classes = common.torch.as_variable(self.test_codes[perm], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            output_classes = self.model(batch_images)
            e = self.loss(batch_classes, output_classes)
            loss += e.item()
            a = self.error(batch_classes, output_classes)
            error += a.item()

        loss /= num_batches
        error /= num_batches

        num_batches = int(math.ceil(self.args.attack_samples / self.args.batch_size))
        assert self.args.attack_samples > 0 and self.args.attack_samples <= self.test_images.shape[0]

        for b in range(num_batches):
            perm = numpy.take(range(self.args.attack_samples), range(b*self.args.batch_size, (b+1)*self.args.batch_size), mode='clip')
            batch_theta = common.torch.as_variable(self.test_theta[perm], self.args.use_gpu)
            batch_classes = common.torch.as_variable(self.test_codes[perm], self.args.use_gpu)

            objective = self.objective_class()
            if isinstance(self.decoder, models.SelectiveDecoder):
                self.decoder.set_code(batch_classes)
            attack = self.setup_attack(self.decoder_classifier, batch_theta, batch_classes)
            attack.set_bound(torch.from_numpy(self.min_bound), torch.from_numpy(self.max_bound))
            s, p, _, _, _ = attack.run(objective, False)

            perturbations = common.torch.as_variable(p, self.args.use_gpu)
            batch_perturbed_theta = batch_theta + perturbations
            batch_perturbed_images = self.decoder(batch_perturbed_theta)

            output_classes = self.model(batch_perturbed_images)
            e = self.loss(batch_classes, output_classes)
            perturbation_loss += e.item()
            a = self.error(batch_classes, output_classes)
            perturbation_error += a.item()

            iterations += numpy.mean(s[s >= 0]) if numpy.sum(s >= 0) > 0 else -1
            norm += numpy.mean(numpy.linalg.norm(p.reshape(p.shape[0], -1), axis=1, ord=self.norm))
            success += numpy.sum(s >= 0) / self.args.batch_size

        perturbation_loss /= num_batches
        perturbation_error /= num_batches
        success /= num_batches
        iterations /= num_batches
        norm /= num_batches
        log('[Training] %d: test %g (%g) %g (%g)' % (self.epoch, loss, error, perturbation_loss, perturbation_error))
        log('[Training] %d: test %g (%g, %g)' % (self.epoch, success, iterations, norm))

        num_batches = int(math.ceil(self.train_images.shape[0]/self.args.batch_size))
        iteration = self.epoch * num_batches
        self.test_statistics = numpy.vstack((self.test_statistics, numpy.array([[
            iteration,  # iterations
            iteration * (1 + self.args.max_iterations) * self.args.batch_size,  # samples seen
            min(num_batches, iteration) * self.args.batch_size + iteration * self.args.max_iterations * self.args.batch_size,  # unique samples seen
            loss,
            error,
            perturbation_loss,
            perturbation_error,
            success,
            iterations,
            norm,
        ]])))

    def plot(self):
        """
        Plot error and accuracy.
        """

        if self.args.loss_file:
            plot.line(self.args.loss_file, [
                self.train_statistics[:, 0],
                self.test_statistics[:, 0],
                self.train_statistics[:, 0],
                self.test_statistics[:, 0],
            ], [
                self.train_statistics[:, 3],
                self.test_statistics[:, 3],
                self.train_statistics[:, 5],
                self.test_statistics[:, 5],
            ], [
                'Train Loss',
                'Test Loss',
                'Train Loss (Perturbed)',
                'Test Loss (Perturbed)',
            ], title='Loss during Training', xlabel='Iteration', ylabel='Loss')
        if self.args.error_file:
            plot.line(self.args.error_file, [
                self.train_statistics[:, 0],
                self.test_statistics[:, 0],
                self.train_statistics[:, 0],
                self.test_statistics[:, 0],
            ], [
                self.train_statistics[:, 4],
                self.test_statistics[:, 4],
                self.train_statistics[:, 6],
                self.test_statistics[:, 6],
            ], [
                'Train Error',
                'Test Error',
                'Train Error (Perturbed)',
                'Test Error (Perturbed)',
            ], title='Error during Training', xlabel='Iteration', ylabel='Error')
        if self.args.success_file:
            plot.line(self.args.success_file, [
                self.train_statistics[:, 0],
                self.test_statistics[:, 0],
                self.train_statistics[:, 0],
                self.test_statistics[:, 0]
            ], [
                self.train_statistics[:, 7],
                self.test_statistics[:, 7],
                self.train_statistics[:, 9],
                self.test_statistics[:, 9]
            ], [
                'Train Success Rate',
                'Test Success Rate',
                'Train Norm',
                'Test Norm'
            ], title='Attack Success Rate / Norm during Training', xlabel='Iteration', ylabel='Success Rate / Norm')
        if self.args.gradient_file:
            plot.line(self.args.gradient_file, [
                self.train_statistics[:, 0],
            ], [
                self.train_statistics[:, -1],
            ], [
                'Train Gradient Norm',
            ], title='Gradient during Training', xlable='Iteration', ylable='Gradient Norm')

    def load_data(self):
        """
        Load data.
        """

        assert self.args.batch_size%4 == 0

        self.train_images = utils.read_hdf5(self.args.train_images_file).astype(numpy.float32)
        log('[Training] read %s' % self.args.train_images_file)

        self.test_images = utils.read_hdf5(self.args.test_images_file).astype(numpy.float32)
        log('[Training] read %s' % self.args.test_images_file)

        # For handling both color and gray images.
        if len(self.train_images.shape) < 4:
            self.train_images = numpy.expand_dims(self.train_images, axis=3)
            self.test_images = numpy.expand_dims(self.test_images, axis=3)
            log('[Training] no color images, adjusted size')
        self.resolution = self.test_images.shape[2]
        log('[Training] resolution %d' % self.resolution)

        self.train_codes = utils.read_hdf5(self.args.train_codes_file).astype(numpy.int)
        assert self.train_codes.shape[1] >= self.args.label_index + 1
        self.train_codes = self.train_codes[:, self.args.label_index]
        log('[Training] read %s' % self.args.train_codes_file)
        self.N_class = numpy.max(self.train_codes) + 1

        self.test_codes = utils.read_hdf5(self.args.test_codes_file).astype(numpy.int)
        assert self.test_codes.shape[1] >= self.args.label_index + 1
        self.test_codes = self.test_codes[:, self.args.label_index]
        log('[Training] read %s' % self.args.test_codes_file)

        self.train_theta = utils.read_hdf5(self.args.train_theta_file).astype(numpy.float32)
        log('[Training] read %s' % self.args.train_theta_file)

        assert self.test_images.shape[0] == self.test_codes.shape[0]

        self.min_bound = numpy.min(self.train_theta, axis=0)
        self.max_bound = numpy.max(self.train_theta, axis=0)
        log('[Training] min bound: %s' % ' '.join(['%g' % self.min_bound[i] for i in range(self.min_bound.shape[0])]))
        log('[Training] max bound: %s' % ' '.join(['%g' % self.max_bound[i] for i in range(self.max_bound.shape[0])]))

        self.test_theta = utils.read_hdf5(self.args.test_theta_file).astype(numpy.float32)
        log('[Training] read %s' % self.args.test_theta_file)

        assert self.train_codes.shape[0] == self.train_images.shape[0]
        assert self.test_codes.shape[0] == self.test_images.shape[0]
        assert self.train_theta.shape[0] == self.train_images.shape[0], '%s != %s' % ('x'.join(list(map(str, self.train_theta.shape))), 'x'.join(list(map(str, self.train_images.shape))))
        assert self.test_theta.shape[0] == self.test_images.shape[0]

        # Select subset of samples
        if self.args.training_samples < 0:
            self.args.training_samples = self.train_images.shape[0]
        else:
            self.args.training_samples = min(self.args.training_samples, self.train_images.shape[0])
        log('[Training] using %d training samples' % self.args.training_samples)

        if self.args.test_samples < 0:
            self.args.test_samples = self.test_images.shape[0]
        else:
            self.args.test_samples = min(self.args.test_samples, self.test_images.shape[0])

        if self.args.early_stopping:
            assert self.args.validation_samples > 0
            assert self.args.training_samples + self.args.validation_samples <= self.train_images.shape[0]
            self.val_images = self.train_images[self.train_images.shape[0] - self.args.validation_samples:]
            self.val_codes = self.train_codes[self.train_codes.shape[0] - self.args.validation_samples:]
            self.train_images = self.train_images[:self.train_images.shape[0] - self.args.validation_samples]
            self.train_codes = self.train_codeſ[:self.train_codes.shape[0] - self.args.validation_samples]
            assert self.val_images.shape[0] == self.args.validation_samples and self.val_codes.shape[0] == self.args.validation_samples

        if self.args.random_samples:
            perm = numpy.random.permutation(self.train_images.shape[0] // 10)
            perm = perm[:self.args.training_samples // 10]
            perm = numpy.repeat(perm, self.N_class, axis=0) * 10 + numpy.tile(numpy.array(range(self.N_class)), (perm.shape[0]))
            self.train_images = self.train_images[perm]
            self.train_codes = self.train_codes[perm]
            self.train_theta = self.train_theta[perm]
        else:
            self.train_images = self.train_images[:self.args.training_samples]
            self.train_codes = self.train_codes[:self.args.training_samples]
            self.train_theta = self.train_theta[:self.args.training_samples]

        self.train_valid = (numpy.max(numpy.abs(self.train_theta), axis=1) <= self.args.bound).astype(int)
        self.test_valid = (numpy.max(numpy.abs(self.test_theta), axis=1) <= self.args.bound).astype(int)

        # Check that the dataset is balanced.
        number_samples = self.train_codes.shape[0]//self.N_class
        for c in range(self.N_class):
            number_samples_ = numpy.sum(self.train_codes == c)
            if number_samples_ != number_samples:
                log('[Training] dataset not balanced, class %d should have %d samples but has %d' % (c, number_samples, number_samples_), LogLevel.WARNING)

    def load_decoder(self):
        """
        Load the decoder.
        """

        assert self.args.decoder_files
        decoder_files = self.args.decoder_files.split(',')
        for decoder_file in decoder_files:
            assert os.path.exists(decoder_file), 'could not find %s' % decoder_file

        log('[Training] using %d input channels' % self.train_images.shape[3])
        decoder_units = list(map(int, self.args.decoder_units.split(',')))

        if len(decoder_files) > 1:
            log('[Training] loading multiple decoders')
            decoders = []
            for i in range(len(decoder_files)):
                decoder = models.LearnedDecoder(self.args.latent_space_size, resolution=(self.train_images.shape[3], self.train_images.shape[1], self.train_images.shape[2]),
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
                log('[Training] loaded %s' % decoder_files[i])
            self.decoder = models.SelectiveDecoder(decoders, resolution=(self.train_images.shape[3], self.train_images.shape[1], self.train_images.shape[2]))
        else:
            log('[Training] loading one decoder')
            decoder = models.LearnedDecoder(self.args.latent_space_size, resolution=(self.train_images.shape[3], self.train_images.shape[1], self.train_images.shape[2]),
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
            log('[Training] read decoder')

            self.decoder = decoder

        self.decoder_classifier = models.DecoderClassifier(self.decoder, self.model)

    def main(self):
        """
        Main which should be overwritten.
        """

        self.load_attack()
        self.load_data()
        self.load_model_and_scheduler()
        self.load_decoder()

        assert self.norm is not None
        assert self.norm in [1, 2, float('inf')]
        log('[Training] using norm %g' % self.norm)

        assert self.model is not None
        assert self.decoder is not None
        assert self.decoder.training is False
        assert self.decoder_classifier is not None
        assert self.scheduler is not None
        self.loop()


if __name__ == '__main__':
    program = TrainLearnedDecoderClassifierAdversarially()
    program.main()