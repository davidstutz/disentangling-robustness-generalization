import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
from common.log import log, Log
from common import paths
import common.torch
import common.numpy
from training import train_decoder_classifier_adversarially
import torch
import numpy
import argparse
import math
if utils.display():
    from common import plot


class TrainDecoderClassifierAugmented(train_decoder_classifier_adversarially.TrainDecoderClassifierAdversarially):
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

        super(TrainDecoderClassifierAugmented, self).__init__(args)

        self.train_statistics = numpy.zeros((0, 8))
        self.test_statistics = numpy.zeros((0, 7))

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

        parser = argparse.ArgumentParser(description='Train classifier.')
        parser.add_argument('-database_file', default=paths.database_file(), help='HDF5 file containing font prototype images.', type=str)
        parser.add_argument('-train_images_file', default=paths.train_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-train_codes_file', default=paths.train_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-train_theta_file', default=paths.train_theta_file(), help='HDF5 file containing transformations.', type=str)
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-label_index', default=2, help='Column index in label file.', type=int)
        parser.add_argument('-test_theta_file', default=paths.test_theta_file(), help='HDF5 file containing transformations.', type=str)
        parser.add_argument('-state_file', default=paths.state_file('robust_manifold_classifier'), help='Snapshot state file.', type=str)
        parser.add_argument('-log_file', default=paths.log_file('robust_manifold_classifier'), help='Log file.', type=str)
        parser.add_argument('-training_file', default=paths.results_file('robust_manifold_training'), help='Training statistics file.', type=str)
        parser.add_argument('-testing_file', default=paths.results_file('robust_manifold_testing'), help='Testing statistics file.', type=str)
        parser.add_argument('-loss_file', default=paths.image_file('loss'), help='Loss plot file.', type=str)
        parser.add_argument('-error_file', default=paths.image_file('error'), help='Error plot file.', type=str)
        parser.add_argument('-gradient_file', default='', help='Gradient plot file.', type=str)
        parser.add_argument('-random_samples', default=False,  action='store_true', help='Randomize the subsampling of the training set.')
        parser.add_argument('-training_samples', default=-1, help='Number of samples used for training.', type=int)
        parser.add_argument('-test_samples', default=-1, help='Number of samples for validation.', type=int)
        parser.add_argument('-validation_samples', default=0, help='Number of samples for validation.', type=int)
        parser.add_argument('-early_stopping', default=False, action='store_true', help='Use early stopping.')
        parser.add_argument('-batch_size', default=64, help='Batch size.', type=int)
        parser.add_argument('-epochs', default=10, help='Number of epochs.', type=int)
        parser.add_argument('-weight_decay', default=0.0001, help='Weight decay importance.', type=float)
        parser.add_argument('-logit_decay', default=0, help='Logit decay importance.', type=float)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')
        parser.add_argument('-skip', default=5, help='Verbosity in iterations.', type=int)
        parser.add_argument('-lr', default=0.005, type=float, help='Base learning rate.')
        parser.add_argument('-lr_decay', default=0.9, type=float, help='Learning rate decay.')
        parser.add_argument('-results_file', default='', help='Results file for evaluation.', type=str)
        parser.add_argument('-debug_directory', default='', help='Debug directory.', type=str)

        # Some network parameters.
        parser.add_argument('-network_architecture', default='standard', help='Classifier architecture to use.', type=str)
        parser.add_argument('-network_activation', default='relu', help='Activation function to use.', type=str)
        parser.add_argument('-network_no_batch_normalization', default=False, help='Do not use batch normalization.', action='store_true')
        parser.add_argument('-network_channels', default=16, help='Channels of first convolutional layer, afterwards channels are doubled.', type=int)
        parser.add_argument('-network_dropout', default=False, action='store_true', help='Whether to use dropout.')
        parser.add_argument('-network_units', default='1024,1024,1024,1024', help='Units for MLP.')

        # Attack parameters.
        parser.add_argument('-norm', default='inf', help='Norm to use.', type=float)
        parser.add_argument('-epsilon', default=1, help='Epsilon allowed for attacks.', type=float)
        parser.add_argument('-max_iterations', default=10, help='Number of iterations for attack.', type=int)

        # Variants.
        parser.add_argument('-strong_variant', default=False, action='store_true', help='Strong data augmentation variant.')
        parser.add_argument('-full_variant', default=False, action='store_true', help='100% variant.')

        return parser

    def train(self):
        """
        Train with fair data augmentation.
        """

        self.model.train()
        assert self.model.training is True

        split = self.args.batch_size // 2
        num_batches = int(math.ceil(self.train_images.shape[0] / self.args.batch_size))
        permutation = numpy.random.permutation(self.train_images.shape[0])

        for b in range(num_batches):
            self.scheduler.update(self.epoch, float(b) / num_batches)

            perm = numpy.take(permutation, range(b*self.args.batch_size, (b+1)*self.args.batch_size), mode='wrap')
            batch_images = common.torch.as_variable(self.train_images[perm], self.args.use_gpu)
            batch_theta = common.torch.as_variable(self.train_theta[perm], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            batch_fonts = self.train_codes[perm, 1]
            batch_classes = self.train_codes[perm, self.args.label_index]
            batch_code = numpy.concatenate((common.numpy.one_hot(batch_fonts, self.N_font), common.numpy.one_hot(batch_classes, self.N_class)), axis=1).astype(numpy.float32)

            batch_code = common.torch.as_variable(batch_code, self.args.use_gpu)
            batch_classes = common.torch.as_variable(batch_classes, self.args.use_gpu)

            loss = error = gradient = 0
            if self.args.full_variant:
                for t in range(self.args.max_iterations):
                    if self.args.strong_variant:
                        # Here, we want to uniformly sample all allowed transformations, so that's OK.
                        min_bound = numpy.repeat(self.min_bound.reshape(1, -1), self.args.batch_size, axis=0)
                        max_bound = numpy.repeat(self.max_bound.reshape(1, -1), self.args.batch_size, axis=0)
                        random = numpy.random.uniform(min_bound, max_bound, (batch_theta.size(0), batch_theta.size(1)))

                        batch_perturbed_theta = common.torch.as_variable(random.astype(numpy.float32), self.args.use_gpu)

                        self.decoder.set_code(batch_code)
                        batch_perturbed_images = self.decoder(batch_perturbed_theta)
                    else:
                        random = common.numpy.uniform_ball(batch_theta.size(0), batch_theta.size(1), epsilon=self.args.epsilon, ord=self.norm)
                        batch_perturbed_theta = batch_theta + common.torch.as_variable(random.astype(numpy.float32), self.args.use_gpu)
                        batch_perturbed_theta = torch.min(common.torch.as_variable(self.max_bound, self.args.use_gpu), batch_perturbed_theta)
                        batch_perturbed_theta = torch.max(common.torch.as_variable(self.min_bound, self.args.use_gpu), batch_perturbed_theta)

                        self.decoder.set_code(batch_code)
                        batch_perturbed_images = self.decoder(batch_perturbed_theta)

                    output_classes = self.model(batch_perturbed_images)

                    self.scheduler.optimizer.zero_grad()
                    l = self.loss(batch_classes, output_classes)
                    l.backward()
                    self.scheduler.optimizer.step()
                    loss += l.item()

                    g = torch.mean(torch.abs(list(self.model.parameters())[0].grad))
                    gradient += g.item()

                    e = self.error(batch_classes, output_classes)
                    error += e.item()

                batch_perturbations = batch_perturbed_images - batch_images
                gradient /= self.args.max_iterations
                loss /= self.args.max_iterations
                error /= self.args.max_iterations
                perturbation_loss = loss
                perturbation_error = error
            else:
                output_classes = self.model(batch_images[:split])

                self.scheduler.optimizer.zero_grad()
                l = self.loss(batch_classes[:split], output_classes)
                l.backward()
                self.scheduler.optimizer.step()
                loss = l.item()

                gradient = torch.mean(torch.abs(list(self.model.parameters())[0].grad))
                gradient = gradient.item()

                e = self.error(batch_classes[:split], output_classes)
                error = e.item()

                perturbation_loss = perturbation_error = 0
                for t in range(self.args.max_iterations):
                    if self.args.strong_variant:
                        # Again, sampling all possible transformations.
                        min_bound = numpy.repeat(self.min_bound.reshape(1, -1), split, axis=0)
                        max_bound = numpy.repeat(self.max_bound.reshape(1, -1), split, axis=0)
                        random = numpy.random.uniform(min_bound, max_bound, (split, batch_theta.size(1)))

                        batch_perturbed_theta = common.torch.as_variable(random.astype(numpy.float32), self.args.use_gpu)

                        self.decoder.set_code(batch_code[split:])
                        batch_perturbed_images = self.decoder(batch_perturbed_theta)
                    else:
                        random = common.numpy.uniform_ball(split, batch_theta.size(1), epsilon=self.args.epsilon, ord=self.norm)
                        batch_perturbed_theta = batch_theta[split:] + common.torch.as_variable(random.astype(numpy.float32), self.args.use_gpu)
                        batch_perturbed_theta = torch.min(common.torch.as_variable(self.max_bound, self.args.use_gpu), batch_perturbed_theta)
                        batch_perturbed_theta = torch.max(common.torch.as_variable(self.min_bound, self.args.use_gpu), batch_perturbed_theta)

                        self.decoder.set_code(batch_code[split:])
                        batch_perturbed_images = self.decoder(batch_perturbed_theta)

                    output_classes = self.model(batch_perturbed_images)

                    self.scheduler.optimizer.zero_grad()
                    l = self.loss(batch_classes[split:], output_classes)
                    l.backward()
                    self.scheduler.optimizer.step()
                    perturbation_loss += l.item()

                    g = torch.mean(torch.abs(list(self.model.parameters())[0].grad))
                    gradient += g.item()

                    e = self.error(batch_classes[split:], output_classes)
                    perturbation_error += e.item()

                batch_perturbations = batch_perturbed_images - batch_images[split:]
                gradient /= self.args.max_iterations + 1
                perturbation_loss /= self.args.max_iterations
                perturbation_error /= self.args.max_iterations

            iteration = self.epoch * num_batches + b + 1
            self.train_statistics = numpy.vstack((self.train_statistics, numpy.array([[
                iteration,  # iterations
                iteration * (1 + self.args.max_iterations) * self.args.batch_size,  # samples seen
                min(num_batches, iteration) * self.args.batch_size + iteration * self.args.max_iterations * self.args.batch_size,  # unique samples seen
                loss,
                error,
                perturbation_loss,
                perturbation_error,
                gradient
            ]])))

            if b % self.args.skip == self.args.skip // 2:
                log('[Training] %d | %d: %g (%g) %g (%g) [%g]' % (
                    self.epoch,
                    b,
                    numpy.mean(self.train_statistics[max(0, iteration - self.args.skip):iteration, 3]),
                    numpy.mean(self.train_statistics[max(0, iteration - self.args.skip):iteration, 4]),
                    numpy.mean(self.train_statistics[max(0, iteration - self.args.skip):iteration, 5]),
                    numpy.mean(self.train_statistics[max(0, iteration - self.args.skip):iteration, 6]),
                    numpy.mean(self.train_statistics[max(0, iteration - self.args.skip):iteration, -1]),
                ))

        self.debug('clean.%d.png' % self.epoch, batch_images.permute(0, 2, 3, 1))
        self.debug('perturbed.%d.png' % self.epoch, batch_perturbed_images.permute(0, 2, 3, 1))
        self.debug('perturbation.%d.png' % self.epoch, batch_perturbations.permute(0, 2, 3, 1), cmap='seismic')

    def test(self):
        """
        Test the model.
        """

        self.model.eval()
        log('[Training] %d set classifier to eval' % self.epoch)

        loss = error = perturbation_loss = perturbation_error = 0
        num_batches = int(math.ceil(self.args.test_samples/self.args.batch_size))
        assert self.model.training is False

        for b in range(num_batches):
            perm = numpy.take(range(self.args.test_samples), range(b*self.args.batch_size, (b+1)*self.args.batch_size), mode='clip')
            batch_images = common.torch.as_variable(self.test_images[perm], self.args.use_gpu)
            batch_theta = common.torch.as_variable(self.test_theta[perm], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            batch_fonts = self.test_codes[perm, 1]
            batch_classes = self.test_codes[perm, self.args.label_index]
            batch_code = numpy.concatenate((common.numpy.one_hot(batch_fonts, self.N_font), common.numpy.one_hot(batch_classes, self.N_class)), axis=1).astype(numpy.float32)

            batch_code = common.torch.as_variable(batch_code, self.args.use_gpu)
            batch_classes = common.torch.as_variable(batch_classes, self.args.use_gpu)

            output_classes = self.model(batch_images)
            e = self.loss(batch_classes, output_classes)
            loss += e.item()
            a = self.error(batch_classes, output_classes)
            error += a.item()

            if self.args.strong_variant:
                min_bound = numpy.repeat(self.min_bound.reshape(1, -1), batch_theta.size(0), axis=0)
                max_bound = numpy.repeat(self.max_bound.reshape(1, -1), batch_theta.size(0), axis=0)
                random = numpy.random.uniform(min_bound, max_bound, (batch_theta.size(0), batch_theta.size(1)))

                batch_perturbed_theta = common.torch.as_variable(random.astype(numpy.float32), self.args.use_gpu)

                self.decoder.set_code(batch_code)
                batch_perturbed_images = self.decoder(batch_perturbed_theta)
            else:
                random = common.numpy.uniform_ball(batch_theta.size(0), batch_theta.size(1), epsilon=self.args.epsilon, ord=self.norm)
                batch_perturbed_theta = batch_theta + common.torch.as_variable(random.astype(numpy.float32), self.args.use_gpu)
                batch_perturbed_theta = torch.min(common.torch.as_variable(self.max_bound, self.args.use_gpu), batch_perturbed_theta)
                batch_perturbed_theta = torch.max(common.torch.as_variable(self.min_bound, self.args.use_gpu), batch_perturbed_theta)

                self.decoder.set_code(batch_code)
                batch_perturbed_images = self.decoder(batch_perturbed_theta)

            output_classes = self.model(batch_perturbed_images)

            l = self.loss(batch_classes, output_classes)
            perturbation_loss += l.item()

            e = self.error(batch_classes, output_classes)
            perturbation_error += e.item()

        loss /= num_batches
        error /= num_batches
        perturbation_loss /= num_batches
        perturbation_error /= num_batches
        log('[Training] %d: test %g (%g) %g (%g)' % (self.epoch, loss, error, perturbation_loss, perturbation_error))

        num_batches = int(math.ceil(self.train_images.shape[0]/self.args.batch_size))
        iteration = self.epoch * num_batches
        self.test_statistics = numpy.vstack((self.test_statistics, numpy.array([[
            iteration,  # iterations
            iteration * (1 + self.args.max_iterations) * self.args.batch_size,  # samples seen
            min(num_batches, iteration) * self.args.batch_size + iteration * self.args.max_iterations * self.args.batch_size,  # unique samples seen
            loss,
            error,
            perturbation_loss,
            perturbation_error
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
        if self.args.gradient_file:
            plot.line(self.args.gradient_file, [
                self.train_statistics[:, 0],
            ], [
                self.train_statistics[:, -1],
            ], [
                'Train Gradient Norm',
            ], title='Gradient during Training', xlable='Iteration', ylable='Gradient Norm')

    def main(self):
        """
        Main which should be overwritten.
        """

        self.load_data()
        self.load_model_and_scheduler()
        self.load_decoder()

        self.norm = float(self.args.norm)
        assert self.norm is not None
        assert self.norm in [1, 2, float('inf')]
        log('[Training] using norm %g' % self.norm)

        assert self.model is not None
        assert self.decoder is not None
        assert self.scheduler is not None
        self.loop()


if __name__ == '__main__':
    program = TrainDecoderClassifierAugmented()
    program.main()