import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
from common.log import log
from common import paths
import common.torch
from training import train_classifier
import numpy
import argparse
import math
import torch
if utils.display():
    from common import plot


class TrainClassifierAugmented(train_classifier.TrainClassifier):
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

        super(TrainClassifierAugmented, self).__init__(args)

        self.train_statistics = numpy.zeros((0, 8))
        self.test_statistics = numpy.zeros((0, 7))

        self.norm = None
        """ (float) Attack norm for data augmentation. """

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Train classifier.')
        parser.add_argument('-train_images_file', default=paths.train_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-train_codes_file', default=paths.train_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-state_file', default=paths.state_file('robust_classifier'), help='Snapshot state file.', type=str)
        parser.add_argument('-log_file', default=paths.log_file('robust_classifier'), help='Log file.', type=str)
        parser.add_argument('-training_file', default=paths.results_file('robust_training'), help='Training statistics file.', type=str)
        parser.add_argument('-testing_file', default=paths.results_file('robust_testing'), help='Testing statistics file.', type=str)
        parser.add_argument('-loss_file', default=paths.image_file('loss'), help='Loss plot file.', type=str)
        parser.add_argument('-error_file', default=paths.image_file('error'), help='Error plot file.', type=str)
        parser.add_argument('-gradient_file', default='', help='Gradient plot file.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-random_samples', default=False, action='store_true', help='Randomize the subsampling of the training set.')
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
        parser.add_argument('-full_variant', default=False, action='store_true', help='100% variant.')
        parser.add_argument('-strong_variant', default=False, action='store_true', help='Strong variant.')

        return parser

    def train(self):
        """
        Train with fair data augmentation.
        """

        self.model.train()
        log('[Training] %d set classifier to train' % self.epoch)
        assert self.model.training is True

        split = self.args.batch_size // 2
        num_batches = int(math.ceil(self.train_images.shape[0] / self.args.batch_size))
        permutation = numpy.random.permutation(self.train_images.shape[0])

        for b in range(num_batches):
            self.scheduler.update(self.epoch, float(b) / num_batches)

            perm = numpy.take(permutation, range(b*self.args.batch_size, (b+1)*self.args.batch_size), mode='wrap')
            batch_images = common.torch.as_variable(self.train_images[perm], self.args.use_gpu)
            batch_classes = common.torch.as_variable(self.train_codes[perm], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            if self.args.full_variant:
                loss = error = gradient = 0
                for t in range(self.args.max_iterations):
                    size = batch_images.size()
                    batch_perturbations = common.numpy.uniform_ball(size[0], numpy.prod(size[1:]), epsilon=self.args.epsilon, ord=self.norm)
                    batch_perturbations = common.torch.as_variable(batch_perturbations.reshape(size).astype(numpy.float32), self.args.use_gpu)
                    batch_perturbations = torch.min(torch.ones_like(batch_images) - batch_images, batch_perturbations)
                    batch_perturbations = torch.max(torch.zeros_like(batch_images) - batch_images, batch_perturbations)

                    batch_perturbed_images = batch_images + batch_perturbations
                    output_perturbed_classes = self.model(batch_perturbed_images)

                    self.scheduler.optimizer.zero_grad()
                    l = self.loss(batch_classes, output_perturbed_classes)
                    l.backward()
                    self.scheduler.optimizer.step()
                    loss += l.item()

                    g = torch.mean(torch.abs(list(self.model.parameters())[0].grad))
                    gradient += g.item()

                    e = self.error(batch_classes, output_perturbed_classes)
                    error += e.item()

                gradient /= self.args.max_iterations
                loss /= self.args.max_iterations
                error /= self.args.max_iterations
                perturbation_loss = loss
                perturbation_error = error

            elif self.args.strong_variant:
                raise NotImplementedError('strong_variant not implemented yet')
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
                    size = batch_images.size()
                    batch_perturbations = common.numpy.uniform_ball(split, numpy.prod(size[1:]), epsilon=self.args.epsilon, ord=self.norm)
                    batch_perturbations = common.torch.as_variable(batch_perturbations.reshape(split, size[1], size[2], size[3]).astype(numpy.float32), self.args.use_gpu)
                    batch_perturbations = torch.min(torch.ones_like(batch_images[split:]) - batch_images[split:], batch_perturbations)
                    batch_perturbations = torch.max(torch.zeros_like(batch_images[split:]) - batch_images[split:], batch_perturbations)

                    batch_perturbed_images = batch_images[split:] + batch_perturbations
                    output_perturbed_classes = self.model(batch_perturbed_images)

                    self.scheduler.optimizer.zero_grad()
                    l = self.loss(batch_classes[split:], output_perturbed_classes)
                    l.backward()
                    self.scheduler.optimizer.step()
                    perturbation_loss += l.item()

                    g = torch.mean(torch.abs(list(self.model.parameters())[0].grad))
                    gradient += g.item()

                    e = self.error(batch_classes[split:], output_perturbed_classes)
                    perturbation_error += e.item()

                gradient /= self.args.max_iterations
                perturbation_loss /= self.args.max_iterations
                perturbation_error /= self.args.max_iterations

            iteration = self.epoch * num_batches + b + 1
            self.train_statistics = numpy.vstack((self.train_statistics, numpy.array([[
                iteration, # iterations
                iteration * (1 + self.args.max_iterations) * self.args.batch_size, # samples seen
                min(num_batches, iteration) * self.args.batch_size + iteration * self.args.max_iterations * self.args.batch_size, # unique samples seen
                loss, # clean loss
                error, # clean error (1-accuracy)
                perturbation_loss, # perturbation loss
                perturbation_error, # perturbation error (1-accuracy)
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
        assert self.model.training is False

        loss = error = perturbation_loss = perturbation_error = 0
        num_batches = int(math.ceil(self.args.test_samples/self.args.batch_size))

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

            images = batch_images.data.cpu().numpy()
            perturbations = common.numpy.uniform_ball(images.shape[0], numpy.prod(images.shape[1:]), epsilon=self.args.epsilon, ord=self.norm)
            perturbations = perturbations.reshape(images.shape)

            perturbations = numpy.minimum(numpy.ones(images.shape) - images, perturbations)
            perturbations = numpy.maximum(numpy.zeros(images.shape) - images, perturbations)

            perturbations = perturbations.astype(numpy.float32)
            batch_perturbed_images = batch_images + common.torch.as_variable(perturbations, self.args.use_gpu)

            output_classes = self.model(batch_perturbed_images)

            e = self.loss(batch_classes, output_classes)
            perturbation_loss += e.item()

            e = self.error(batch_classes, output_classes)
            perturbation_error += e.item()

        loss /= num_batches
        error /= num_batches
        perturbation_loss /= num_batches
        perturbation_error /= num_batches
        log('[Training] %d: test %g (%g) %g (%g)' % (self.epoch, loss, error, perturbation_loss, perturbation_error))

        num_batches = int(math.ceil(self.train_images.shape[0]/self.args.batch_size))
        iteration = self.epoch*num_batches
        self.test_statistics = numpy.vstack((self.test_statistics, numpy.array([[
            iteration, # iterations
            iteration * (1 + self.args.max_iterations) * self.args.batch_size,  # samples seen
            min(num_batches, iteration) * self.args.batch_size + iteration * self.args.max_iterations * self.args.batch_size,  # unique samples seen
            loss,
            error,
            perturbation_loss,  # perturbation loss
            perturbation_error,  # perturbation error (1-accuracy)
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

        self.norm = float(self.args.norm)
        assert self.norm is not None
        assert self.norm in [1, 2, float('inf')]
        log('[Training] using norm %g' % self.norm)

        assert self.model is not None
        assert self.scheduler is not None
        self.loop()


if __name__ == '__main__':
    program = TrainClassifierAugmented()
    program.main()