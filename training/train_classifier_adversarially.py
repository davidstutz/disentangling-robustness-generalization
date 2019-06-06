import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
import models
from common.log import log, Log, LogLevel
from common.scheduler import ADAMScheduler
from common.state import State
from common import cuda
from common.timer import elapsed
from common import paths
import common.torch
import common.numpy
from training import train_classifier
import torch
import numpy
import argparse
import math
import functools
if utils.display():
    from common import plot


class TrainClassifierAdversarially(train_classifier.TrainClassifier):
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

        super(TrainClassifierAdversarially, self).__init__(args)

        self.train_statistics = numpy.zeros((0, 11))
        self.test_statistics = numpy.zeros((0, 10))

        self.attack_class = None
        """ (attacks.UntargetedAttack) Attack to use (as class). """

        self.objective_class = None
        """ (attacks.UntargetedObjective) Objective to use (as class). """

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
        parser.add_argument('-loss_file', default=paths.image_file('robust_loss'), help='Loss plot file.', type=str)
        parser.add_argument('-error_file', default=paths.image_file('robust_error'), help='Error plot file.', type=str)
        parser.add_argument('-success_file', default=paths.image_file('robust_success'), help='Success rate plot file.', type=str)
        parser.add_argument('-gradient_file', default=paths.image_file('robust_gradient'), help='Gradient plot file.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
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
        parser.add_argument('-debug_directory', default='', help='Debug directory.', type=str)

        # Some network parameters.
        parser.add_argument('-network_architecture', default='standard', help='Classifier architecture to use.', type=str)
        parser.add_argument('-network_activation', default='relu', help='Activation function to use.', type=str)
        parser.add_argument('-network_no_batch_normalization', default=False, help='Do not use batch normalization.', action='store_true')
        parser.add_argument('-network_channels', default=16, help='Channels of first convolutional layer, afterwards channels are doubled.', type=int)
        parser.add_argument('-network_dropout', default=False, action='store_true', help='Whether to use dropout.')
        parser.add_argument('-network_units', default='1024,1024,1024,1024', help='Units for MLP.')
        parser.add_argument('-anneal_epochs', default=0, help='Anneal iterations in the first epochs.', type=int)

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

        # Variants.
        parser.add_argument('-full_variant', default=False, action='store_true', help='100% variant.')
        parser.add_argument('-training_mode', default=False, action='store_true', help='Training mode variant for attack.')

        return parser

    def setup_attack(self, model, batch_images, batch_classes):
        """
        Setup and initialize attack.

        :param model: model to attack
        :type model: torch.nn.Module
        :param batch_images: images to attack
        :type batch_images: torch.autograd.Variable
        :param batch_classes: true classes to attack
        :type batch_classes: torch.autograd.Variable
        """

        attack = self.attack_class(model, batch_images, batch_classes, self.args.epsilon)

        if getattr(attack, 'set_c_0', None) is not None:
            attack.set_c_0(self.args.c_0)
        if getattr(attack, 'set_c_1', None) is not None:
            attack.set_c_1(self.args.c_1)
        if getattr(attack, 'set_c_2', None) is not None:
            attack.set_c_2(self.args.c_2)
        if getattr(attack, 'set_max_projections', None) is not None:
            attack.set_max_projections(self.args.max_projections)

        max_iterations = self.args.max_iterations
        if self.epoch < self.args.anneal_epochs:
            max_iterations = int(numpy.linspace(0, self.args.max_iterations, self.args.anneal_epochs)[self.epoch])
        attack.set_max_iterations(max_iterations)
        attack.set_base_lr(self.args.base_lr)

        # Training mode means that the attack is run through to the end!
        if self.args.training_mode:
            attack.set_training_mode()
            assert attack.training_mode is True
        else:
            assert attack.training_mode is False

        attack.initialize_random()

        return attack

    def train(self):
        """
        Train adversarially.
        """

        split = self.args.batch_size // 2
        num_batches = int(math.ceil(self.train_images.shape[0] / self.args.batch_size))
        permutation = numpy.random.permutation(self.train_images.shape[0])

        for b in range(num_batches):
            self.scheduler.update(self.epoch, float(b) / num_batches)

            perm = numpy.take(permutation, range(b*self.args.batch_size, (b+1)*self.args.batch_size), mode='wrap')
            batch_classes = common.torch.as_variable(self.train_codes[perm], self.args.use_gpu)
            batch_images = common.torch.as_variable(self.train_images[perm], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            self.model.eval()
            assert self.model.training is False

            if self.args.full_variant:
                batch_images = common.torch.as_variable(self.train_images[perm], self.args.use_gpu)
                batch_images = batch_images.permute(0, 3, 1, 2)

                objective = self.objective_class()
                attack = self.setup_attack(self.model, batch_images, batch_classes)
                success, perturbations, probabilities, norm, _ = attack.run(objective, self.args.verbose)
                batch_perturbations = common.torch.as_variable(perturbations.astype(numpy.float32), self.args.use_gpu)
                batch_perturbed_images = batch_images + batch_perturbations

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
                objective = self.objective_class()
                attack = self.setup_attack(self.model, batch_images[split:], batch_classes[split:])
                success, perturbations, probabilities, norm, _ = attack.run(objective, self.args.verbose)

                batch_perturbations = common.torch.as_variable(perturbations.astype(numpy.float32), self.args.use_gpu)
                batch_perturbed_images = batch_images[split:] + batch_perturbations
                batch_input_images = torch.cat((batch_images[:split], batch_perturbed_images), dim=0)

                self.model.train()
                assert self.model.training is True

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
                iteration,
                iteration * self.args.batch_size,
                min(num_batches, iteration) * self.args.batch_size,
                loss,
                error,
                perturbation_loss,
                perturbation_error,
                success,
                iterations,
                norm,
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
                log('[Training] %d | %d: %g (%g, %g)' % (
                    self.epoch,
                    b,
                    numpy.mean(self.train_statistics[max(0, iteration - self.args.skip):iteration, 7]),
                    numpy.mean(self.train_statistics[max(0, iteration - self.args.skip):iteration, 8]),
                    numpy.mean(self.train_statistics[max(0, iteration - self.args.skip):iteration, 9]),
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

        loss = error = perturbation_loss = perturbation_error = success = iterations = norm = 0
        num_batches = int(math.ceil(self.args.test_samples/self.args.batch_size))

        for b in range(num_batches):
            perm = numpy.take(range(self.args.test_samples), range(b*self.args.batch_size, (b+1)*self.args.batch_size), mode='clip')
            batch_images = common.torch.as_variable(self.test_images[perm], self.args.use_gpu)
            batch_classes = common.torch.as_variable(self.test_codes[perm], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            output_classes = self.model(batch_images)
            e = self.loss(batch_classes, output_classes)
            loss += e.data # 0-dim tensor
            a = self.error(batch_classes, output_classes)
            error += a.data

        loss /= num_batches
        error /= num_batches

        num_batches = int(math.ceil(self.args.attack_samples/self.args.batch_size))
        assert self.args.attack_samples > 0 and self.args.attack_samples <= self.test_images.shape[0]

        for b in range(num_batches):
            perm = numpy.take(range(self.args.attack_samples), range(b*self.args.batch_size, (b+1)*self.args.batch_size), mode='clip')
            batch_images = common.torch.as_variable(self.test_images[perm], self.args.use_gpu)
            batch_classes = common.torch.as_variable(self.test_codes[perm], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            objective = self.objective_class()
            attack = self.setup_attack(self.model, batch_images, batch_classes)
            s, p, _, _, _ = attack.run(objective, False)

            batch_images = batch_images + common.torch.as_variable(p.astype(numpy.float32), self.args.use_gpu)
            output_classes = self.model(batch_images)

            e = self.loss(batch_classes, output_classes)
            perturbation_loss += e.item()

            e = self.error(batch_classes, output_classes)
            perturbation_error += e.item()

            iterations += numpy.mean(s[s >= 0]) if numpy.sum(s >= 0) > 0 else -1
            norm += numpy.mean(numpy.linalg.norm(p.reshape(p.shape[0], -1), axis=1, ord=self.norm))
            success += numpy.sum(s >= 0)/self.args.batch_size

        perturbation_error /= num_batches
        perturbation_loss /= num_batches
        success /= num_batches
        iterations /= num_batches
        norm /= num_batches
        log('[Training] %d: test %g (%g) %g (%g)' % (self.epoch, loss, error, perturbation_loss, perturbation_error))
        log('[Training] %d: test %g (%g, %g)' % (self.epoch, success, iterations, norm))

        num_batches = int(math.ceil(self.train_images.shape[0]/self.args.batch_size))
        iteration = self.epoch*num_batches
        self.test_statistics = numpy.vstack((self.test_statistics, numpy.array([[
            iteration,
            iteration * self.args.batch_size,
            min(num_batches, iteration) * self.args.batch_size,
            loss,
            error,
            perturbation_loss,
            perturbation_error,
            success,
            iterations,
            norm
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

    def load_attack(self):
        """
        load attack.
        """

        self.attack_class = utils.get_class('attacks', self.args.attack)
        if not self.attack_class:
            log('[Error] could not find attack %s' % self.args.attack, LogLevel.ERROR)
            exit(1)
        log('[Training] found %s' % self.attack_class)
        # attack is instantiated per sample

        if self.args.attack.lower().find('l1') > 0:
            self.norm = 1
        elif self.args.attack.lower().find('l2') > 0:
            self.norm = 2
        elif self.args.attack.lower().find('linf') > 0:
            self.norm = float('inf')
        else:
            raise NotImplementedError('could not determine norm from attack name')

        self.objective_class = utils.get_class('attacks', self.args.objective)
        if not self.objective_class:
            log('[Error] could not find objective %s' % self.args.objective, LogLevel.ERROR)
            exit(1)
        log('[Training] found %s' % self.objective_class)

    def main(self):
        """
        Main which should be overwritten.
        """

        self.load_attack()
        self.load_data()
        self.load_model_and_scheduler()

        assert self.norm is not None
        assert self.norm in [1, 2, float('inf')]
        log('[Training] using norm %g' % self.norm)

        assert self.model is not None
        assert self.scheduler is not None
        self.loop()


if __name__ == '__main__':
    program = TrainClassifierAdversarially()
    program.main()