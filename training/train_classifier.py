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
import torch
import numpy
import argparse
import math
import functools
if utils.display():
    from common import plot
    from common import vis


class TrainClassifier:
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

        self.args = None
        """ Arguments of program. """

        parser = self.get_parser()
        if args is not None:
            self.args = parser.parse_args(args)
        else:
            self.args = parser.parse_args()

        self.train_images = None
        """ (numpy.ndarray) Images to train on. """

        self.train_codes = None
        """ (numpy.ndarray) Codes for training. """

        self.val_images = None
        """ (numpy.ndarray) Images to validate on. """

        self.val_codes = None
        """ (numpy.ndarray) Codes to validate on. """

        self.val_error = None
        """ (float) Validation error. """

        self.test_images = None
        """ (numpy.ndarray) Images to test on. """

        self.test_codes = None
        """ (numpy.ndarray) Codes for testing. """

        self.model = None
        """ (encoder.Encoder) Model to train. """

        self.scheduler = None
        """ (Scheduler) Scheduler for training. """

        self.train_statistics = numpy.zeros((0, 6))
        """ (numpy.ndarray) Will hold training statistics. """

        self.test_statistics = numpy.zeros((0, 5))
        """ (numpy.ndarray) Will hold testing statistics. """

        self.epoch = 0
        """ (int) Current epoch. """

        self.N_class = None
        """ (int) Number of classes. """

        self.results = dict()
        """ (dict) Results. """

        utils.makedir(os.path.dirname(self.args.state_file))
        utils.makedir(os.path.dirname(self.args.log_file))

        if self.args.log_file:
            Log.get_instance().attach(open(self.args.log_file, 'w'))

        log('-- ' + self.__class__.__name__)
        for key in vars(self.args):
            log('[Training] %s=%s' % (key, str(getattr(self.args, key))))

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
        parser.add_argument('-train_images_file', default=paths.train_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-train_codes_file', default=paths.train_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-state_file', default=paths.state_file('classifier'), help='Snapshot state file.', type=str)
        parser.add_argument('-log_file', default=paths.log_file('classifier'), help='Log file.', type=str)
        parser.add_argument('-training_file', default=paths.results_file('training'), help='Training statistics file.', type=str)
        parser.add_argument('-testing_file', default=paths.results_file('testing'), help='Testing statistics file.', type=str)
        parser.add_argument('-loss_file', default=paths.image_file('loss'), help='Loss plot file.', type=str)
        parser.add_argument('-error_file', default=paths.image_file('error'), help='Error plot file.', type=str)
        parser.add_argument('-gradient_file', default=paths.image_file('gradient'), help='Gradient plot file.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-training_samples', default=-1, help='Number of samples used for training.', type=int)
        parser.add_argument('-validation_samples', default=0, help='Number of samples for validation.', type=int)
        parser.add_argument('-test_samples', default=-1, help='Number of samples for validation.', type=int)
        parser.add_argument('-early_stopping', default=False, action='store_true', help='Use early stopping.')
        parser.add_argument('-random_samples', default=False, action='store_true', help='Randomize the subsampling of the training set.')
        parser.add_argument('-batch_size', default=64, help='Batch size.', type=int)
        parser.add_argument('-epochs', default=10, help='Number of epochs.', type=int)
        parser.add_argument('-weight_decay', default=0.0001, help='Weight decay importance.', type=float)
        parser.add_argument('-logit_decay', default=0, help='Logit decay importance.', type=float)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')
        parser.add_argument('-skip', default=5, help='Verbosity in iterations.', type=int)
        parser.add_argument('-lr', default=0.01, type=float, help='Base learning rate.')
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

        return parser

    def debug(self, filename, images, cmap='gray'):
        """
        Simple debugging.

        :param filename: filename in debug_directory
        :type filename: str
        :param images: images
        :type images: numpy.ndarray
        """

        if type(images) == torch.autograd.Variable or type(images) == torch.Tensor:
            images = images.cpu().detach().numpy()

        assert type(images) == numpy.ndarray
        assert images.shape[3] == 1 or images.shape[3] == 3

        if utils.display() and self.args.debug_directory:
            utils.makedir(self.args.debug_directory)
            vis.mosaic(os.path.join(self.args.debug_directory, filename), images, cmap=cmap)

    def error(self, batch_classes, output_classes):
        """
        Accuracy.

        :param batch_classes: predicted classes
        :type batch_classes: torch.autograd.Variable
        :param output_classes: target classes
        :type output_classes: torch.autograd.Variable
        :return: accuracy
        :rtype: torch.autograd.Variable
        """

        values, indices = torch.max(torch.nn.functional.softmax(output_classes, dim=1), dim=1)
        errors = torch.abs(indices - batch_classes)
        return torch.sum(errors > 0).float() / batch_classes.size()[0]

    def loss(self, batch_classes, output_classes):
        """
        Loss.

        :param batch_classes: predicted classes
        :type batch_classes: torch.autograd.Variable
        :param output_classes: target classes
        :type output_classes: torch.autograd.Variable
        :return: error
        :rtype: torch.autograd.Variable
        """

        return torch.nn.functional.cross_entropy(output_classes, batch_classes, size_average=True) \
            + self.args.logit_decay*torch.max(torch.sum(torch.abs(output_classes), dim=1))

    def train(self):
        """
        Train for one epoch.
        """

        self.model.train()
        log('[Training] %d set classifier to train' % self.epoch)
        assert self.model.training is True

        num_batches = int(math.ceil(self.train_images.shape[0]/self.args.batch_size))
        permutation = numpy.random.permutation(self.train_images.shape[0])

        for b in range(num_batches):
            self.scheduler.update(self.epoch, float(b)/num_batches)

            perm = numpy.take(permutation, range(b*self.args.batch_size, (b+1)*self.args.batch_size), mode='wrap')
            assert perm.shape[0] == self.args.batch_size

            batch_images = common.torch.as_variable(self.train_images[perm], self.args.use_gpu)
            batch_true_classes = common.torch.as_variable(self.train_codes[perm], self.args.use_gpu)
            batch_training_classes = common.torch.as_variable(self.train_codes[perm], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            output_classes = self.model(batch_images)

            self.scheduler.optimizer.zero_grad()
            loss = self.loss(batch_training_classes, output_classes)
            loss.backward()
            self.scheduler.optimizer.step()
            loss = loss.item()

            gradient = torch.mean(torch.abs(list(self.model.parameters())[0].grad))
            gradient = gradient.item()

            error = self.error(batch_true_classes, output_classes)
            error = error.item()

            iteration = self.epoch*num_batches + b + 1
            self.train_statistics = numpy.vstack((self.train_statistics, numpy.array([
                iteration,
                iteration*self.args.batch_size,
                min(num_batches, iteration)*self.args.batch_size,
                loss,
                error,
                gradient
            ])))

            if b % self.args.skip == self.args.skip // 2:
                log('[Training] %d | %d: %g (%g) [%g]' % (
                    self.epoch,
                    b,
                    numpy.mean(self.train_statistics[max(0, iteration - self.args.skip):iteration, 3]),
                    numpy.mean(self.train_statistics[max(0, iteration - self.args.skip):iteration, 4]),
                    numpy.mean(self.train_statistics[max(0, iteration - self.args.skip):iteration, -1]),
                ))

        # Only debug last iterations for efficiency!
        self.debug('clean.png', batch_images.permute(0, 2, 3, 1))

    def test(self):
        """
        Test the model.
        """

        self.model.eval()
        log('[Training] %d set classifier to eval' % self.epoch)
        assert self.model.training is False

        loss = 0
        error = 0
        num_batches = int(math.ceil(self.args.test_samples/self.args.batch_size))

        for b in range(num_batches):
            perm = numpy.take(range(self.args.test_samples), range(b*self.args.batch_size, (b+1)*self.args.batch_size), mode='clip')
            batch_images = common.torch.as_variable(self.test_images[perm], self.args.use_gpu)
            batch_classes = common.torch.as_variable(self.test_codes[perm], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            output_classes = self.model(batch_images)
            e = self.loss(batch_classes, output_classes)
            loss += e.data # 0-dim tensor
            e = self.error(batch_classes, output_classes)
            error += e.data

        loss /= num_batches
        error /= num_batches
        log('[Training] %d: test %g (%g)' % (self.epoch, loss, error))

        num_batches = int(math.ceil(self.train_images.shape[0]/self.args.batch_size))
        iteration = self.epoch*num_batches
        self.test_statistics = numpy.vstack((self.test_statistics, numpy.array([[
            iteration,
            iteration*self.args.batch_size,
            min(num_batches, iteration) * self.args.batch_size,
            loss,
            error,
        ]])))

    def validate(self):
        """
        Validate for early stopping.
        """

        self.model.eval()
        log('[Training] %d set classifier to eval' % self.epoch)
        assert self.model.training is False

        loss = 0
        error = 0
        num_batches = int(math.ceil(self.val_images.shape[0] / self.args.batch_size))

        for b in range(num_batches):
            perm = numpy.take(range(self.val_images.shape[0]), range(b*self.args.batch_size, (b+1)*self.args.batch_size), mode='clip')
            batch_images = common.torch.as_variable(self.val_images[perm], self.args.use_gpu)
            batch_classes = common.torch.as_variable(self.val_codes[perm], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            output_classes = self.model(batch_images)

            e = self.loss(batch_classes, output_classes)
            loss += e.item()
            e = self.error(batch_classes, output_classes)
            error += e.item()

        loss /= num_batches
        error /= num_batches
        log('[Training] %d: val %g (%g)' % (self.epoch, loss, error))

        if self.val_error is None or error < self.val_error:
            self.val_error = error
            State.checkpoint(self.model, self.scheduler.optimizer, self.epoch, self.args.state_file + '.es')
            log('[Training] %d: early stopping checkoint' % self.epoch)

    def plot(self):
        """
        Plot error and accuracy.
        """

        if self.args.loss_file:
            plot.line(self.args.loss_file, [
                self.train_statistics[:, 0],
                self.test_statistics[:, 0],
            ], [
                self.train_statistics[:, 3],
                self.test_statistics[:, 3],
            ], [
                'Train Loss',
                'Test Loss',
            ], title='Loss during Training', xlabel='Iteration', ylabel='Loss')
        if self.args.error_file:
            plot.line(self.args.error_file, [
                self.train_statistics[:, 0],
                self.test_statistics[:, 0],
            ], [
                self.train_statistics[:, 4],
                self.test_statistics[:, 4],
            ], [
                'Train Error',
                'Test Error',
            ], title='Error during Training', xlabel='Iteration', ylabel='Error')
        if self.args.gradient_file:
            plot.line(self.args.gradient_file, [
                self.train_statistics[:, 0],
            ], [
                self.train_statistics[:, -1],
            ], [
                'Train Gradient Norm',
            ], title='Gradient during Training', xlable='Iteration', ylable='Gradient Norm')

    def loop(self):
        """
        Main loop for training and testing, saving ...
        """

        while self.epoch < self.args.epochs:
            log('[Training] %s' % self.scheduler.report())

            # Note that we test first, to also get the error of the untrained model.
            testing = elapsed(functools.partial(self.test))
            training = elapsed(functools.partial(self.train))
            log('[Training] %gs training, %gs testing' % (training, testing))

            if self.args.early_stopping:
                validation = elapsed(functools.partial(self.validate))
                log('[Training] %gs validation' % validation)

            # Save model checkpoint after each epoch.
            utils.remove(self.args.state_file + '.%d' % (self.epoch - 1))
            State.checkpoint(self.model, self.scheduler.optimizer, self.epoch, self.args.state_file + '.%d' % self.epoch)
            log('[Training] %d: checkpoint' % self.epoch)
            torch.cuda.empty_cache() # necessary?

            # Save statistics and plots.
            if self.args.training_file:
                utils.write_hdf5(self.args.training_file, self.train_statistics)
                log('[Training] %d: wrote %s' % (self.epoch, self.args.training_file))
            if self.args.testing_file:
                utils.write_hdf5(self.args.testing_file, self.test_statistics)
                log('[Training] %d: wrote %s' % (self.epoch, self.args.testing_file))

            if utils.display():
                self.plot()
            self.epoch += 1 # !

        # Final testing.
        testing = elapsed(functools.partial(self.test))
        log('[Training] %gs testing' % (testing))

        # Save model checkpoint after each epoch.
        utils.remove(self.args.state_file + '.%d' % (self.epoch - 1))
        State.checkpoint(self.model, self.scheduler.optimizer, self.epoch, self.args.state_file)
        log('[Training] %d: checkpoint' % self.epoch)

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
        self.resolution = self.train_images.shape[2]
        log('[Training] resolution %d' % self.resolution)

        self.train_codes = utils.read_hdf5(self.args.train_codes_file).astype(numpy.int)
        assert self.train_codes.shape[1] >= self.args.label_index + 1
        self.train_codes = self.train_codes[:, self.args.label_index]
        log('[Training] read %s' % self.args.train_codes_file)

        self.test_codes = utils.read_hdf5(self.args.test_codes_file).astype(numpy.int)
        assert self.test_codes.shape[1] >= self.args.label_index + 1
        self.test_codes = self.test_codes[:, self.args.label_index]
        log('[Training] read %s' % self.args.test_codes_file)

        assert self.train_codes.shape[0] == self.train_images.shape[0]
        assert self.test_codes.shape[0] == self.test_images.shape[0]

        # Select subset of samples
        if self.args.training_samples < 0:
            self.args.training_samples = self.train_images.shape[0]
        else:
            self.args.training_samples = min(self.args.training_samples, self.train_images.shape[0])
        self.N_class = numpy.max(self.train_codes) + 1
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

        log('[Training] found %d classes' % self.N_class)
        if self.args.random_samples:
            perm = numpy.random.permutation(self.train_images.shape[0]//10)
            perm = perm[:self.args.training_samples//10]
            perm = numpy.repeat(perm, self.N_class, axis=0)*10 + numpy.tile(numpy.array(range(self.N_class)), (perm.shape[0]))
            self.train_images = self.train_images[perm]
            self.train_codes = self.train_codes[perm]
        else:
            self.train_images = self.train_images[:self.args.training_samples]
            self.train_codes = self.train_codes[:self.args.training_samples]

        # Check that the dataset is balanced.
        number_samples = self.train_codes.shape[0]//self.N_class
        for c in range(self.N_class):
            number_samples_ = numpy.sum(self.train_codes == c)
            if number_samples_ != number_samples:
                log('[Training] dataset not balanced, class %d should have %d samples but has %d' % (c, number_samples, number_samples_), LogLevel.WARNING)

    def load_model_and_scheduler(self):
        """
        Load model.
        """

        params = {
            'lr': self.args.lr,
            'lr_decay': self.args.lr_decay,
            'lr_min': 0.0000001,
            'weight_decay': self.args.weight_decay,
        }

        log('[Training] using %d input channels' % self.train_images.shape[3])
        network_units = list(map(int, self.args.network_units.split(',')))
        self.model = models.Classifier(self.N_class, resolution=(self.train_images.shape[3], self.train_images.shape[1], self.train_images.shape[2]),
                                       architecture=self.args.network_architecture,
                                       activation=self.args.network_activation,
                                       batch_normalization=not self.args.network_no_batch_normalization,
                                       start_channels=self.args.network_channels,
                                       dropout=self.args.network_dropout,
                                       units=network_units)

        self.epoch = 0
        if os.path.exists(self.args.state_file):
            state = State.load(self.args.state_file)
            log('[Training] loaded %s' % self.args.state_file)

            self.model.load_state_dict(state.model)

            # needs to be done before costructing optimizer.
            if self.args.use_gpu and not cuda.is_cuda(self.model):
                self.model = self.model.cuda()
                log('[Training] model is not CUDA')
            log('[Training] loaded model')

            optimizer = torch.optim.Adam(self.model.parameters(), params['lr'])
            optimizer.load_state_dict(state.optimizer)
            self.scheduler = ADAMScheduler(optimizer, **params)

            self.epoch = state.epoch + 1
            self.scheduler.update(self.epoch)

            assert os.path.exists(self.args.training_file) and os.path.exists(self.args.testing_file)
            self.train_statistics = utils.read_hdf5(self.args.training_file)
            log('[Training] read %s' % self.args.training_file)
            self.test_statistics = utils.read_hdf5(self.args.testing_file)
            log('[Training] read %s' % self.args.testing_file)

            if utils.display():
                self.plot()
        else:
            if self.args.use_gpu and not cuda.is_cuda(self.model):
                self.model = self.model.cuda()
                log('[Training] model is not CUDA')
            log('[Training] did not load model, using new one')

            self.scheduler = ADAMScheduler(self.model.parameters(), **params)
            self.scheduler.initialize()  # !

        log(self.model)

    def main(self):
        """
        Main which should be overwritten.
        """

        self.load_data()
        self.load_model_and_scheduler()

        assert self.model is not None
        assert self.scheduler is not None
        self.loop()


if __name__ == '__main__':
    program = TrainClassifier()
    program.main()