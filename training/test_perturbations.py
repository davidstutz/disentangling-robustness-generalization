import os
import sys

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
import models
from common.log import log, Log, LogLevel
from common.state import State
from common import cuda
from common import paths
import common.torch
import common.numpy
import math
import torch
import numpy
import argparse


class TestPerturbations:
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

        self.test_codes = None
        """ (numpy.ndarray) Codes for testing. """

        self.perturbation_codes = None
        """ (numpy.ndarray) Perturbation codes for testing. """

        self.model = None
        """ (encoder.Encoder) Model to train. """

        self.perturbations = None
        """ (numpy.ndarray) Perturbations per test image. """

        self.original_accuracy = None
        """ (numpy.ndarray) Success of classifier. """

        self.transfer_accuracy = None
        """ (numpy.ndarray) Success of classifier. """

        self.original_success = None
        """ (numpy.ndarray) Success per test image. """

        self.transfer_success = None
        """ (numpy.ndarray) Success per test image. """

        if self.args.log_file:
            utils.makedir(os.path.dirname(self.args.log_file))
            Log.get_instance().attach(open(self.args.log_file, 'w'))

        log('-- ' + self.__class__.__name__)
        for key in vars(self.args):
            log('[Testing] %s=%s' % (key, str(getattr(self.args, key))))

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

        parser = argparse.ArgumentParser(description='Attack classifier.')
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing images.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-classifier_file', default=paths.state_file('classifier'), help='Snapshot state file of classifier.', type=str)
        parser.add_argument('-perturbations_file', default=paths.results_file('classifier/perturbations'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-original_success_file', default=paths.results_file('classifier/success'), help='HDF5 file containing success.', type=str)
        parser.add_argument('-transfer_success_file', default=paths.results_file('classifier/transfer_success', help='HDF5 file containing transfer success.'), type=str)
        parser.add_argument('-original_accuracy_file', default=paths.results_file('classifier/accuracy'), help='HDF5 file containing accuracy.', type=str)
        parser.add_argument('-transfer_accuracy_file', default=paths.results_file('classifier/transfer_accuracy', help='HDF5 file containing transfer accuracy.'), type=str)
        parser.add_argument('-log_file', default=paths.log_file('classifier/attacks'), help='Log file.', type=str)
        parser.add_argument('-batch_size', default=128, help='Batch size of attack.', type=int)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')

        # Some network parameters.
        parser.add_argument('-network_architecture', default='standard', help='Classifier architecture to use.', type=str)
        parser.add_argument('-network_activation', default='relu', help='Activation function to use.', type=str)
        parser.add_argument('-network_no_batch_normalization', default=False, help='Do not use batch normalization.', action='store_true')
        parser.add_argument('-network_channels', default=16, help='Channels of first convolutional layer, afterwards channels are doubled.', type=int)
        parser.add_argument('-network_dropout', default=False, action='store_true', help='Whether to use dropout.')
        parser.add_argument('-network_units', default='1024,1024,1024,1024', help='Units for MLP.')

        return parser

    def test(self):
        """
        Test classifier to identify valid samples to attack.
        """

        self.model.eval()
        assert self.model.training is False
        assert self.perturbation_codes.shape[0] == self.perturbations.shape[0]
        assert self.test_codes.shape[0] == self.test_images.shape[0]
        assert len(self.perturbations.shape) == 4
        assert len(self.test_images.shape) == 4

        perturbations_accuracy = None
        num_batches = int(math.ceil(self.perturbations.shape[0] / self.args.batch_size))

        for b in range(num_batches):
            b_start = b * self.args.batch_size
            b_end = min((b + 1) * self.args.batch_size, self.perturbations.shape[0])
            batch_perturbations = common.torch.as_variable(self.perturbations[b_start: b_end], self.args.use_gpu)
            batch_classes = common.torch.as_variable(self.perturbation_codes[b_start: b_end], self.args.use_gpu)
            batch_perturbations = batch_perturbations.permute(0, 3, 1, 2)

            output_classes = self.model(batch_perturbations)
            values, indices = torch.max(torch.nn.functional.softmax(output_classes, dim=1), dim=1)
            errors = torch.abs(indices - batch_classes)
            perturbations_accuracy = common.numpy.concatenate(perturbations_accuracy, errors.data.cpu().numpy())

            for n in range(batch_perturbations.size(0)):
                log('[Testing] %d: original success=%d, transfer accuracy=%d' % (n, self.original_success[b_start + n], errors[n].item()))

        self.transfer_success[perturbations_accuracy == 0] = -1
        self.transfer_success = self.transfer_success.reshape((self.N_samples, self.N_attempts))
        self.transfer_success = numpy.swapaxes(self.transfer_success, 0, 1)

        utils.makedir(os.path.dirname(self.args.transfer_success_file))
        utils.write_hdf5(self.args.transfer_success_file, self.transfer_success)
        log('[Testing] wrote %s' % self.args.transfer_success_file)

        num_batches = int(math.ceil(self.test_images.shape[0] / self.args.batch_size))
        for b in range(num_batches):
            b_start = b * self.args.batch_size
            b_end = min((b + 1) * self.args.batch_size, self.test_images.shape[0])
            batch_images = common.torch.as_variable(self.test_images[b_start: b_end], self.args.use_gpu)
            batch_classes = common.torch.as_variable(self.test_codes[b_start: b_end], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            output_classes = self.model(batch_images)
            values, indices = torch.max(torch.nn.functional.softmax(output_classes, dim=1), dim=1)
            errors = torch.abs(indices - batch_classes)

            self.transfer_accuracy = common.numpy.concatenate(self.transfer_accuracy, errors.data.cpu().numpy())

            if b % 100 == 0:
                log('[Testing] computing accuracy %d' % b)

        self.transfer_accuracy = self.transfer_accuracy == 0
        log('[Testing] original accuracy=%g' % (numpy.sum(self.original_accuracy)/float(self.original_accuracy.shape[0])))
        log('[Testing] transfer accuracy=%g' % (numpy.sum(self.transfer_accuracy)/float(self.transfer_accuracy.shape[0])))
        log('[Testing] accuracy difference=%g' % (numpy.sum(self.transfer_accuracy != self.original_accuracy)/float(self.transfer_accuracy.shape[0])))
        log('[Testing] accuracy difference on %d samples=%g' % (self.N_samples, numpy.sum(self.transfer_accuracy[:self.N_samples] != self.original_accuracy[:self.N_samples])/float(self.N_samples)))
        self.transfer_accuracy = numpy.logical_and(self.transfer_accuracy, self.original_accuracy)

        utils.makedir(os.path.dirname(self.args.transfer_accuracy_file))
        utils.write_hdf5(self.args.transfer_accuracy_file, self.transfer_accuracy)
        log('[Testing] wrote %s' % self.args.transfer_accuracy_file)

    def load_models(self):
        """
        Load models.
        """

        N_class = numpy.max(self.test_codes) + 1
        network_units = list(map(int, self.args.network_units.split(',')))
        log('[Testing] using %d input channels' % self.test_images.shape[3])
        self.model = models.Classifier(N_class, resolution=(self.test_images.shape[3], self.test_images.shape[1], self.test_images.shape[2]),
                                       architecture=self.args.network_architecture,
                                       activation=self.args.network_activation,
                                       batch_normalization=not self.args.network_no_batch_normalization,
                                       start_channels=self.args.network_channels,
                                       dropout=self.args.network_dropout,
                                       units=network_units)
        assert os.path.exists(self.args.classifier_file), 'state file %s not found' % self.args.classifier_file
        state = State.load(self.args.classifier_file)
        log('[Testing] read %s' % self.args.classifier_file)

        self.model.load_state_dict(state.model)
        if self.args.use_gpu and not cuda.is_cuda(self.model):
            log('[Testing] classifier is not CUDA')
            self.model = self.model.cuda()
        log('[Testing] loaded classifier')

        # !
        self.model.eval()
        log('[Testing] set classifier to eval')

    def load_data(self):
        """
        Load data.
        """

        self.test_images = utils.read_hdf5(self.args.test_images_file).astype(numpy.float32)
        if len(self.test_images.shape) < 4:
            self.test_images = numpy.expand_dims(self.test_images, axis=3)
        log('[Testing] read %s' % self.args.test_images_file)

        self.test_codes = utils.read_hdf5(self.args.test_codes_file).astype(numpy.int)
        self.test_codes = self.test_codes[:, self.args.label_index]
        log('[Testing] read %s' % self.args.test_codes_file)

        self.perturbations = utils.read_hdf5(self.args.perturbations_file).astype(numpy.float32)
        self.N_attempts = self.perturbations.shape[0]
        self.N_samples = self.perturbations.shape[1]
        self.perturbations = numpy.swapaxes(self.perturbations, 0, 1)
        if len(self.perturbations.shape) <= 4:
            self.perturbations = self.perturbations.reshape((self.perturbations.shape[0] * self.perturbations.shape[1], self.perturbations.shape[2], self.perturbations.shape[3], 1))
        else:
            self.perturbations = self.perturbations.reshape((self.perturbations.shape[0] * self.perturbations.shape[1], self.perturbations.shape[2], self.perturbations.shape[3], self.perturbations.shape[4]))

        log('[Testing] read %s' % self.args.perturbations_file)

        self.original_success = utils.read_hdf5(self.args.original_success_file)
        self.original_success = numpy.swapaxes(self.original_success, 0, 1)
        self.original_success = self.original_success.reshape((self.original_success.shape[0] * self.original_success.shape[1]))
        log('[Testing] read %s' % self.args.original_success_file)

        self.original_accuracy = utils.read_hdf5(self.args.original_accuracy_file)
        log('[Testing] read %s' % self.args.original_accuracy_file)

        self.perturbation_codes = numpy.repeat(self.test_codes[:self.N_samples], self.N_attempts, axis=0)
        self.transfer_success = numpy.copy(self.original_success)

    def main(self):
        """
        Main.
        """

        self.load_data()
        self.load_models()
        self.test()


if __name__ == '__main__':
    program = TestPerturbations()
    program.main()
