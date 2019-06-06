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
import torch
import numpy
import argparse
import math


class TestClassifier:
    """
    Test a trained classifier.
    """

    def __init__(self, args=None):
        """
        Initialize.

        :param args: arguments
        :type args: list
        """

        self.args = None
        """ Arguments of program. """

        self.test_images = None
        """ (numpy.ndarray) Images to test on. """

        self.test_codes = None
        """ (numpy.ndarray) Codes for testing. """

        self.model = None
        """ (encoder.Encoder) Model to train. """

        self.loss = None
        """ (float) Will hold evalauted loss. """

        self.error = None
        """ (float) Will hold evaluated error. """

        self.accuracy = None
        """ (numpy.ndarray) Will hold success. """

        self.results = dict()
        """ (dict) Will hold evaluation results. """

        parser = self.get_parser()
        if args is not None:
            self.args = parser.parse_args(args)
        else:
            self.args = parser.parse_args() # sys.args

        utils.makedir(os.path.dirname(self.args.log_file))
        if self.args.log_file:
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

        parser = argparse.ArgumentParser(description='Test classifier.')
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-state_file', default=paths.state_file('classifier'), help='Snapshot state file.', type=str)
        parser.add_argument('-accuracy_file', default=paths.results_file('classifier/accuracy'), help='Correctly classified test samples of classifier.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-results_file', default='', help='Results file for evaluation.', type=str)
        parser.add_argument('-batch_size', default=64, help='Batch size.', type=int)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')
        parser.add_argument('-log_file', default=paths.log_file('test_classifier'), help='Log file.', type=str)

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
        Test the model.
        """

        assert self.model is not None
        assert self.model.training is False
        assert self.test_images.shape[0] == self.test_codes.shape[0], 'number of samples have to match'

        self.loss = 0.
        self.error = 0.
        num_batches = int(math.ceil(self.test_images.shape[0]/self.args.batch_size))

        for b in range(num_batches):
            b_start = b*self.args.batch_size
            b_end = min((b + 1)*self.args.batch_size, self.test_images.shape[0])
            batch_images = common.torch.as_variable(self.test_images[b_start: b_end], self.args.use_gpu)
            batch_classes = common.torch.as_variable(self.test_codes[b_start: b_end], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            output_classes = self.model(batch_images)
            e = torch.nn.functional.cross_entropy(output_classes, batch_classes, size_average=True)
            self.loss += e.item()

            values, indices = torch.max(torch.nn.functional.softmax(output_classes, dim=1), dim=1)
            errors = torch.abs(indices - batch_classes)
            e = torch.sum(errors > 0).float()/batch_classes.size()[0]
            self.error += e.item()

            self.accuracy = common.numpy.concatenate(self.accuracy, errors.data.cpu().numpy())

        self.loss /= num_batches
        self.error /= num_batches
        log('[Testing] test loss %g; test error %g' % (self.loss, self.error))

        self.accuracy = self.accuracy == 0
        if self.args.accuracy_file:
            utils.write_hdf5(self.args.accuracy_file, self.accuracy)
            log('[Testing] wrote %s' % self.args.accuracy_file)

        accuracy = numpy.sum(self.accuracy) / self.accuracy.shape[0]
        if numpy.abs(1 - accuracy - self. error) < 1e-4:
            log('[Testing] accuracy file is with %g accuracy correct' % accuracy)

        self.results = {
            'loss': self.loss,
            'error': self.error,
        }
        if self.args.results_file:
            utils.write_pickle(self.args.results_file, self.results)
            log('[Testing] wrote %s' % self.args.results_file)

    def main(self):
        """
        Main which should be overwritten.
        """

        self.test_images = utils.read_hdf5(self.args.test_images_file).astype(numpy.float32)
        log('[Testing] read %s' % self.args.test_images_file)

        # For handling both color and gray images.
        if len(self.test_images.shape) < 4:
            self.test_images = numpy.expand_dims(self.test_images, axis=3)
            log('[Testing] no color images, adjusted size')
        self.resolution = self.test_images.shape[2]
        log('[Testing] resolution %d' % self.resolution)

        self.test_codes = utils.read_hdf5(self.args.test_codes_file).astype(numpy.int)
        self.test_codes = self.test_codes[:, self.args.label_index]
        log('[Testing] read %s' % self.args.test_codes_file)

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

        assert os.path.exists(self.args.state_file), 'state file %s not found' % self.args.state_file
        state = State.load(self.args.state_file)
        log('[Testing] read %s' % self.args.state_file)

        self.model.load_state_dict(state.model)
        if self.args.use_gpu and not cuda.is_cuda(self.model):
            log('[Testing] model is not CUDA')
            self.model = self.model.cuda()
        log('[Testing] loaded model')

        self.model.eval()
        log('[Testing] set classifier to eval')

        self.test()


if __name__ == '__main__':
    program = TestClassifier()
    program.main()