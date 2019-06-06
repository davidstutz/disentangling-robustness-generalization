import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
import models
from common.log import log, Log
from common import paths
import common.numpy
import math
import numpy
import argparse


class ComputeSTNPerturbations:
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
        """ (numpy.ndarray) Font classes. """

        self.model = None
        """ (encoder.Encoder) Model to train. """

        self.perturbations = None
        """ (numpy.ndarray) Perturbations per test image. """

        self.perturbation_images = None
        """ (numpy.ndarray) Perturbation images. """

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

        parser = argparse.ArgumentParser(description='Attack decoder and classifier.')
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file with test images.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing test codes.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-perturbations_file', default=paths.results_file('decoder/perturbations'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-perturbation_images_file', default=paths.results_file('decoder/perturbation_images'), help='HDF5 file for perturbation images.', type=str)
        parser.add_argument('-log_file', default=paths.log_file('decoder/attacks'), help='Log file.', type=str)
        parser.add_argument('-batch_size', default=128, help='Batch size of attack.', type=int)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')

        # Some decoder parameters.
        parser.add_argument('-N_theta', default=6, help='Numer of transformations.', type=int)

        return parser

    def test(self):
        """
        Test classifier to identify valid samples to attack.
        """

        num_batches = int(math.ceil(self.perturbations.shape[0] / self.args.batch_size))

        for b in range(num_batches):
            b_start = b * self.args.batch_size
            b_end = min((b + 1) * self.args.batch_size, self.perturbations.shape[0])
            batch_images = common.torch.as_variable(self.test_images[b_start: b_end], self.args.use_gpu)
            batch_inputs = common.torch.as_variable(self.perturbations[b_start: b_end], self.args.use_gpu)

            self.model.set_image(batch_images)
            output_images = self.model(batch_inputs)

            output_images = numpy.squeeze(numpy.transpose(output_images.cpu().detach().numpy(), (0, 2, 3, 1)))
            self.perturbation_images = common.numpy.concatenate(self.perturbation_images, output_images)

            if b % 100 == 0:
                log('[Testing] computing perturbation images %d' % b)

        utils.makedir(os.path.dirname(self.args.perturbation_images_file))
        if len(self.perturbation_images.shape) > 3:
            self.perturbation_images = self.perturbation_images.reshape(self.N_samples, self.N_attempts, self.perturbation_images.shape[1], self.perturbation_images.shape[2], self.perturbation_images.shape[3])
        else:
            self.perturbation_images = self.perturbation_images.reshape(self.N_samples, self.N_attempts, self.perturbation_images.shape[1], self.perturbation_images.shape[2])
        self.perturbation_images = numpy.swapaxes(self.perturbation_images, 0, 1)
        utils.write_hdf5(self.args.perturbation_images_file, self.perturbation_images)
        log('[Testing] wrote %s' % self.args.perturbation_images_file)

    def load_model(self):
        """
        Load model.
        """

        decoder = models.STNDecoder(self.args.N_theta)
        log('[Testing] set up STN decoder')
        self.model = decoder

    def load_data(self):
        """
        Load data.
        """

        self.test_images = utils.read_hdf5(self.args.test_images_file)
        log('[Testing] read %s' % self.args.test_images_file)

        if len(self.test_images.shape) < 4:
            self.test_images = numpy.expand_dims(self.test_images, axis=3)
        self.resolution = self.test_images.shape[1]
        self.image_channels = self.test_images.shape[3]

        self.test_codes = utils.read_hdf5(self.args.test_codes_file)
        self.test_codes = self.test_codes[:, self.args.label_index]
        log('[Testing] read %s' % self.args.test_codes_file)

        self.perturbations = utils.read_hdf5(self.args.perturbations_file).astype(numpy.float32)
        self.N_attempts = self.perturbations.shape[0]
        self.N_samples = self.perturbations.shape[1]
        self.perturbations = numpy.swapaxes(self.perturbations, 0, 1)
        self.perturbations = self.perturbations.reshape((self.perturbations.shape[0] * self.perturbations.shape[1], -1))
        log('[Testing] read %s' % self.args.perturbations_file)

        self.test_codes = numpy.repeat(self.test_codes[:self.N_samples], self.N_attempts, axis=0)
        self.test_images = numpy.repeat(self.test_images[:self.N_samples], self.N_attempts, axis=0)

    def main(self):
        """
        Main.
        """

        self.load_data()
        self.load_model()
        self.test()


if __name__ == '__main__':
    program = ComputeSTNPerturbations()
    program.main()