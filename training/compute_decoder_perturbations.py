import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
import models
from common.log import log, Log
from common import paths
import common.numpy
import math
import torch
import numpy
import argparse


class ComputeDecoderPerturbations:
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

        self.test_fonts = None
        """ (numpy.ndarray) Font classes. """

        self.test_classes = None
        """ (numpy.ndarray) Character classes. """

        self.N_attempts = None
        """ (int) Number of attempts. """

        self.N_samples = None
        """ (int) Number of samples. """

        self.N_font = None
        """ (int) Number of fonts. """

        self.N_class = None
        """ (int) Number of classes. """

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
        parser.add_argument('-database_file', default=paths.database_file(), help='HDF5 file containing font prototype images.', type=str)
        parser.add_argument('-test_theta_file', default=paths.test_theta_file(), help='HDF5 file for thetas.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-perturbations_file', default=paths.results_file('decoder/perturbations'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-perturbation_images_file', default=paths.results_file('decoder/perturbation_images'), help='HDF5 file for perturbation images.', type=str)
        parser.add_argument('-log_file', default=paths.log_file('decoder/attacks'), help='Log file.', type=str)
        parser.add_argument('-batch_size', default=128, help='Batch size of attack.', type=int)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')

        return parser

    def test(self):
        """
        Test classifier to identify valid samples to attack.
        """

        num_batches = int(math.ceil(self.perturbations.shape[0] / self.args.batch_size))

        for b in range(num_batches):
            b_start = b * self.args.batch_size
            b_end = min((b + 1) * self.args.batch_size, self.perturbations.shape[0])
            batch_fonts = self.test_fonts[b_start: b_end]
            batch_classes = self.test_classes[b_start: b_end]
            batch_code = numpy.concatenate((common.numpy.one_hot(batch_fonts, self.N_font), common.numpy.one_hot(batch_classes, self.N_class)), axis=1).astype(numpy.float32)

            batch_inputs = common.torch.as_variable(self.perturbations[b_start: b_end], self.args.use_gpu)
            batch_code = common.torch.as_variable(batch_code, self.args.use_gpu)

            # This basically allows to only optimize over theta, keeping the font/class code fixed.
            self.model.set_code(batch_code)
            output_images = self.model(batch_inputs)

            output_images = numpy.squeeze(numpy.transpose(output_images.cpu().detach().numpy(), (0, 2, 3, 1)))
            self.perturbation_images = common.numpy.concatenate(self.perturbation_images, output_images)

            if b%100 == 0:
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

        database = utils.read_hdf5(self.args.database_file).astype(numpy.float32)
        log('[Testing] read %sd' % self.args.database_file)

        self.N_font = database.shape[0]
        self.N_class = database.shape[1]

        database = database.reshape((database.shape[0] * database.shape[1], database.shape[2], database.shape[3]))
        database = torch.from_numpy(database)
        if self.args.use_gpu:
            database = database.cuda()
        database = torch.autograd.Variable(database, False)

        test_theta = utils.read_hdf5(self.args.test_theta_file)
        N_theta = test_theta.shape[1]
        log('[Testing] read %s' % self.args.test_theta_file)

        log('[Testing] using %d N_theta' % N_theta)
        self.model = models.AlternativeOneHotDecoder(database, self.N_font, self.N_class, N_theta)
        self.model.eval()

    def load_data(self):
        """
        Load data.
        """

        test_codes = utils.read_hdf5(self.args.test_codes_file).astype(numpy.int)
        self.test_fonts = test_codes[:, 1]
        self.test_classes = test_codes[:, 2]
        log('[Testing] read %s' % self.args.test_codes_file)

        self.perturbations = utils.read_hdf5(self.args.perturbations_file).astype(numpy.float32)
        self.N_attempts = self.perturbations.shape[0]
        self.N_samples = self.perturbations.shape[1]
        self.perturbations = numpy.swapaxes(self.perturbations, 0, 1)
        self.perturbations = self.perturbations.reshape((self.perturbations.shape[0] * self.perturbations.shape[1], -1))
        log('[Testing] read %s' % self.args.perturbations_file)

        self.test_fonts = numpy.repeat(self.test_fonts[:self.N_samples], self.N_attempts, axis=0)
        self.test_classes = numpy.repeat(self.test_classes[:self.N_samples], self.N_attempts, axis=0)

    def main(self):
        """
        Main.
        """

        self.load_data()
        self.load_model()
        self.test()


if __name__ == '__main__':
    program = ComputeDecoderPerturbations()
    program.main()