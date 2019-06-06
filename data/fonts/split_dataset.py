import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common import utils
from common.log import log
from common import paths

import argparse
import numpy


class SplitDataset:
    """
    Split dataset into training and testing.
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

        log('-- ' + self.__class__.__name__)
        for key in vars(self.args):
            log('[Data] %s=%s' % (key, str(getattr(self.args, key))))

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Split generated dataset into training and test sets.')
        parser.add_argument('-codes_file', default=paths.codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-theta_file', default=paths.theta_file(), help='HDF5 file containing transformations.', type=str)
        parser.add_argument('-images_file', default=paths.images_file(), help='HDF5 file containing transformed images.', type=str)
        parser.add_argument('-train_codes_file', default=paths.train_codes_file(), help='HDF5 file containing transformed images.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing transformed images.', type=str)
        parser.add_argument('-train_theta_file', default=paths.train_theta_file(), help='HDF5 file containing transformed images.', type=str)
        parser.add_argument('-test_theta_file', default=paths.test_theta_file(), help='HDF5 file containing transformed images.', type=str)
        parser.add_argument('-train_images_file', default=paths.train_images_file(), help='HDF5 file containing transformed images.', type=str)
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing transformed images.', type=str)
        parser.add_argument('-N_train', default=960000, help='Train/test split.', type=int)
        return parser

    def main(self):
        """
        Main method.
        """

        codes = utils.read_hdf5(self.args.codes_file)
        log('[Data] read %s' % self.args.codes_file)

        theta = utils.read_hdf5(self.args.theta_file)
        log('[Data] read %s' % self.args.theta_file)

        images = utils.read_hdf5(self.args.images_file)
        log('[Data] read %s' % self.args.images_file)

        #
        # The set is not splitted randomly or so.
        # This simplifies training set subselection while enforcing balanced datasets.
        # For example, for 10 classes, every subset that is a multiple of 10 will
        # be balanced by construction.
        #

        N = codes.shape[0]
        N_train = self.args.N_train

        train_codes = codes[:N_train]
        test_codes = codes[N_train:]

        train_theta = theta[:N_train]
        test_theta = theta[N_train:]

        train_images = images[:N_train]
        test_images = images[N_train:]

        utils.write_hdf5(self.args.train_codes_file, train_codes)
        log('[Data] wrote %s' % self.args.train_codes_file)
        utils.write_hdf5(self.args.test_codes_file, test_codes)
        log('[Data] wrote %s' % self.args.test_codes_file)

        utils.write_hdf5(self.args.train_theta_file, train_theta)
        log('[Data] wrote %s' % self.args.train_theta_file)
        utils.write_hdf5(self.args.test_theta_file, test_theta)
        log('[Data] wrote %s' % self.args.test_theta_file)

        utils.write_hdf5(self.args.train_images_file, train_images)
        log('[Data] wrote %s' % self.args.train_images_file)
        utils.write_hdf5(self.args.test_images_file, test_images)
        log('[Data] wrote %s' % self.args.test_images_file)


if __name__ == '__main__':
    program = SplitDataset()
    program.main()
