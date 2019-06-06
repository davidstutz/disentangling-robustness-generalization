import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common.log import log, logw
from common import utils
from common import paths
import argparse
import numpy


class CheckDataset:
    """
    Inspect the transformed images.
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

        parser = argparse.ArgumentParser(description='Inspect transformed images.')
        parser.add_argument('-database_file', default=paths.database_file(), type=str)
        parser.add_argument('-codes_file', default=paths.codes_file(), type=str)
        parser.add_argument('-theta_file', default=paths.theta_file(), type=str)
        parser.add_argument('-images_file', default=paths.images_file(), type=str)
        parser.add_argument('-train_codes_file', default=paths.train_codes_file(), type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), type=str)
        parser.add_argument('-train_theta_file', default=paths.train_theta_file(), type=str)
        parser.add_argument('-test_theta_file', default=paths.test_theta_file(), type=str)
        parser.add_argument('-train_images_file', default=paths.train_images_file(), type=str)
        parser.add_argument('-test_images_file', default=paths.test_images_file(), type=str)

        return parser

    def main(self):
        """
        Main method.
        """

        with logw('[Data] read %s' % self.args.database_file):
            database = utils.read_hdf5(self.args.database_file)
        with logw('[Data] read %s' % self.args.codes_file):
            codes = utils.read_hdf5(self.args.codes_file)
        with logw('[Data] read %s' % self.args.theta_file):
            theta = utils.read_hdf5(self.args.theta_file)
        with logw('[Data] read %s' % self.args.images_file):
            images = utils.read_hdf5(self.args.images_file)

        with logw('[Data] read %s' % self.args.train_codes_file):
            train_codes = utils.read_hdf5(self.args.train_codes_file)
        with logw('[Data] read %s' % self.args.train_theta_file):
            train_theta = utils.read_hdf5(self.args.train_theta_file)
        with logw('[Data] read %s' % self.args.train_images_file):
            train_images = utils.read_hdf5(self.args.train_images_file)

        with logw('[Data] read %s' % self.args.test_codes_file):
            test_codes = utils.read_hdf5(self.args.test_codes_file)
        with logw('[Data] read %s' % self.args.test_theta_file):
            test_theta = utils.read_hdf5(self.args.test_theta_file)
        with logw('[Data] read %s' % self.args.test_images_file):
            test_images = utils.read_hdf5(self.args.test_images_file)

        log('[Data] database: %s' % 'x'.join([str(dim) for dim in database.shape]))
        log('[Data] codes: %s' % 'x'.join([str(dim) for dim in codes.shape]))
        log('[Data] theta: %s' % 'x'.join([str(dim) for dim in theta.shape]))
        log('[Data] images: %s' % 'x'.join([str(dim) for dim in images.shape]))

        log('[Data] train_codes: %s' % 'x'.join([str(dim) for dim in train_codes.shape]))
        log('[Data] train_theta: %s' % 'x'.join([str(dim) for dim in train_theta.shape]))
        log('[Data] train_images: %s' % 'x'.join([str(dim) for dim in train_images.shape]))

        log('[Data] test_codes: %s' % 'x'.join([str(dim) for dim in test_codes.shape]))
        log('[Data] test_theta: %s' % 'x'.join([str(dim) for dim in test_theta.shape]))
        log('[Data] test_images: %s' % 'x'.join([str(dim) for dim in test_images.shape]))


if __name__ == '__main__':
    program = CheckDataset()
    program.main()
