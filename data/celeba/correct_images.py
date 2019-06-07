import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common.log import log
from common import utils
from common import paths
import argparse
import imageio
from matplotlib import pyplot
import skimage.transform
import numpy


class CorrectImages:
    """
    Read, convert and resize images.
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
            log('%s=%s' % (key, str(getattr(self.args, key))))

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser()

        return parser

    def main(self):
        """
        Main.
        """

        train_images_file = paths.celeba_train_images_file()
        test_images_file = paths.celeba_test_images_file()

        assert os.path.exists(train_images_file)
        assert os.path.exists(test_images_file)

        train_images = utils.read_hdf5(train_images_file)
        log('read %s' % train_images_file)

        test_images = utils.read_hdf5(test_images_file)
        log('read %s' % test_images_file)

        log('[Data] before train: %g %g' % (numpy.min(train_images), numpy.max(train_images)))
        log('[Data] before test: %g %g' % (numpy.min(train_images), numpy.max(train_images)))

        train_images *= 255
        test_images *= 255

        log('[Data] after train: %g %g' % (numpy.min(train_images), numpy.max(train_images)))
        log('[Data] after test: %g %g' % (numpy.min(train_images), numpy.max(train_images)))

        utils.write_hdf5(train_images_file, train_images.astype(numpy.float32))
        log('[Data] wrote %s' % train_images_file)
        utils.write_hdf5(test_images_file, test_images.astype(numpy.float32))
        log('[Data] wrote %s' % test_images_file)


if __name__ == '__main__':
    program = CorrectImages()
    program.main()