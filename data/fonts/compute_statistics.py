import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common.log import log
from common import utils
from common import plot
from common import paths

import argparse
import numpy


class ComputeStatistics:
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

        parser = argparse.ArgumentParser(description='Compute data statistics.')
        parser.add_argument('-images_file', default=paths.train_images_file(), help='HDF5 file created.', type=str)
        return parser

    def main(self):
        """
        Main method.
        """

        images = utils.read_hdf5(self.args.images_file)
        log('[Data] read %s' % self.args.images_file)
        log('[Data] #images: %d' % images.shape[0])

        images = images.reshape((images.shape[0], -1))
        l2_norm = numpy.average(numpy.linalg.norm(images, ord=2, axis=1))
        l1_norm = numpy.average(numpy.linalg.norm(images, ord=1, axis=1))
        linf_norm = numpy.average(numpy.linalg.norm(images, ord=float('inf'), axis=1))
        log('[Data] average L_2 norm: %g' % l2_norm)
        log('[Data] average L_1 norm: %g' % l1_norm)
        log('[Data] average L_inf norm: %g' % linf_norm)


if __name__ == '__main__':
    program = ComputeStatistics()
    program.main()
