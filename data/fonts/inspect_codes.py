import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common import utils
from common.log import log
from common import paths

import argparse
import numpy
numpy.set_printoptions(edgeitems=50)


class InspectCodes:
    """
    Inspect the generated codes.
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

        parser = argparse.ArgumentParser(description='Inspect generated codes.')
        parser.add_argument('-theta_file', default=paths.theta_file(), help='HDF5 file containing transformations.', type=str)
        parser.add_argument('-codes_file', default=paths.codes_file(), help='HDF5 file containing codes.', type=str)
        return parser

    def main(self):
        """
        Main method.
        """

        theta = utils.read_hdf5(self.args.theta_file)
        log('[Data] read %s' % self.args.theta_file)

        if theta.shape[1] == 1:
            log('[Data] theta min: [%f]' % (
                numpy.min(theta[:, 0])))
            log('[Data] theta max: [%f]' % (
                numpy.max(theta[:, 0])))
        elif theta.shape[1] == 2:
            log('[Data] theta min: [%f, %f]' % (
                numpy.min(theta[:, 0]), numpy.min(theta[:, 1])))
            log('[Data] theta max: [%f, %f]' % (
                numpy.max(theta[:, 0]), numpy.max(theta[:, 1])))
        elif theta.shape[1] == 3:
            log('[Data] theta min: [%f, %f, %f]' % (
                numpy.min(theta[:, 0]), numpy.min(theta[:, 1]), numpy.min(theta[:, 2])))
            log('[Data] theta max: [%f, %f, %f]' % (
                numpy.max(theta[:, 0]), numpy.max(theta[:, 1]), numpy.max(theta[:, 2])))
        elif theta.shape[1] == 4:
            log('[Data] theta min: [%f, %f, %f, %f]' % (
                numpy.min(theta[:, 0]), numpy.min(theta[:, 1]), numpy.min(theta[:, 2]),
                numpy.min(theta[:, 3])))
            log('[Data] theta max: [%f, %f, %f, %f]' % (
                numpy.max(theta[:, 0]), numpy.max(theta[:, 1]), numpy.max(theta[:, 2]),
                numpy.max(theta[:, 3])))
        elif theta.shape[1] == 6:
            log('[Data] theta min: [%f, %f, %f, %f, %f, %f]' % (
                numpy.min(theta[:, 0]), numpy.min(theta[:, 1]), numpy.min(theta[:, 2]),
                numpy.min(theta[:, 3]), numpy.min(theta[:, 4]), numpy.min(theta[:, 5])))
            log('[Data] theta max: [%f, %f, %f, %f, %f, %f]' % (
                numpy.max(theta[:, 0]), numpy.max(theta[:, 1]), numpy.max(theta[:, 2]),
                numpy.max(theta[:, 3]), numpy.max(theta[:, 4]), numpy.max(theta[:, 5])))

        codes = utils.read_hdf5(self.args.codes_file)
        log('[Data] read %s' % self.args.codes_file)
        print(codes)

        #latent = numpy.concatenate((codes.reshape((codes.shape[0], 1)), theta), axis=1)
        #log(latent)


if __name__ == '__main__':
    program = InspectCodes()
    program.main()
