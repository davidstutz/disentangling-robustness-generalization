import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common import utils
from common.log import log
from common import paths

import argparse
import numpy
import math
numpy.set_printoptions(edgeitems=50)


class GenerateCodes:
    """
    Generate codes - mainly needed for randomly sampling transformation parameters.
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

        parser = argparse.ArgumentParser(description='Generate latent codes.')
        parser.add_argument('-database_file', default=paths.database_file(), help='HDF5 file created.', type=str)
        parser.add_argument('-codes_file', default=paths.codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-theta_file', default=paths.theta_file(), help='HDF5 file containing transformations.', type=str)
        parser.add_argument('-number_transformations', default=2, help='Number of transformations, applied in this order: scale, rotation, translation in x, translation in y, shear in x and shear in y.', type=int)
        parser.add_argument('-min_scale', default=0.9, help='Minimum scale (relative, with 1 being original scale).', type=float)
        parser.add_argument('-max_scale', default=1.1, help='Maximum scale (relative, with 1 being original scale).', type=float)
        parser.add_argument('-min_rotation', default=-math.pi/4, help='Minimum rotation (e.g., -math.pi).', type=float)
        parser.add_argument('-max_rotation', default=math.pi/4, help='Maximum rotation (e.g., math.pi).', type=float)
        parser.add_argument('-min_translation', default=-0.2, help='Minimum translation in both x and y (relative to size).', type=float)
        parser.add_argument('-max_translation', default=0.2, help='Maximum translation in both x and y (relative to size).', type=float)
        parser.add_argument('-min_shear', default=-0.5, help='Minimum shear (relative to size).', type=float)
        parser.add_argument('-max_shear', default=0.5, help='Maximum shear (relative to size).', type=float)
        parser.add_argument('-min_color', default=0.5, help='Minimum color value, maximum is 1.', type=float)
        parser.add_argument('-multiplier', default=1000, help='How many times to multiply each font/letter.', type=int)
        return parser

    def main(self):
        """
        Main method.
        """

        database = utils.read_hdf5(self.args.database_file)
        log('[Data] read %s' % self.args.database_file)

        # one-hot size of code
        N_fonts = database.shape[0]
        N_classes = database.shape[1]
        N = N_fonts*N_classes

        #
        # Fonts and codes are created in the following way (example for 10 classes):
        #
        # font class
        # 0    0
        # 0    1
        # ...
        # 0    9
        # 1    0
        # 1    1
        # ...
        # 1    9
        #
        # This scheme is then repeated according to the multiplier.
        # The advantage of this scheme is that a balanced subset can be selected
        # in multiples of 10.
        #

        codes_fonts = numpy.expand_dims(numpy.repeat(numpy.array(range(N_fonts)), N_classes, axis=0), axis=1)
        codes_classes = numpy.expand_dims(numpy.tile(numpy.array(range(N_classes)), (N_fonts)), axis=1)
        codes = numpy.concatenate((numpy.expand_dims(numpy.arange(N), axis=1), codes_fonts, codes_classes), axis=1)
        codes = numpy.tile(codes, (self.args.multiplier, 1))

        N_theta = self.args.number_transformations
        theta = numpy.zeros((self.args.multiplier * N, N_theta))

        assert N_theta > 0
        if N_theta > 0: # translation x
            theta[:, 0] = numpy.random.uniform(self.args.min_translation, self.args.max_translation, size=(self.args.multiplier * N))
        if N_theta > 1: # translation y
            theta[:, 1] = numpy.random.uniform(self.args.min_translation, self.args.max_translation, size=(self.args.multiplier * N))
        if N_theta > 2: # shear x
            theta[:, 2] = numpy.random.uniform(self.args.min_shear, self.args.max_shear, size=(self.args.multiplier * N))
        if N_theta > 3: # shear y
            theta[:, 3] = numpy.random.uniform(self.args.min_shear, self.args.max_shear, size=(self.args.multiplier * N))
        if N_theta > 4: # scale
            theta[:, 4] = numpy.random.uniform(self.args.min_scale, self.args.max_scale, size=(self.args.multiplier * N))
        if N_theta > 5: # rotation
            theta[:, 5] = numpy.random.uniform(self.args.min_rotation, self.args.max_rotation, size=(self.args.multiplier * N))
        if N_theta > 6:
            theta[:, 6] = numpy.random.uniform(self.args.min_color, 1, size=(self.args.multiplier * N))
        if N_theta > 7:
            theta[:, 7] = numpy.random.uniform(self.args.min_color, 1, size=(self.args.multiplier * N))
        if N_theta > 8:
            theta[:, 8] = numpy.random.uniform(self.args.min_color, 1, size=(self.args.multiplier * N))

        utils.write_hdf5(self.args.codes_file, codes)
        log('[Data] wrote %s' % self.args.codes_file)
        utils.write_hdf5(self.args.theta_file, theta)
        log('[Data] wrote %s' % self.args.theta_file)


if __name__ == '__main__':
    program = GenerateCodes()
    program.main()