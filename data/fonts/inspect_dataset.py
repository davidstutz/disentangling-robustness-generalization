import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common.log import log
from common import utils
from common import vis
from common import paths

import argparse
from matplotlib import pyplot
import numpy


class InspectDataset:
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
        parser.add_argument('-images_file', default=paths.test_images_file(), help='HDF5 file created.', type=str)
        parser.add_argument('-output_directory', default='./', help='Output directory.', type=str)
        parser.add_argument('-max_images', default=50, help='Numbe rof visualizations.', type=str)#

        return parser

    def main(self):
        """
        Main method.
        """

        images = utils.read_hdf5(self.args.images_file)
        log('[Data] read %s' % self.args.images_file)

        rows = 10
        cols = 10
        print(images.shape)
        for n in range(min(self.args.max_images, images.shape[0]//(rows*cols))):
            log('[Data] %d/%d' % ((n + 1)*rows*cols, images.shape[0]))
            plot_file = os.path.join(self.args.output_directory, str(n) + '.png')
            vis.mosaic(plot_file, images[n*rows*cols:(n+1)*rows*cols], cols=10)
            log('[Data] wrote %s' % plot_file)


if __name__ == '__main__':
    program = InspectDataset()
    program.main()
