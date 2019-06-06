import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common.log import log
from common import utils
from common import plot
from common import paths
from common import vis
import argparse
import math


class VisualizeIndividual:
    """
    Visualize images in mosaic.

    :param args: arguments
    :type args: list
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
            log('[Visualization] %s=%s' % (key, str(getattr(self.args, key))))

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Visualize images as mosaic.')
        parser.add_argument('-images_file', default=paths.results_file('reconstructions'), help='HDF5 file containing images to visualize', type=str)
        parser.add_argument('-codes_file', default='', help='HDF5 label file.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-label', default=-1, help='Label.', type=int)
        parser.add_argument('-vmin', default=None, help='Minimum value vor visualization.', type=float)
        parser.add_argument('-vmax', default=None, help='Maximum value for visualization.', type=float)
        parser.add_argument('-max_images', default=55, type=int, help='Number of images to output.')
        parser.add_argument('-output_directory', default=paths.experiment_file('', ''), help='Output directory.', type=str)
        return parser

    def main(self):
        """
        Main.
        """

        if not os.path.exists(self.args.images_file):
            log('[Visualization] file %s not found' % self.args.images_file)
            exit(1)

        if not os.path.exists(self.args.output_directory) and self.args.output_directory:
            log('[Visualization] creating %s' % self.args.output_directory)

        images = utils.read_hdf5(self.args.images_file)
        if len(images.shape) > 3 and images.shape[3] != 3:
            images = images.reshape(-1, images.shape[2], images.shape[3])
        log('[Visualization] read %s' % self.args.images_file)

        if self.args.codes_file:
            codes = utils.read_hdf5(self.args.codes_file)
            log('[Visualization] read %s' % self.args.codes_file)
            codes = codes[:, self.args.label_index]

            if self.args.label >= 0:
                images = images[codes == self.args.label]

        for n in range(min(images.shape[0], self.args.max_images)):
            png_file = os.path.join(self.args.output_directory, '%d.png' % n)
            vis.image(png_file, images[n], 5, 'gray', self.args.vmin, self.args.vmax)
            log('[Visualization] wrote %s' % png_file)


if __name__ == '__main__':
    program = VisualizeIndividual()
    program.main()