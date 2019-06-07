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


class ConvertImages:
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

        filepaths = utils.read_ordered_directory(paths.raw_celeba_images_dir())
        log('reading %s' % paths.raw_celeba_images_dir())

        images = []
        for filepath in filepaths:
            log('processing %s' % os.path.basename(filepath))
            image = imageio.imread(filepath)
            width = 54
            height = int(width*image.shape[0]/float(image.shape[1]))
            image = skimage.transform.resize(image, (height, width))
            image = image[5:image.shape[0] - 5, 3:image.shape[1]-3, :]
            # Note that images are already scaled to [0, 1] here!
            #image = image/255.
            #print(numpy.min(image), numpy.max(image))
            assert numpy.min(image) >= 0 and numpy.max(image) <= 1
            images.append(image)

            #print(image.shape)
            #pyplot.imshow(image)
            #pyplot.show()

        images = numpy.array(images)
        log('%g %g' % (numpy.min(images), numpy.max(images)))
        N = images.shape[0]
        N_train = int(0.9 * N)

        train_images = images[:N_train]
        test_images = images[N_train:]

        utils.write_hdf5(paths.celeba_train_images_file(), train_images.astype(numpy.float32))
        log('wrote %s' % paths.celeba_train_images_file())
        utils.write_hdf5(paths.celeba_test_images_file(), test_images.astype(numpy.float32))
        log('wrote %s' % paths.celeba_test_images_file())


if __name__ == '__main__':
    program = ConvertImages()
    program.main()