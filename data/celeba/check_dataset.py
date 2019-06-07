import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common.log import log, LogLevel
from common import utils
from common import paths
import argparse
from matplotlib import pyplot
import numpy


class CheckDataset:
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
        train_labels_file = paths.celeba_train_labels_file()
        test_labels_file = paths.celeba_test_labels_file()

        assert os.path.exists(train_images_file)
        assert os.path.exists(test_images_file)
        assert os.path.exists(train_labels_file)
        assert os.path.exists(test_labels_file)

        train_images = utils.read_hdf5(train_images_file)
        test_images = utils.read_hdf5(test_images_file)
        train_labels = utils.read_hdf5(train_labels_file)
        test_labels = utils.read_hdf5(test_labels_file)

        print('train_images: %s' % 'x'.join([str(dim) for dim in train_images.shape]))
        print('test_images: %s' % 'x'.join([str(dim) for dim in test_images.shape]))
        print('train_labels: %s' % 'x'.join([str(dim) for dim in train_labels.shape]))
        print('test_labels: %s' % 'x'.join([str(dim) for dim in test_labels.shape]))

        attributes = [
            '5_o_Clock_Shadow',
            'Arched_Eyebrows',
            'Attractive',
            'Bags_Under_Eyes',
            'Bald',
            'Bangs',
            'Big_Lips',
            'Big_Nose',
            'Black_Hair',
            'Blond_Hair',
            'Blurry',
            'Brown_Hair',
            'Bushy_Eyebrows',
            'Chubby',
            'Double_Chin',
            'Eyeglasses',
            'Goatee',
            'Gray_Hair',
            'Heavy_Makeup',
            'High_Cheekbones',
            'Male',
            'Mouth_Slightly_Open',
            'Mustache',
            'Narrow_Eyes',
            'No_Beard',
            'Oval_Face',
            'Pale_Skin',
            'Pointy_Nose',
            'Receding_Hairline',
            'Rosy_Cheeks',
            'Sideburns',
            'Smiling',
            'Straight_Hair',
            'Wavy_Hair',
            'Wearing_Earrings',
            'Wearing_Hat',
            'Wearing_Lipstick',
            'Wearing_Necklace',
            'Wearing_Necktie',
            'Young',
        ]

        for i in range(min(10, train_labels.shape[0])):
            log('%i: ', LogLevel.INFO, '')
            for j in range(len(attributes)):
                if train_labels[i, j] > 0:
                    log('%s ' % attributes[j])
            pyplot.imshow(train_images[i])
            pyplot.show()


if __name__ == '__main__':
    program = CheckDataset()
    program.main()