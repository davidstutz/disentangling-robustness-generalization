import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common.log import log
from common import utils
from common import paths
import argparse
import numpy


class ConvertLabels:
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

        with open(paths.raw_celeba_labels_file(), 'r') as f:
            lines = f.read().split('\n')
            lines = [line for line in lines if line]
            lines = lines[1:]

            attributes = [str(attribute) for attribute in lines[0].split(' ') if attribute]
            lines = lines[1:]

            labels = []
            for line in lines:
                values = [int(value) for value in line.split(' ')[1:] if value]
                assert len(values) == len(attributes)
                labels.append(values)

            labels = numpy.array(labels)
            labels[labels == -1] = 0

            def statistics(labels):
                """
                Label statistics.
                """

                for i in range(len(attributes)):
                    positive = numpy.sum(labels[:, i] == 1)
                    negative = numpy.sum(labels[:, i] == 0)
                    log('%d. attribute %s: %d %d' % (i, attributes[i], positive, negative))

            N = labels.shape[0]
            N_train = int(0.9*N)

            train_labels = labels[:N_train]
            test_labels = labels[N_train:]

            statistics(labels)
            statistics(train_labels)
            statistics(test_labels)

            utils.write_hdf5(paths.celeba_train_labels_file(), train_labels.reshape(-1, 1).astype(numpy.int))
            log('wrote %s' % paths.celeba_train_labels_file())
            utils.write_hdf5(paths.celeba_test_labels_file(), test_labels.reshape(-1, 1).astype(numpy.int))
            log('wrote %s' % paths.celeba_test_labels_file())


if __name__ == '__main__':
    program = ConvertLabels()
    program.main()