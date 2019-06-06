import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common import utils
import models
from common.log import log
from common import paths
import common.numpy

import argparse
import numpy
import math
import torch


class GenerateDataset:
    """
    Runs the decoder to generate a dataset of transformed letters.
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

        parser = argparse.ArgumentParser(description='Create HDF5 file of rendered font letters and digits.')
        parser.add_argument('-database_file', default=paths.database_file(), help='HDF5 file containing prototype images.', type=str)
        parser.add_argument('-codes_file', default=paths.codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-theta_file', default=paths.theta_file(), help='HDF5 file containing transformations.', type=str)
        parser.add_argument('-batch_size', default=32, help='Batch size.', type=int)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')
        parser.add_argument('-images_file', default=paths.images_file(), help='HDF5 file containing transformed images.', type=str)
        parser.set_defaults(use_gpu=True)
        return parser

    def main(self):
        """
        Main method.
        """

        database = utils.read_hdf5(self.args.database_file)
        log('[Data] read %s' % self.args.database_file)

        N_font = database.shape[0]
        N_class = database.shape[1]

        assert database.shape[2] == database.shape[3]
        database = database.reshape((database.shape[0]*database.shape[1], database.shape[2], database.shape[3]))
        database = torch.from_numpy(database).float()
        if self.args.use_gpu:
            database = database.cuda()

        database = torch.autograd.Variable(database)

        codes = utils.read_hdf5(self.args.codes_file)
        codes = codes[:, 0]
        codes = common.numpy.one_hot(codes, N_font*N_class)
        log('[Data] read %s' % self.args.codes_file)

        theta = utils.read_hdf5(self.args.theta_file)
        N = theta.shape[0]
        N_theta = theta.shape[1]
        log('[Data] read %s' % self.args.theta_file)

        model = models.OneHotDecoder(database, N_theta)
        images = []

        num_batches = int(math.ceil(float(N)/self.args.batch_size))
        for b in range(num_batches):
            batch_theta = torch.from_numpy(theta[b*self.args.batch_size: min((b + 1)*self.args.batch_size, N)])
            batch_codes = torch.from_numpy(codes[b*self.args.batch_size: min((b + 1)*self.args.batch_size, N)])
            batch_codes, batch_theta = batch_codes.float(), batch_theta.float()

            if self.args.use_gpu:
                batch_codes, batch_theta = batch_codes.cuda(), batch_theta.cuda()

            batch_codes, batch_theta = torch.autograd.Variable(batch_codes), torch.autograd.Variable(batch_theta)
            output = model(batch_codes, batch_theta)

            images.append(output.data.cpu().numpy().squeeze())
            if b%1000 == 0:
                log('[Data] processed %d/%d batches' % (b + 1, num_batches))

        images = numpy.concatenate(images, axis=0)
        if len(images.shape) > 3:
            images = numpy.transpose(images, (0, 2, 3, 1))
        utils.write_hdf5(self.args.images_file, images)
        log('[Data] wrote %s' % self.args.images_file)


if __name__ == '__main__':
    program = GenerateDataset()
    program.main()