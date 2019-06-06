import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common.log import log
from common import utils
from common import plot
from common import paths

import argparse
import numpy


class SummarizeDataset:
    """
    Summarize the dataset.
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

        parser = argparse.ArgumentParser(description='Summarize dataset including plots and statistics.')
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-test_theta_file', default=paths.test_theta_file(), help='HDF5 file containing transformations.', type=str)
        parser.add_argument('-skip', default=10, help='Number of samples to skip for visualization.', type=int)
        parser.add_argument('-pca_images_file', default=paths.image_file('data/images_pca'), help='File for PCA visualization.', type=str)
        parser.add_argument('-umap_images_file', default=paths.image_file('data/images_umap'), help='File for Umap visualization.', type=str)
        parser.add_argument('-lle_images_file', default=None, help='File for PLLE visualization.', type=str)
        parser.add_argument('-mlle_images_file', default=None, help='File for MLLE visualization.', type=str)
        parser.add_argument('-mds_images_file', default=paths.image_file('data/images_mds'), help='File for MDS visualization.', type=str)
        parser.add_argument('-tsne_images_file', default=paths.image_file('data/images_tsne'), help='File for t-SNE visualization.', type=str)
        parser.add_argument('-pca_theta_file', default=paths.image_file('data/theta_pca'), help='File for PCA visualization.', type=str)
        parser.add_argument('-umap_theta_file', default=paths.image_file('data/theta_umap'), help='File for Umap visualization.', type=str)
        parser.add_argument('-lle_theta_file', default=None, help='File for PLLE visualization.', type=str)
        parser.add_argument('-mlle_theta_file', default=None, help='File for MLLE visualization.', type=str)
        parser.add_argument('-mds_theta_file', default=paths.image_file('data/theta_mds'), help='File for MDS visualization.', type=str)
        parser.add_argument('-tsne_theta_file', default=paths.image_file('data/theta_tsne'), help='File for t-SNE visualization.', type=str)
        return parser

    def main(self):
        """
        Main method.
        """

        images = utils.read_hdf5(self.args.test_images_file)
        log('[Data] read %s' % self.args.test_images_file)

        codes = utils.read_hdf5(self.args.test_codes_file)
        log('[Data] read %s' % self.args.test_codes_file)

        data = images.reshape(images.shape[0], images.shape[1]*images.shape[2])
        labels = codes[:, 2]

        skip = self.args.skip
        if self.args.pca_images_file is not None:
            plot.manifold(self.args.pca_images_file, data[::skip, :], labels[::skip], 'pca')
            log('[Data] ran PCA')
        if self.args.umap_images_file is not None:
            plot.manifold(self.args.umap_images_file, data[::skip, :], labels[::skip], 'umap')
            log('[Data] ran UMAP')
        if self.args.lle_images_file is not None:
            plot.manifold(self.args.lle_images_file, data[::skip, :], labels[::skip], 'lle')
            log('[Data] ran LLE')
        if self.args.mlle_images_file is not None:
            plot.manifold(self.args.mlle_images_file, data[::skip, :], labels[::skip], 'mlle')
            log('[Data] ran MLLE')
        if self.args.mds_images_file is not None:
            plot.manifold(self.args.mds_images_file, data[::skip, :], labels[::skip], 'mds')
            log('[Data] ran MDS')
        if self.args.tsne_images_file is not None:
            plot.manifold(self.args.tsne_images_file, data[::skip, :], labels[::skip], 'tsne')
            log('[Data] ran t-SNE')

        theta = utils.read_hdf5(self.args.test_theta_file)
        log('[Data] read %s' % self.args.test_theta_file)

        if self.args.pca_theta_file is not None:
            plot.manifold(self.args.pca_theta_file, theta[::skip, :], labels[::skip], 'pca', None)
            log('[Data] ran PCA')
        if self.args.umap_theta_file is not None:
            plot.manifold(self.args.umap_theta_file, theta[::skip, :], labels[::skip], 'umap', None)
            log('[Data] ran UMAP')
        if self.args.lle_theta_file is not None:
            plot.manifold(self.args.lle_theta_file, theta[::skip, :], labels[::skip], 'lle', None)
            log('[Data] ran LLE')
        if self.args.mlle_theta_file is not None:
            plot.manifold(self.args.mlle_theta_file, theta[::skip, :], labels[::skip], 'mlle', None)
            log('[Data] ran MLLE')
        if self.args.mds_theta_file is not None:
            plot.manifold(self.args.mds_theta_file, theta[::skip, :], labels[::skip], 'mds', None)
            log('[Data] ran MDS')
        if self.args.tsne_theta_file is not None:
            plot.manifold(self.args.tsne_theta_file, theta[::skip, :], labels[::skip], 'tsne', None)
            log('[Data] ran t-SNE')


if __name__ == '__main__':
    program = SummarizeDataset()
    program.main()
