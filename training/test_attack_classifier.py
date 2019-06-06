import os
import sys

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
from common.log import log
from common import paths

import numpy
import argparse
import sklearn.decomposition
import sklearn.neighbors
import terminaltables


if utils.display():
    from common import plot


class TestAttackClassifier:
    """
    Test a trained classifier.
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

        self.test_images = None
        """ (numpy.ndarray) Images to test on. """

        self.perturbation_images = None
        """ (numpy.ndarray) Images corresponding to perturbations. """

        self.train_images = None
        """ (numpy.ndarray) Images to train on. """

        self.test_theta = None
        """ (numpy.ndarray) Transformation parameters. """

        self.train_theta = None
        """ (numpy.ndarra) Transformation parameters, i.e. latent codes. """

        self.perturbations = None
        """ (numpy.ndarray) Perturbations per test image. """

        self.test_codes = None
        """ (numpy.ndarray) Test codes."""

        self.perturbation_codes = None
        """ (numpy.ndarray) Codes corresponding to perturbations. """

        self.accuracy = None
        """ (numpy.ndarray) Success indicator of test samples. """

        self.success = None
        """ (numpy.ndarray) Success indicator for perturbations."""

        self.pca = None
        """ (sklearn.decomposition.IncrementalPCA) PCA to make nearest neighbor more efficient. """

        self.neighbors = None
        """ (sklearn.neighbors.NearestNeighbors) Nearest neighbor model. """

        self.norms = [1, 2, float('inf')]
        """ ([float]) Norms to evaluate. """

        self.results = []
        """ (dict) Dictionary containing all statistics. """

        for n in range(len(self.norms)):
            self.results.append(dict())

        log('-- ' + self.__class__.__name__)
        for key in vars(self.args):
            log('[Testing] %s=%s' % (key, str(getattr(self.args, key))))

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Test attacks on classifier.')
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-train_images_file', default=paths.train_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_theta_file', default=paths.test_theta_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-train_theta_file', default=paths.train_theta_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing labels.', type=str)
        parser.add_argument('-label_index', default=2, help='Column index in label file.', type=int)
        parser.add_argument('-accuracy_file', default=paths.results_file('classifier/accuracy'), help='Correctly classified test samples of classifier.', type=str)
        parser.add_argument('-perturbations_file', default=paths.results_file('classifier/perturbations'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-success_file', default=paths.results_file('classifier/success'), help='HDF5 file indicating attack success.', type=str)
        parser.add_argument('-results_file', default='', help='Path to pickled results file.', type=str)
        parser.add_argument('-plot_directory', default=paths.experiment_dir('classifier'), help='Path to PNG plot file for success rate.', type=str)
        parser.add_argument('-plot_manifolds', default=False, action='store_true', help='Whether to plot manifolds.')
        parser.add_argument('-latent', default=False, action='store_true', help='Latent statistics.')

        return parser

    def load_data(self):
        """
        Load data and model.
        """

        self.test_images = utils.read_hdf5(self.args.test_images_file)
        self.test_images = self.test_images.reshape((self.test_images.shape[0], -1))
        log('[Testing] read %s' % self.args.test_images_file)

        self.train_images = utils.read_hdf5(self.args.train_images_file)
        self.train_images = self.train_images.reshape((self.train_images.shape[0], -1))
        log('[Testing] read %s' % self.args.train_images_file)

        self.train_theta = utils.read_hdf5(self.args.train_theta_file)
        log('[Testing] read %s' % self.args.train_theta_file)

        self.test_theta = utils.read_hdf5(self.args.test_theta_file)
        log('[Testing] read %s' % self.args.test_theta_file)

        self.test_codes = utils.read_hdf5(self.args.test_codes_file)
        self.test_codes = self.test_codes[:, self.args.label_index]
        log('[Testing] read %s' % self.args.test_codes_file)

        self.perturbations = utils.read_hdf5(self.args.perturbations_file)
        if len(self.perturbations.shape) > 3:
            self.perturbations = self.perturbations.reshape((self.perturbations.shape[0], self.perturbations.shape[1], -1))
        self.perturbation_images = self.test_images[:self.perturbations.shape[1]]
        self.perturbation_codes = self.test_codes[:self.perturbations.shape[1]]
        log('[Testing] read %s' % self.args.perturbations_file)
        assert not numpy.any(self.perturbations != self.perturbations), 'NaN in perturbations'

        self.success = utils.read_hdf5(self.args.success_file)
        log('[Testing] read %s' % self.args.success_file)

        self.accuracy = utils.read_hdf5(self.args.accuracy_file)
        self.accuracy = self.accuracy[:self.perturbations.shape[1]]
        log('[Testing] read %s' % self.args.accuracy_file)

    def compute_nearest_neighbors(self, images):
        """
        Compute distances in image and latent space.

        :param images: images to get nearest neighbors for
        :type images: numpy.ndarray
        :param norm: norm to use
        :type norm: float
        """

        fit = 100000
        if self.pca is None:
            self.pca = sklearn.decomposition.IncrementalPCA(n_components=20)
            self.pca.fit(self.train_images[:fit])
            log('[Testing] fitted PCA')
        if self.neighbors is None:
            data = self.pca.transform(self.train_images)
            self.neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=10, algorithm='kd_tree')
            self.neighbors.fit(data[:fit])
            log('[Testing] fitted nearest neighbor')

        data = self.pca.transform(images)
        _, indices = self.neighbors.kneighbors(data)
        return indices

    def compute_statistics(self):
        """
        Compute statistics based on distances.
        """

        N_class = numpy.max(self.test_codes) + 1
        num_attempts = self.perturbations.shape[0]

        perturbations = numpy.swapaxes(self.perturbations, 0, 1)
        perturbations = perturbations.reshape((perturbations.shape[0]*perturbations.shape[1], perturbations.shape[2]))
        success = numpy.swapaxes(self.success, 0, 1)
        success = success.reshape((success.shape[0]*success.shape[1]))

        accuracy = numpy.repeat(self.accuracy, num_attempts, axis=0)
        # Raw success is the base for all statistics, as we need to consider only these
        # attacks that are successful and where the classifier originally was correct.
        raw_overall_success = numpy.logical_and(success >= 0, accuracy)
        log('[Testing] %d valid attacks' % numpy.sum(raw_overall_success))

        # For off-manifold attacks this should not happen, but save is save.
        if not numpy.any(raw_overall_success):
            for n in range(len(self.norms)):
                for type in ['raw_success', 'raw_iteration', 'raw_average', 'raw_latent']:
                    self.results[n][type] = 0
                for type in ['raw_class_success', 'raw_class_average', 'raw_class_latent']:
                    self.results[n][type] = numpy.zeros((N_class))
            if self.args.results_file:
                utils.write_pickle(self.args.results_file, self.results)
                log('[Testing] wrote %s' % self.args.results_file)
            log('[Testing] no successful attacks found, no plots')
            return

        perturbation_images = numpy.repeat(self.perturbation_images, num_attempts, axis=0)
        perturbation_codes = numpy.repeat(self.perturbation_codes, num_attempts, axis=0)

        #
        # We compute some simple statistics:
        # - raw success rate: fraction of successful attack without considering epsilon
        # - corrected success rate: fraction of successful attacks within epsilon-ball
        # - raw average perturbation: average distance to original samples (for successful attacks)
        # - corrected average perturbation: average distance to original samples for perturbations
        #   within epsilon-ball (for successful attacks).
        # These statistics can also be computed per class.
        # And these statistics are computed with respect to three norms.

        if self.args.plot_directory and utils.display():
            iterations = success[raw_overall_success]
            x = numpy.arange(numpy.max(iterations) + 1)
            y = numpy.bincount(iterations)
            plot_file = os.path.join(self.args.plot_directory, 'iterations')
            plot.bar(plot_file, x, y, title='Distribution of Iterations of Successful Attacks', xlabel='Number of Iterations', ylabel='Count')
            log('[Testing] wrote %s' % plot_file)

        for n in range(len(self.norms)):
            norm = self.norms[n]
            delta = numpy.linalg.norm(perturbation_images - perturbations, norm, axis=1)

            if self.args.plot_directory and utils.display():
                plot_file = os.path.join(self.args.plot_directory, 'distances_l%g' % norm)
                plot.histogram(plot_file, delta[raw_overall_success], 50, title='Distribution of $L_{%g}$ Distances of Successful Attacks' % norm,
                              xlabel='Distance', ylabel='Count')
                log('[Testing] wrote %s' % plot_file)

            #debug_accuracy = numpy.sum(accuracy) / accuracy.shape[0]
            #debug_attack_fraction = numpy.sum(raw_overall_success) / numpy.sum(success >= 0)
            #debug_test_fraction = numpy.sum(raw_overall_success) / numpy.sum(accuracy)
            #log('[Testing] attacked model accuracy: %g' % debug_accuracy)
            #log('[Testing] only %g of successful attacks are valid' % debug_attack_fraction)
            #log('[Testing] only %g of correct samples are successfully attacked' % debug_test_fraction)

            N_accuracy = numpy.sum(accuracy)
            self.results[n]['raw_success'] = numpy.sum(raw_overall_success) / N_accuracy
            self.results[n]['raw_iteration'] = numpy.average(success[raw_overall_success])
            self.results[n]['raw_average'] = numpy.average(delta[raw_overall_success]) if numpy.any(raw_overall_success) else 0
            self.results[n]['raw_latent'] = 0

            raw_class_success = numpy.zeros((N_class, perturbation_images.shape[0]), bool)
            self.results[n]['raw_class_success'] = numpy.zeros((N_class))
            self.results[n]['raw_class_average'] = numpy.zeros((N_class))
            self.results[n]['raw_class_latent'] = numpy.zeros((N_class))

            for c in range(N_class):
                N_samples = numpy.sum(numpy.logical_and(accuracy, perturbation_codes == c))
                if N_samples <= 0:
                    continue;

                raw_class_success[c] = numpy.logical_and(raw_overall_success, perturbation_codes == c)
                self.results[n]['raw_class_success'][c] = numpy.sum(raw_class_success[c]) / N_samples
                if numpy.any(raw_class_success[c]):
                    self.results[n]['raw_class_average'][c] = numpy.average(delta[raw_class_success[c].astype(bool)])

        if self.args.results_file:
            utils.write_pickle(self.args.results_file, self.results)
            log('[Testing] wrote %s' % self.args.results_file)

    def compute_latent_statistics(self):
        """
        Compute latent statistics.
        """

        N_class = numpy.max(self.test_codes) + 1
        num_attempts = self.perturbations.shape[0]

        perturbations = numpy.swapaxes(self.perturbations, 0, 1)
        perturbations = perturbations.reshape((perturbations.shape[0] * perturbations.shape[1], perturbations.shape[2]))
        success = numpy.swapaxes(self.success, 0, 1)
        success = success.reshape((success.shape[0] * success.shape[1]))

        accuracy = numpy.repeat(self.accuracy, num_attempts, axis=0)
        # Raw success is the base for all statistics, as we need to consider only these
        # attacks that are successful and where the classifier originally was correct.
        raw_overall_success = numpy.logical_and(success >= 0, accuracy)

        # For off-manifold attacks this should not happen, but save is save.
        if not numpy.any(raw_overall_success):
            for n in range(len(self.norms)):
                for type in ['raw_success', 'raw_iteration', 'raw_average', 'raw_latent']:
                    self.results[n][type] = 0
                for type in ['raw_class_success', 'raw_class_average', 'raw_class_latent']:
                    self.results[n][type] = numpy.zeros((N_class))
            if self.args.results_file:
                utils.write_pickle(self.args.results_file, self.results)
                log('[Testing] wrote %s' % self.args.results_file)
            log('[Testing] no successful attacks found, no plots')
            return

        perturbation_images = numpy.repeat(self.perturbation_images, num_attempts, axis=0)
        perturbation_codes = numpy.repeat(self.perturbation_codes, num_attempts, axis=0)

        #
        # Compute nearest neighbors for perturbations and test images,
        # to backproject them into the latent space.
        # Also compute the dot product betweenm perturbations and a local
        # plane approximation base don the three nearest neighbors.
        #

        log('[Testing] computing nearest neighbor ...')
        nearest_neighbors_indices = self.compute_nearest_neighbors(perturbation_images)
        nearest_neighbors = self.train_theta[nearest_neighbors_indices[:, 0]]
        perturbation_nearest_neighbor_indices = self.compute_nearest_neighbors(perturbations)
        perturbation_nearest_neighbor = self.train_theta[perturbation_nearest_neighbor_indices[:, 0]]

        # Compute statistics over the perturbation with respect to the plane
        # defined by the three nearest neighbors of the corresponding test sample.
        if self.args.plot_directory and self.args.plot_manifolds and utils.display():
            pure_perturbations = perturbations[raw_overall_success] - perturbation_images[raw_overall_success]
            pure_perturbations_norm = numpy.linalg.norm(pure_perturbations, ord=2, axis=1)
            for k in range(10):
                direction = perturbation_images[raw_overall_success] - self.train_images[nearest_neighbors_indices[:, k][raw_overall_success]]
                direction_norm = numpy.linalg.norm(direction, ord=2, axis=1)
                dot_products = numpy.einsum('ij,ij->i', direction, pure_perturbations)
                dot_product_norms = numpy.multiply(pure_perturbations_norm, direction_norm)
                dot_product_norms[dot_product_norms == 0] = 1
                dot_products /= dot_product_norms
                dot_products = numpy.degrees(numpy.arccos(dot_products))

                # matplotlib's hsitogram plots give weird error if there are NaN values, so simple check:
                if dot_products.shape[0] > 0 and not numpy.any(dot_products != dot_products):
                    plot_file = os.path.join(self.args.plot_directory, 'dot_products_nn%d' % k)
                    plot.histogram(plot_file, dot_products, 100,
                                   title='Dot Products Between Adversarial Perturbations and Direction to Nearest Neighbor %d' % k,
                                   xlabel='Dot Product (Between Normalized Vectors)', ylabel='Count')
                    log('[Testing] wrote %s' % plot_file)

        #
        # We compute some simple statistics:
        # - raw success rate: fraction of successful attack without considering epsilon
        # - corrected success rate: fraction of successful attacks within epsilon-ball
        # - raw average perturbation: average distance to original samples (for successful attacks)
        # - corrected average perturbation: average distance to original samples for perturbations
        #   within epsilon-ball (for successful attacks).
        # These statistics can also be computed per class.
        # And these statistics are computed with respect to three norms.

        if self.args.plot_directory and utils.display():
            iterations = success[raw_overall_success]
            x = numpy.arange(numpy.max(iterations) + 1)
            y = numpy.bincount(iterations)
            plot_file = os.path.join(self.args.plot_directory, 'iterations')
            plot.bar(plot_file, x, y,
                     title='Distribution of Iterations of Successful Attacks', xlabel='Number of Iterations', ylabel='Count')
            log('[Testing] wrote %s' % plot_file)

        for n in range(len(self.norms)):
            norm = self.norms[n]
            delta = numpy.linalg.norm(perturbation_images - perturbations, norm, axis=1)
            latent_delta = numpy.linalg.norm(nearest_neighbors - perturbation_nearest_neighbor, norm, axis=1)

            if self.args.plot_directory and utils.display():
                plot_file = os.path.join(self.args.plot_directory, 'distances_l%g' % norm)
                plot.histogram(plot_file, delta[raw_overall_success], 50, title='Distribution of $L_{%g}$ Distances of Successful Attacks' % norm,
                               xlabel='Distance', ylabel='Count')
                log('[Testing] wrote %s' % plot_file)

            #debug_accuracy = numpy.sum(accuracy) / accuracy.shape[0]
            #debug_attack_fraction = numpy.sum(raw_overall_success) / numpy.sum(success >= 0)
            #debug_test_fraction = numpy.sum(raw_overall_success) / numpy.sum(accuracy)
            #log('[Testing] attacked model accuracy: %g' % debug_accuracy)
            #log('[Testing] only %g of successful attacks are valid' % debug_attack_fraction)
            #log('[Testing] only %g of correct samples are successfully attacked' % debug_test_fraction)

            N_accuracy = numpy.sum(accuracy)
            self.results[n]['raw_success'] = numpy.sum(raw_overall_success) / N_accuracy
            self.results[n]['raw_iteration'] = numpy.average(success[raw_overall_success])
            self.results[n]['raw_average'] = numpy.average(delta[raw_overall_success]) if numpy.any(raw_overall_success) else 0
            self.results[n]['raw_latent'] = numpy.average(latent_delta[raw_overall_success]) if numpy.any(raw_overall_success) else 0

            raw_class_success = numpy.zeros((N_class, perturbation_images.shape[0]), bool)
            self.results[n]['raw_class_success'] = numpy.zeros((N_class))
            self.results[n]['raw_class_average'] = numpy.zeros((N_class))
            self.results[n]['raw_class_latent'] = numpy.zeros((N_class))

            for c in range(N_class):
                N_samples = numpy.sum(numpy.logical_and(accuracy, perturbation_codes == c))
                if N_samples <= 0:
                    continue;

                raw_class_success[c] = numpy.logical_and(raw_overall_success, perturbation_codes == c)
                self.results[n]['raw_class_success'][c] = numpy.sum(raw_class_success[c]) / N_samples
                if numpy.any(raw_class_success[c]):
                    self.results[n]['raw_class_average'][c] = numpy.average(delta[raw_class_success[c].astype(bool)])
                if numpy.any(raw_class_success[c]):
                    self.results[n]['raw_class_latent'][c] = numpy.average(latent_delta[raw_class_success[c].astype(bool)])

        if self.args.results_file:
            utils.write_pickle(self.args.results_file, self.results)
            log('[Testing] wrote %s' % self.args.results_file)

    def plot_manifolds(self):
        """
        Plot manifolds.
        """

        fit = self.test_images.shape[0] // 100
        N_class = numpy.max(self.test_codes) + 1
        num_attempts = self.perturbations.shape[0]
        success = numpy.swapaxes(self.success, 0, 1)
        success = success.reshape((success.shape[0] * success.shape[1]))
        perturbations = self.perturbations.reshape((self.perturbations.shape[0] * self.perturbations.shape[1], -1))
        perturbations = perturbations[success >= 0]
        codes = numpy.repeat(self.perturbation_codes, num_attempts, axis=0)
        codes = codes[success >= 0]

        #
        # Plot all classes and adversarial examples in image space for individual classes as well as all classes.
        #

        manifold_visualization = plot.ManifoldVisualization('tsne', pre_pca=40)
        manifold_visualization.fit(self.test_images[:fit])
        log('[Testing] computed t-SNE on test images')

        for n in range(N_class):
            labels = ['Class %d' % (nn + 1) for nn in range(N_class)] + ['Adversarial Examples Class %d' % (n + 1)]
            data = numpy.concatenate((
                self.test_images[:fit],
                perturbations[codes == n]
            ))
            classes = numpy.concatenate((
                self.test_codes[:fit],
                numpy.ones((perturbations[codes == n].shape[0])) * N_class,
            ))
            plot_file = os.path.join(self.args.plot_directory, 'perturbations_%d' % (n + 1))
            manifold_visualization.visualize(plot_file, data, classes, labels,
                                             title='Adversarial Examples Class %d\n(The adversarial examples are projected into the embedding using learned SVRs)' % n)
            log('[Testing] wrote %s' % plot_file)

        labels = ['Class %d' % (n + 1) for n in range(N_class)] + ['Adversarial Examples Class %d' % (n + 1) for n in
                                                                   range(N_class)]
        data = numpy.concatenate((
            self.test_images[:fit],
            perturbations
        ))
        classes = numpy.concatenate((
            self.test_codes[:fit],
            codes + N_class,
        ))
        plot_file = os.path.join(self.args.plot_directory, 'perturbations')
        manifold_visualization.visualize(plot_file, data, classes, labels,
                                         title='Adversarial Examples\n(The adversarial examples are projected into the embedding using learned SVRs)')
        log('[Testing] wrote %s' % plot_file)

        #
        # Plot all classes and adversarial examples in latent space.
        #

        perturbation_neighbors_indices = self.compute_nearest_neighbors(perturbations)
        perturbation_neighbors = self.train_theta[perturbation_neighbors_indices[:, 0]]
        perturbation_neighbors = numpy.squeeze(perturbation_neighbors)
        perturbation_codes = numpy.repeat(self.perturbation_codes, num_attempts, axis=0)
        perturbation_neighbors = numpy.concatenate((perturbation_neighbors[success >= 0], perturbation_codes[success >= 0]), axis=1)
        plot_theta = numpy.concatenate((self.test_theta, self.test_codes), axis=1)
        manifold_visualization = plot.ManifoldVisualization('tsne', pre_pca=None)
        manifold_visualization.fit(plot_theta[:fit])
        log('[Testing] computed t-SNE on test codes')

        for c in range(N_class):
            labels = ['Class %d' % (cc + 1) for cc in range(N_class)] + ['Adversarial Examples Class %d' % (c + 1)]
            data = numpy.concatenate((
                plot_theta[:fit],
                perturbation_neighbors[codes == c]
            ))
            classes = numpy.concatenate((
                self.test_codes[:fit],
                numpy.ones((perturbation_neighbors[codes == c].shape[0])) * N_class,
            ))
            plot_file = os.path.join(self.args.plot_directory, 'latent_perturbations_%d' % (c + 1))
            manifold_visualization.visualize(plot_file, data, classes, labels,
                                             title='Adversarial Examples Class %d\n(The adversarial examples are projected into the embedding using learned SVRs)' % c)
            log('[Testing] wrote %s' % plot_file)

        labels = ['Class %d' % (c + 1) for c in range(N_class)] + ['Adversarial Examples Class %d' % (c + 1) for c in range(N_class)]
        data = numpy.concatenate((
            plot_theta[:fit],
            perturbation_neighbors
        ))
        classes = numpy.concatenate((
            self.test_codes[:fit],
            codes + N_class,
        ))
        plot_file = os.path.join(self.args.plot_directory, 'latent_perturbations')
        manifold_visualization.visualize(plot_file, data, classes, labels,
                                         title='Adversarial Examples\n(The adversarial examples are projected into the embedding using learned SVRs)')
        log('[Testing] wrote %s' % plot_file)

    def print_statistics(self):
        """
        Print statistics.
        """

        N_class = numpy.max(self.test_codes) + 1
        table_data = []

        for n in range(len(self.norms)):
            table_headings = ['(Class) Norm', 'Metric, Epsilon', 'Raw']
            table_data.append(table_headings)
            norm = self.norms[n]

            table_row = ['    L_%.3g' % norm, 'Success Rate', '%.3g' % self.results[n]['raw_success']]
            table_data.append(table_row)

            table_row = ['    L_%.3g' % norm, 'Image Distance', '%.3g' % self.results[n]['raw_average']]
            table_data.append(table_row)

            table_row = ['    L_%.3g' % norm, 'Latent Distance', '%.3g' % self.results[n]['raw_latent']]
            table_data.append(table_row)

            for c in range(N_class):
                table_row = ['(%d) L_%.3g' % (c, norm), 'Success Rate', '%.3g' % self.results[n]['raw_class_success'][c]]
                table_data.append(table_row)

                table_row = ['(%d) L_%.3g' % (c, norm), 'Image Distance', '%.3g' % self.results[n]['raw_class_average'][c]]
                table_data.append(table_row)

                table_row = ['(%d) L_%.3g' % (c, norm), 'Latent Distance', '%.3g' % self.results[n]['raw_class_latent'][c]]
                table_data.append(table_row)
            table_data.append(['---']*2)

        table = terminaltables.AsciiTable(table_data)
        log(table.table)

    def main(self):
        """
        Main.
        """

        self.load_data()
        if self.args.latent:
            self.compute_latent_statistics()
        else:
            self.compute_statistics()
        self.print_statistics()
        if self.args.plot_directory and self.args.plot_manifolds and utils.display():
            self.plot_manifolds()


if __name__ == '__main__':
    program = TestAttackClassifier()
    program.main()
