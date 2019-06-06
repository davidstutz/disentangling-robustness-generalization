import os
import sys

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
from common.log import log
from common import paths
from common import cuda
import common.numpy
import common.torch
import models
from common.state import State
import math
import numpy
import torch
import argparse
import sklearn.decomposition
import sklearn.neighbors
import terminaltables
import sklearn.metrics
if utils.display():
    from common import plot


class TestAttackClassifierDistal:
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
        parser.add_argument('-classifier_file', default=paths.state_file('classifier'), help='Snapshot state file of classifier.', type=str)
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-label_index', default=2, help='Column index in label file.', type=int)
        parser.add_argument('-perturbations_file', default=paths.results_file('classifier/perturbations'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-success_file', default=paths.results_file('classifier/success'), help='HDF5 file indicating attack success.', type=str)
        parser.add_argument('-probabilities_file', default=paths.results_file('classifier/probabilities'), help='HDF5 file containing attack probabilities.')
        parser.add_argument('-results_file', default='', help='Path to pickled results file.', type=str)
        parser.add_argument('-plot_directory', default=paths.experiment_dir('classifier'), help='Path to PNG plot file for success rate.', type=str)
        parser.add_argument('-plot_manifolds', default=False, action='store_true', help='Whether to plot manifolds.')
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')
        parser.add_argument('-batch_size', default=128, help='Batch size of attack.', type=int)

        # Some network parameters.
        parser.add_argument('-network_architecture', default='standard', help='Classifier architecture to use.', type=str)
        parser.add_argument('-network_activation', default='relu', help='Activation function to use.', type=str)
        parser.add_argument('-network_no_batch_normalization', default=False, help='Do not use batch normalization.', action='store_true')
        parser.add_argument('-network_channels', default=16, help='Channels of first convolutional layer, afterwards channels are doubled.', type=int)
        parser.add_argument('-network_dropout', default=False, action='store_true', help='Whether to use dropout.')
        parser.add_argument('-network_units', default='1024,1024,1024,1024', help='Units for MLP.')

        return parser

    def load_data(self):
        """
        Load data and model.
        """

        self.test_images = utils.read_hdf5(self.args.test_images_file).astype(numpy.float32)
        log('[Testing] read %s' % self.args.test_images_file)

        if len(self.test_images.shape) < 4:
            self.test_images = numpy.expand_dims(self.test_images, axis=3)

        self.test_codes = utils.read_hdf5(self.args.test_codes_file)
        self.test_codes = self.test_codes[:, self.args.label_index]
        log('[Testing] read %s' % self.args.test_codes_file)

        self.perturbations = utils.read_hdf5(self.args.perturbations_file)
        if len(self.perturbations.shape) > 3:
            self.perturbations = self.perturbations.reshape((self.perturbations.shape[0], self.perturbations.shape[1], -1))
        self.perturbation_images = self.test_images[:self.perturbations.shape[1]].reshape(self.perturbations.shape[1], -1)
        self.perturbation_codes = self.test_codes[:self.perturbations.shape[1]]
        log('[Testing] read %s' % self.args.perturbations_file)
        assert not numpy.any(self.perturbations != self.perturbations), 'NaN in perturbations'

        self.success = utils.read_hdf5(self.args.success_file)
        log('[Testing] read %s' % self.args.success_file)

        self.probabilities = utils.read_hdf5(self.args.probabilities_file)
        log('[Testing] read %s' % self.args.probabilities_file)

    def load_models(self):
        """
        Load models.
        """

        self.N_class = numpy.max(self.test_codes) + 1
        network_units = list(map(int, self.args.network_units.split(',')))
        log('[Testing] using %d input channels' % self.test_images.shape[3])
        self.model = models.Classifier(self.N_class, resolution=(self.test_images.shape[3], self.test_images.shape[1], self.test_images.shape[2]),
                                       architecture=self.args.network_architecture,
                                       activation=self.args.network_activation,
                                       batch_normalization=not self.args.network_no_batch_normalization,
                                       start_channels=self.args.network_channels,
                                       dropout=self.args.network_dropout,
                                       units=network_units)
        assert os.path.exists(self.args.classifier_file), 'state file %s not found' % self.args.classifier_file
        state = State.load(self.args.classifier_file)
        log('[Testing] read %s' % self.args.classifier_file)

        self.model.load_state_dict(state.model)
        if self.args.use_gpu and not cuda.is_cuda(self.model):
            log('[Testing] classifier is not CUDA')
            self.model = self.model.cuda()
        log('[Testing] loaded classifier')

        # !
        self.model.eval()
        log('[Testing] set classifier to eval')

    def test(self):
        """
        Test classifier to identify valid samples to attack.
        """

        num_batches = int(math.ceil(self.test_images.shape[0] / self.args.batch_size))
        self.test_probabilities = None

        for b in range(num_batches):
            b_start = b * self.args.batch_size
            b_end = min((b + 1) * self.args.batch_size, self.test_images.shape[0])

            batch_images = common.torch.as_variable(self.test_images[b_start: b_end], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            output_classes = self.model(batch_images)
            output_probabilities = torch.nn.functional.softmax(output_classes, 1)

            self.test_probabilities = common.numpy.concatenate(self.test_probabilities, output_probabilities.data.cpu().numpy())

            if b % 100 == 0:
                log('[Testing] computing test probabilities %d' % b)

    def compute_statistics(self):
        """
        Compute statistics based on distances.
        """

        num_attempts = self.perturbations.shape[0]

        perturbations = numpy.swapaxes(self.perturbations, 0, 1)
        perturbations = perturbations.reshape((perturbations.shape[0]*perturbations.shape[1], perturbations.shape[2]))
        success = numpy.swapaxes(self.success, 0, 1)
        success = success.reshape((success.shape[0]*success.shape[1]))

        probabilities = numpy.swapaxes(self.probabilities, 0, 1)
        probabilities = probabilities.reshape((probabilities.shape[0] * probabilities.shape[1], -1))
        confidences = numpy.max(probabilities, 1)

        perturbation_probabilities = self.test_probabilities[:self.success.shape[1]]
        perturbation_probabilities = numpy.repeat(perturbation_probabilities, num_attempts, axis=0)
        perturbation_confidences = numpy.max(perturbation_probabilities, 1)

        probability_ratios = confidences/perturbation_confidences

        raw_overall_success = success >= 0
        log('[Testing] %d valid attacks' % numpy.sum(raw_overall_success))

        # For off-manifold attacks this should not happen, but save is save.
        if not numpy.any(raw_overall_success):
            for type in ['raw_success', 'raw_iteration', 'raw_roc', 'raw_confidence_weighted_success', 'raw_confidence', 'raw_ratios']:
                self.results[type] = 0
            if self.args.results_file:
                utils.write_pickle(self.args.results_file, self.results)
                log('[Testing] wrote %s' % self.args.results_file)
            log('[Testing] no successful attacks found, no plots')
            return

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

            plot_file = os.path.join(self.args.plot_directory, 'probabilities')
            plot.histogram(plot_file, confidences[raw_overall_success], 50)
            log('[Testing] wrote %s' % plot_file)

            plot_file = os.path.join(self.args.plot_directory, 'probability_ratios')
            plot.histogram(plot_file, probability_ratios, 50)
            log('[Testing] wrote %s' % plot_file)

            plot_file = os.path.join(self.args.plot_directory, 'test_probabilities')
            plot.histogram(plot_file, self.test_probabilities[numpy.arange(self.test_probabilities.shape[0]), self.test_codes], 50)
            log('[Testing] wrote %s' % plot_file)

        y_true = numpy.concatenate((numpy.zeros(confidences.shape[0]), numpy.ones(perturbation_confidences.shape[0])))
        y_score = numpy.concatenate((confidences, perturbation_confidences))
        roc_auc_score = sklearn.metrics.roc_auc_score(y_true, y_score)

        self.results['raw_roc'] = roc_auc_score
        self.results['raw_confidence_weighted_success'] = numpy.sum(confidences[raw_overall_success]) / numpy.sum(perturbation_confidences)
        self.results['raw_confidence'] = numpy.mean(probabilities[raw_overall_success])
        self.results['raw_ratios'] = numpy.mean(probability_ratios[raw_overall_success])
        self.results['raw_success'] = numpy.sum(raw_overall_success) / success.shape[0]
        self.results['raw_iteration'] = numpy.average(success[raw_overall_success])

        if self.args.results_file:
            utils.write_pickle(self.args.results_file, self.results)
            log('[Testing] wrote %s' % self.args.results_file)

    def print_statistics(self):
        """
        Print statistics.
        """

        table_data = []

        for n in range(len(self.norms)):
            table_headings = ['(Class) Norm', 'Metric, Epsilon', 'Raw']
            table_data.append(table_headings)
            norm = self.norms[n]

            table_row = ['    L_%.3g' % norm, 'Success Rate', '%.3g' % self.results[n]['raw_success']]
            table_data.append(table_row)

            table_row = ['    L_%.3g' % norm, 'Confidence', '%.3g' % self.results[n]['raw_confidence']]
            table_data.append(table_row)

            table_row = ['    L_%.3g' % norm, 'Confidence Ratios', '%.3g' % self.results[n]['raw_ratios']]
            table_data.append(table_row)

            table_row = ['    L_%.3g' % norm, 'Confidence Weighted Success', '%.3g' % self.results[n]['raw_confidence_weighted_success']]
            table_data.append(table_row)

            table_row = ['    L_%.3g' % norm, 'Confidence ROC', '%.3g' % self.results[n]['raw_roc']]
            table_data.append(table_row)

            table_data.append(['---']*2)

        table = terminaltables.AsciiTable(table_data)
        log(table.table)

    def main(self):
        """
        Main.
        """

        self.load_data()
        self.load_models()
        self.test()
        self.compute_statistics()
        self.print_statistics()


if __name__ == '__main__':
    program = TestAttackClassifierDistal()
    program.main()
