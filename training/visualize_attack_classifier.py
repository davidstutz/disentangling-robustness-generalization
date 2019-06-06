import os
import sys

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
import models
from common.log import log, Log, LogLevel
from common.state import State
from common import cuda
from common import paths
import common.torch
import common.numpy
import torch
import math
import numpy
import argparse
from matplotlib import pyplot
from common import vis


class VisualizeAttackClassifier:
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

        self.test_codes = None
        """ (numpy.ndarray) Test codes. """

        self.model = None
        """ (encoder.Encoder) Model to train. """

        self.perturbations = None
        """ (numpy.ndarray) Perturbations per test image. """

        self.success = None
        """ (numpy.ndarray) Success indicator. """

        self.image_representations = None
        """ (numpy.ndarray) Representations for visualization. """

        self.perturbation_representations = None
        """ (numpy.ndarray) Representations for visualization. """

        log('-- ' + self.__class__.__name__)
        for key in vars(self.args):
            log('[Visualization] %s=%s' % (key, str(getattr(self.args, key))))

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Visualize attacks on classifier.')
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-classifier_file', default=paths.state_file('classifier'), help='Snapshot state file of classifier.', type=str)
        parser.add_argument('-perturbations_file', default=paths.results_file('classifier/perturbations'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-success_file', default=paths.results_file('classifier/success'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-accuracy_file', default=paths.results_file('classifier/accuracy'), help='Correctly classified test samples of classifier.', type=str)
        parser.add_argument('-selection_file', default='', help='Selection file.', type=str)
        parser.add_argument('-output_directory', default=paths.experiment_dir('classifier/perturbations'), help='Directory to store visualizations.', type=str)
        parser.add_argument('-batch_size', default=128, help='Batch size of attack.', type=int)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')

        # Some network parameters.
        parser.add_argument('-network_architecture', default='standard', help='Classifier architecture to use.', type=str)
        parser.add_argument('-network_activation', default='relu', help='Activation function to use.', type=str)
        parser.add_argument('-network_no_batch_normalization', default=False, help='Do not use batch normalization.', action='store_true')
        parser.add_argument('-network_channels', default=16, help='Channels of first convolutional layer, afterwards channels are doubled.', type=int)
        parser.add_argument('-network_dropout', default=False, action='store_true', help='Whether to use dropout.')
        parser.add_argument('-network_units', default='1024,1024,1024,1024', help='Units for MLP.')

        return parser

    def load_data_and_model(self):
        """
        Load data and model.
        """

        self.test_images = utils.read_hdf5(self.args.test_images_file).astype(numpy.float32)
        if len(self.test_images.shape) < 4:
            self.test_images = numpy.expand_dims(self.test_images, axis=3)
        resolution = self.test_images.shape[2]
        log('[Visualization] read %s' % self.args.test_images_file)

        self.test_codes = utils.read_hdf5(self.args.test_codes_file).astype(numpy.int)
        self.test_codes = self.test_codes[:, self.args.label_index]
        N_class = numpy.max(self.test_codes) + 1
        log('[Visualization] read %s' % self.args.test_codes_file)

        self.perturbations = utils.read_hdf5(self.args.perturbations_file).astype(numpy.float32)
        if len(self.perturbations.shape) < 5:
            self.perturbations = numpy.expand_dims(self.perturbations, axis=4)

        self.perturbations = numpy.swapaxes(self.perturbations, 0, 1)
        self.test_images = self.test_images[:self.perturbations.shape[0]]
        log('[Visualization] read %s' % self.args.perturbations_file)

        self.success = utils.read_hdf5(self.args.success_file)
        self.success = numpy.swapaxes(self.success, 0, 1)
        self.success = self.success >= 0
        log('[Visualization] read %s' % self.args.success_file)

        if self.args.selection_file:
            selection = utils.read_hdf5(self.args.selection_file)
            log('[Visualization] read %s' % self.args.selection_file)

            selection = numpy.swapaxes(selection, 0, 1)
            selection = selection[:self.success.shape[0]]
            selection = selection >= 0

            assert len(selection.shape) == len(self.success.shape)
            self.success = numpy.logical_and(self.success, selection)
            log('[Visualization] updated selection')

        self.accuracy = utils.read_hdf5(self.args.accuracy_file)
        log('[Visualization] read %s' % self.args.success_file)

        log('[Visualization] using %d input channels' % self.test_images.shape[3])
        network_units = list(map(int, self.args.network_units.split(',')))
        self.model = models.Classifier(N_class, resolution=(self.test_images.shape[3], self.test_images.shape[1], self.test_images.shape[2]),
                                       architecture=self.args.network_architecture,
                                       activation=self.args.network_activation,
                                       batch_normalization=not self.args.network_no_batch_normalization,
                                       start_channels=self.args.network_channels,
                                       dropout=self.args.network_dropout,
                                       units=network_units)

        assert os.path.exists(self.args.classifier_file), 'state file %s not found' % self.args.classifier_file
        state = State.load(self.args.classifier_file)
        log('[Visualization] read %s' % self.args.classifier_file)

        self.model.load_state_dict(state.model)
        if self.args.use_gpu and not cuda.is_cuda(self.model):
            log('[Visualization] classifier is not CUDA')
            self.model = self.model.cuda()
        log('[Visualization] loaded classifier')

        self.model.eval()
        log('[Visualization] set model to eval')

    def run_model(self):
        """
        Run model.
        """

        def run(model, images, batch_size, use_gpu):
            """
            Run the model for given images.

            :param model: classifier
            :type model: torch.nn.Module
            :param images: images
            :type images: numpy.ndarray
            :param batch_size: batch size
            :type batch_size: int
            :param use_gpu: whether to use GPU
            :type use_gpu: bool
            :return: representations
            :rtype: numpy.ndarray
            """

            assert model.training is False

            representations = None
            num_batches = int(math.ceil(images.shape[0] / batch_size))

            for b in range(num_batches):
                b_start = b * batch_size
                b_end = min((b + 1) * batch_size, images.shape[0])
                batch_images = common.torch.as_variable(images[b_start: b_end], use_gpu)
                batch_images = batch_images.permute(0, 3, 1, 2)

                output_representations = torch.nn.functional.softmax(model.forward(batch_images), dim=1)
                representations = common.numpy.concatenate(representations, output_representations.data.cpu().numpy())
                log('[Visualization] %d/%d' % (b + 1, num_batches))

            return representations

        self.image_representations = run(self.model, self.test_images, self.args.batch_size, self.args.use_gpu)

        shape = self.perturbations.shape
        perturbations = self.perturbations.reshape((shape[0] * shape[1], shape[2], shape[3], shape[4]))
        self.perturbation_representations = run(self.model, perturbations, self.args.batch_size, self.args.use_gpu)
        self.perturbation_representations = self.perturbation_representations.reshape((shape[0], shape[1], -1))

    def visualize_perturbations(self):
        """
        Visualize perturbations.
        """

        num_attempts = self.perturbations.shape[1]
        num_attempts = min(num_attempts, 6)
        utils.makedir(self.args.output_directory)

        count = 0
        for i in range(min(1000, self.perturbations.shape[0])):

            if not numpy.any(self.success[i]) or not self.accuracy[i]:
                continue
            elif count > 200:
                break

            #fig, axes = pyplot.subplots(num_attempts, 5)
            #if num_attempts == 1:
            #    axes = [axes] # dirty hack for axis indexing

            for j in range(num_attempts):
                image = self.test_images[i]
                attack = self.perturbations[i][j]
                perturbation = attack - image
                max_perturbation = numpy.max(numpy.abs(perturbation))
                perturbation /= max_perturbation

                image_representation = self.image_representations[i]
                attack_representation = self.perturbation_representations[i][j]

                image_label = numpy.argmax(image_representation)
                attack_label = numpy.argmax(attack_representation)

                #axes[j][0].imshow(numpy.squeeze(image), interpolation='nearest', cmap='gray', vmin=0, vmax=1)
                #axes[j][1].imshow(numpy.squeeze(perturbation), interpolation='nearest', cmap='seismic', vmin=-1, vmax=1)
                #axes[j][1].text(0, -image.shape[1]//8, 'x' + str(max_perturbation))
                #axes[j][2].imshow(numpy.squeeze(attack), interpolation='nearest', cmap='gray', vmin=0, vmax=1)

                #vmin = min(numpy.min(image_representation), numpy.min(attack_representation))
                #vmax = max(numpy.max(image_representation), numpy.max(attack_representation))
                #axes[j][3].imshow(image_representation.reshape(1, -1), interpolation='nearest', vmin=vmin, vmax=vmax)
                #axes[j][3].text(0, -1, 'Label:' + str(image_label))
                #axes[j][4].imshow(attack_representation.reshape(1, -1), interpolation='nearest', vmin=vmin, vmax=vmax)
                #axes[j][4].text(0, -1, 'Label:' + str(attack_label))

                image_file = os.path.join(self.args.output_directory, '%d_%d_image_%d.png' % (i, j, image_label))
                attack_file = os.path.join(self.args.output_directory, '%d_%d_attack_%d.png' % (i, j, attack_label))
                perturbation_file = os.path.join(self.args.output_directory, '%d_%d_perturbation_%g.png' % (i, j, max_perturbation))

                vis.image(image_file, image, scale=10)
                vis.image(attack_file, attack, scale=10)
                vis.perturbation(perturbation_file, perturbation, scale=10)

                if len(perturbation.shape) > 2:
                    perturbation_magnitude = numpy.linalg.norm(perturbation, ord=2, axis=2)
                    max_perturbation_magnitude = numpy.max(numpy.abs(perturbation_magnitude))
                    perturbation_magnitude /= max_perturbation_magnitude

                    perturbation_file = os.path.join(self.args.output_directory, '%d_%d_perturbation_magnitude_%g.png' % (i, j, max_perturbation_magnitude))
                    vis.perturbation(perturbation_file, perturbation_magnitude, scale=10)

            #plot_file = os.path.join(self.args.output_directory, str(i) + '.png')
            #pyplot.savefig(plot_file)
            #pyplot.close(fig)
            count += 1

    def main(self):
        """
        Main.
        """

        self.load_data_and_model()
        self.run_model()
        self.visualize_perturbations()


if __name__ == '__main__':
    program = VisualizeAttackClassifier()
    program.main()
