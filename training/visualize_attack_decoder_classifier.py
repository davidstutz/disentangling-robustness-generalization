import os
import sys

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
import models
from common.log import log
from common.state import State
from common import cuda
from common import paths
import common.numpy
import common.torch

import math
import torch
import numpy
import argparse
from matplotlib import pyplot
from common import vis


class VisualizeAttackDecoderClassifier:
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

        self.test_theta = None
        """ (numpy.ndarray) Transformations for testing. """

        self.text_codes = None
        """ (numpy.ndarray) Codes for testing. """

        self.classifier = None
        """ (Classifier) Classifier to check. """

        self.decoder = None
        """ (Decoder) Decoder to check. """

        self.model = None
        """ (encoder.Encoder) Model to train. """

        self.perturbations = None
        """ (numpy.ndarray) Perturbations per test image. """

        self.success = None
        """ (numpy.ndarray) Success indicator. """

        self.theta_representations = None
        """ (numpy.ndarray) Representations for visualization. """

        self.perturbation_representations = None
        """ (numpy.ndarray) Representations for visualization. """

        self.theta_images = None
        """ (numpy.ndarray) Images for visualization. """

        self.perturbation_images = None
        """ (numpy.ndarray) Images for visualization. """

        log('-- ' + self.__class__.__name__)
        for key in vars(self.args):
            log('[Visualization] %s=%s' % (key, str(getattr(self.args, key))))

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Visualize attacks on decoder and classifier.')
        parser.add_argument('-database_file', default=paths.database_file(), help='HDF5 file containing font prototype images.', type=str)
        parser.add_argument('-test_theta_file', default=paths.test_theta_file(), help='HDF5 file containing transformations.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing images.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-classifier_file', default=paths.state_file('classifier'), help='Snapshot state file of classifier.', type=str)
        parser.add_argument('-perturbations_file', default=paths.results_file('decoder/perturbations'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-success_file', default=paths.results_file('decoder/success'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-accuracy_file', default=paths.results_file('decoder/accuracy'), help='Correctly classified test samples of classifier.', type=str)
        parser.add_argument('-output_directory', default=paths.experiment_dir('decoder/perturbations'), help='Directory to store visualizations.', type=str)
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

        database = utils.read_hdf5(self.args.database_file).astype(numpy.float32)
        log('[Visualization] read %s' % self.args.database_file)

        N_font = database.shape[0]
        N_class = database.shape[1]
        resolution = database.shape[2]

        database = database.reshape((database.shape[0] * database.shape[1], database.shape[2], database.shape[3]))
        database = torch.from_numpy(database)
        if self.args.use_gpu:
            database = database.cuda()
        database = torch.autograd.Variable(database, False)

        self.test_images = utils.read_hdf5(self.args.test_images_file).astype(numpy.float32)
        if len(self.test_images.shape) < 4:
            self.test_images = numpy.expand_dims(self.test_images, axis=3)

        self.perturbations = utils.read_hdf5(self.args.perturbations_file).astype(numpy.float32)
        self.perturbations = numpy.swapaxes(self.perturbations, 0, 1)
        log('[Visualization] read %s' % self.args.perturbations_file)

        self.success = utils.read_hdf5(self.args.success_file)
        self.success = numpy.swapaxes(self.success, 0, 1)
        log('[Visualization] read %s' % self.args.success_file)

        self.accuracy = utils.read_hdf5(self.args.accuracy_file)
        log('[Visualization] read %s' % self.args.success_file)

        self.test_theta = utils.read_hdf5(self.args.test_theta_file).astype(numpy.float32)
        self.test_theta = self.test_theta[:self.perturbations.shape[0]]
        N_theta = self.test_theta.shape[1]
        log('[Visualization] using %d N_theta' % N_theta)
        log('[Visualization] read %s' % self.args.test_theta_file)

        self.test_codes = utils.read_hdf5(self.args.test_codes_file).astype(numpy.int)
        self.test_codes = self.test_codes[:self.perturbations.shape[0]]
        self.test_codes = self.test_codes[:, 1:3]
        self.test_codes = numpy.concatenate((common.numpy.one_hot(self.test_codes[:, 0], N_font), common.numpy.one_hot(self.test_codes[:, 1], N_class)), axis=1).astype(numpy.float32)
        log('[Attack] read %s' % self.args.test_codes_file)

        image_channels = 1 if N_theta <= 7 else 3
        network_units = list(map(int, self.args.network_units.split(',')))
        log('[Visualization] using %d input channels' % image_channels)
        self.classifier = models.Classifier(N_class, resolution=(image_channels, resolution, resolution),
                                            architecture=self.args.network_architecture,
                                            activation=self.args.network_activation,
                                            batch_normalization=not self.args.network_no_batch_normalization,
                                            start_channels=self.args.network_channels,
                                            dropout=self.args.network_dropout,
                                            units=network_units)
        self.decoder = models.AlternativeOneHotDecoder(database, N_font, N_class, N_theta)
        self.decoder.eval()

        assert os.path.exists(self.args.classifier_file), 'state file %s not found' % self.args.classifier_file
        state = State.load(self.args.classifier_file)
        log('[Visualization] read %s' % self.args.classifier_file)

        self.classifier.load_state_dict(state.model)
        if self.args.use_gpu and not cuda.is_cuda(self.classifier):
            log('[Visualization] classifier is not CUDA')
            self.classifier = self.classifier.cuda()
        log('[Visualization] loaded classifier')

        self.classifier.eval()
        log('[Visualization] set classifier to eval')

    def run_model(self):
        """
        Run model.
        """

        def run(decoder, classifier, theta, codes, batch_size, use_gpu):
            """
            Run the model for given images.

            :param decoder: decoder
            :type decoder: torch.nn.Module
            :param classifier: classifier
            :type classifier: torch.nn.Module
            :param theta: transformation codes
            :type theta: numpy.ndarray
            :param codes: codes
            :type codes: numpy.ndarray
            :param batch_size: batch size
            :type batch_size: int
            :param use_gpu: whether to use GPU
            :type use_gpu: bool
            :return: representations, images
            :rtype: numpy.ndarray, numpy.ndarray
            """

            assert decoder.training is False
            assert classifier.training is False

            images = None
            representations = None
            num_batches = int(math.ceil(theta.shape[0] / batch_size))

            for b in range(num_batches):
                b_start = b * batch_size
                b_end = min((b + 1) * batch_size, theta.shape[0])

                batch_theta = common.torch.as_variable(theta[b_start: b_end], use_gpu)
                batch_code = common.torch.as_variable(codes[b_start: b_end], use_gpu)

                decoder.set_code(batch_code)
                output_images = decoder.forward(batch_theta)
                output_representations = torch.nn.functional.softmax(classifier.forward(output_images), dim=1)

                output_images = numpy.transpose(output_images.data.cpu().numpy(), (0, 2, 3, 1))
                images = common.numpy.concatenate(images, output_images)

                output_representations = output_representations.data.cpu().numpy()
                representations = common.numpy.concatenate(representations, output_representations)
                log('[Visualization] %d/%d' % (b + 1, num_batches))

            return representations, images

        self.theta_representations, self.theta_images = run(self.decoder, self.classifier, self.test_theta, self.test_codes, self.args.batch_size, self.args.use_gpu)

        shape = self.perturbations.shape
        perturbations = self.perturbations.reshape((shape[0] * shape[1], shape[2]))
        self.perturbation_representations, self.perturbation_images = run(self.decoder, self.classifier, perturbations, numpy.repeat(self.test_codes, shape[1], axis=0), self.args.batch_size, self.args.use_gpu)
        self.perturbation_representations = self.perturbation_representations.reshape((shape[0], shape[1], -1))
        self.perturbation_images = self.perturbation_images.reshape((shape[0], shape[1], self.perturbation_images.shape[1], self.perturbation_images.shape[2], self.perturbation_images.shape[3]))

    def visualize_perturbations(self):
        """
        Visualize perturbations.
        """

        num_attempts = self.perturbations.shape[1]
        num_attempts = min(num_attempts, 6)
        utils.makedir(self.args.output_directory)

        count = 0
        for i in range(min(1000, self.perturbations.shape[0])):

            log('[Visualization] sample %d, iterations %s and correctly classified: %s' % (i + 1, ' '.join(list(map(str, self.success[i]))), self.accuracy[i]))
            if not numpy.any(self.success[i] >= 0) or not self.accuracy[i]:
                continue
            elif count > 200:
                break

            #fig, axes = pyplot.subplots(num_attempts, 8)
            #if num_attempts == 1:
            #    axes = [axes] # dirty hack for axis indexing

            for j in range(num_attempts):
                theta = self.test_theta[i]
                theta_attack = self.perturbations[i][j]
                theta_perturbation = theta_attack - theta

                image = self.test_images[i]
                image_attack = self.perturbation_images[i][j]
                image_perturbation = image_attack - image

                max_theta_perturbation = numpy.max(numpy.abs(theta_perturbation))
                theta_perturbation /= max_theta_perturbation

                max_image_perturbation = numpy.max(numpy.abs(image_perturbation))
                image_perturbation /= max_image_perturbation

                image_representation = self.theta_representations[i]
                attack_representation = self.perturbation_representations[i][j]

                image_label = numpy.argmax(image_representation)
                attack_label = numpy.argmax(attack_representation)

                #vmin = min(numpy.min(theta), numpy.min(theta_attack))
                #vmax = max(numpy.max(theta), numpy.max(theta_attack))
                #axes[j][0].imshow(theta.reshape(1, -1), interpolation='nearest', vmin=vmin, vmax=vmax)
                #axes[j][1].imshow(numpy.squeeze(image), interpolation='nearest', cmap='gray', vmin=0, vmax=1)
                #axes[j][2].imshow(theta_perturbation.reshape(1, -1), interpolation='nearest', vmin=vmin, vmax=vmax)
                #axes[j][2].text(0, -1, 'x' + str(max_theta_perturbation))
                #axes[j][3].imshow(numpy.squeeze(image_perturbation), interpolation='nearest', cmap='seismic', vmin=-1, vmax=1)
                #axes[j][3].text(0, -image.shape[1]//8, 'x' + str(max_image_perturbation))
                #axes[j][4].imshow(theta_attack.reshape(1, -1), interpolation='nearest', vmin=vmin, vmax=vmax)
                #axes[j][5].imshow(numpy.squeeze(image_attack), interpolation='nearest', cmap='gray', vmin=0, vmax=1)

                #axes[j][6].imshow(image_representation.reshape(1, -1), interpolation='nearest', vmin=vmin, vmax=vmax)
                #axes[j][6].text(0, -1, 'Label:' + str(image_label))
                #axes[j][7].imshow(attack_representation.reshape(1, -1), interpolation='nearest', vmin=vmin, vmax=vmax)
                #axes[j][7].text(0, -1, 'Label:' + str(attack_label))

                image_file = os.path.join(self.args.output_directory, '%d_%d_image_%d.png' % (i, j, image_label))
                attack_file = os.path.join(self.args.output_directory, '%d_%d_attack_%d.png' % (i, j, attack_label))
                perturbation_file = os.path.join(self.args.output_directory, '%d_%d_perturbation_%g.png' % (i, j, max_image_perturbation))

                vis.image(image_file, image, scale=10)
                vis.image(attack_file, image_attack, scale=10)
                vis.perturbation(perturbation_file, image_perturbation, scale=10)

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
    program = VisualizeAttackDecoderClassifier()
    program.main()