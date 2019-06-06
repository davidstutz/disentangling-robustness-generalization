import os
import sys

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
import models
from common.log import log, LogLevel
from common.state import State
from common import cuda
from common import paths
import common.numpy
import common.torch
from common import vis
import math
import torch
import numpy
import argparse
from matplotlib import pyplot


class VisualizeAttackLearnedDecoderClassifier:
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
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing images.', type=str)
        parser.add_argument('-test_theta_file', default=paths.results_file('test_theta'), help='HDF5 file containing transformations.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-decoder_files', default=paths.state_file('decoder'), help='Decoder state file.', type=str)
        parser.add_argument('-classifier_file', default=paths.state_file('classifier'), help='Snapshot state file of classifier.', type=str)
        parser.add_argument('-perturbations_file', default=paths.results_file('learned_decoder/perturbations'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-success_file', default=paths.results_file('learned_decoder/success'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-accuracy_file', default=paths.results_file('learned_decoder/accuracy'), help='Correctly classified test samples of classifier.', type=str)
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

        # Some decoder parameters.
        parser.add_argument('-latent_space_size', default=10, help='Size of latent space.', type=int)
        parser.add_argument('-decoder_architecture', default='standard', help='Architecture to use.', type=str)
        parser.add_argument('-decoder_activation', default='relu', help='Activation function to use.', type=str)
        parser.add_argument('-decoder_no_batch_normalization', default=False, help='Do not use batch normalization.', action='store_true')
        parser.add_argument('-decoder_channels', default=16, help='Channels of first convolutional layer, afterwards channels are doubled.', type=int)
        parser.add_argument('-decoder_units', default='1024,1024,1024,1024', help='Units for MLP.')

        return parser

    def load_data_and_model(self):
        """
        Load data and model.
        """

        self.test_images = utils.read_hdf5(self.args.test_images_file).astype(numpy.float32)
        if len(self.test_images.shape) < 4:
            self.test_images = numpy.expand_dims(self.test_images, axis=3)
        resolution = (self.test_images.shape[3], self.test_images.shape[1], self.test_images.shape[2])
        log('[Visualization] read %s' % self.args.test_images_file)

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
        log('[Visualization] read %s' % self.args.test_theta_file)

        self.test_codes = utils.read_hdf5(self.args.test_codes_file).astype(numpy.int)
        self.test_codes = self.test_codes[:, self.args.label_index]
        self.N_class = numpy.max(self.test_codes) + 1
        self.test_codes = self.test_codes[:self.perturbations.shape[0]]
        log('[Visualization] read %s' % self.args.test_codes_file)

        network_units = list(map(int, self.args.network_units.split(',')))
        self.classifier = models.Classifier(self.N_class, resolution=resolution,
                                            architecture=self.args.network_architecture,
                                            activation=self.args.network_activation,
                                            batch_normalization=not self.args.network_no_batch_normalization,
                                            start_channels=self.args.network_channels,
                                            dropout=self.args.network_dropout,
                                            units=network_units)

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

        assert self.args.decoder_files
        decoder_files = self.args.decoder_files.split(',')
        for decoder_file in decoder_files:
            assert os.path.exists(decoder_file), 'could not find %s' % decoder_file

        log('[Visualization] using %d input channels' % self.test_images.shape[3])
        decoder_units = list(map(int, self.args.decoder_units.split(',')))

        if len(decoder_files) > 1:
            log('[Visualization] loading multiple decoders')
            decoders = []
            for i in range(len(decoder_files)):
                decoder = models.LearnedDecoder(self.args.latent_space_size, resolution=resolution,
                                                architecture=self.args.decoder_architecture,
                                                start_channels=self.args.decoder_channels,
                                                activation=self.args.decoder_activation,
                                                batch_normalization=not self.args.decoder_no_batch_normalization,
                                                units=decoder_units)

                state = State.load(decoder_files[i])
                decoder.load_state_dict(state.model)
                if self.args.use_gpu and not cuda.is_cuda(decoder):
                    decoder = decoder.cuda()
                decoders.append(decoder)

                decoder.eval()
                log('[Visualization] loaded %s' % decoder_files[i])
            self.decoder = models.SelectiveDecoder(decoders, resolution=resolution)
        else:
            log('[Visualization] loading one decoder')
            decoder = models.LearnedDecoder(self.args.latent_space_size, resolution=resolution,
                                            architecture=self.args.decoder_architecture,
                                            start_channels=self.args.decoder_channels,
                                            activation=self.args.decoder_activation,
                                            batch_normalization=not self.args.decoder_no_batch_normalization,
                                            units=decoder_units)

            state = State.load(decoder_files[0])
            decoder.load_state_dict(state.model)
            if self.args.use_gpu and not cuda.is_cuda(decoder):
                decoder = decoder.cuda()
            decoder.eval()
            log('[Visualization] read decoder')

            self.decoder = decoder

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

            assert classifier.training is False
            assert decoder.training is False

            images = None
            representations = None
            num_batches = int(math.ceil(theta.shape[0] / batch_size))

            for b in range(num_batches):
                b_start = b * batch_size
                b_end = min((b + 1) * batch_size, theta.shape[0])

                batch_theta = common.torch.as_variable(theta[b_start: b_end], use_gpu)
                batch_code = common.torch.as_variable(codes[b_start: b_end], use_gpu)

                if isinstance(decoder, models.SelectiveDecoder):
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
            if not (numpy.any(self.success[i] >= 0) or not self.accuracy[i]) and i != 152:
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

                if len(image_perturbation.shape) > 2:
                    perturbation_magnitude = numpy.linalg.norm(image_perturbation, ord=2, axis=2)
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
    program = VisualizeAttackLearnedDecoderClassifier()
    program.main()