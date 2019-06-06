import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from common import utils
import models
from common.log import log, Log, LogLevel
from common.state import State
from common import cuda
from common import paths
import common.numpy
from common import plot
import math
import torch
import numpy
import argparse


class PlotAttackSTNClassifier:
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

        self.test_codes = None
        """ (numpy.ndarray) Test classes. """

        self.test_images = None
        """ (numpy.ndarray) Test images. """

        self.resolution = None
        """ (int) Resolution. """

        self.N_class = None
        """ (int) Number of classes. """

        self.attack_class = None
        """ (attacks.UntargetedAttack) Attack to use (as class). """

        self.objective_class = None
        """ (attacks.UntargetedObjective) Objective to use (as class). """

        self.model = None
        """ (encoder.Encoder) Model to train. """

        self.perturbations = None
        """ (numpy.ndarray) Perturbations per test image. """

        self.success = None
        """ (numpy.ndarray) Success per test image. """

        self.min_bound = None
        """ (numpy.ndarray) Minimum bound for codes. """

        self.max_bound = None
        """ (numpy.ndarray) Maximum bound for codes. """

        if self.args.log_file:
            utils.makedir(os.path.dirname(self.args.log_file))
            Log.get_instance().attach(open(self.args.log_file, 'w'))

        log('-- ' + self.__class__.__name__)
        for key in vars(self.args):
            log('[Visualization] %s=%s' % (key, str(getattr(self.args, key))))

    def __del__(self):
        """
        Remove log file.
        """

        if self.args is not None:
            if self.args.log_file:
                Log.get_instance().detach(self.args.log_file)

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Attack decoder and classifier.')
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-classifier_file', default=paths.state_file('classifier'), help='Snapshot state file of classifier.', type=str)
        parser.add_argument('-output_directory', default=paths.experiment_dir('output'), help='Output directory.', type=str)
        parser.add_argument('-log_file', default=paths.log_file('learned_decoder/attacks'), help='Log file.', type=str)
        parser.add_argument('-objective', default='UntargetedF0', help='Objective to use.', type=str)
        parser.add_argument('-max_samples', default=10, help='How many samples from the test set to attack.', type=int)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')
        parser.add_argument('-N_theta', default=6, help='Numer of transformations.', type=int)
        parser.add_argument('-translation_x', default='-0.1,0.1', type=str, help='Minimum and maximum translation in x.')
        parser.add_argument('-translation_y', default='-0.1,0.1', type=str, help='Minimum and maximum translation in y')
        parser.add_argument('-shear_x', default='-0.25,0.25', type=str, help='Minimum and maximum shear in x.')
        parser.add_argument('-shear_y', default='-0.25,0.25', type=str, help='Minimum and maximum shear in y.')
        parser.add_argument('-scale', default='0.95,1.05', type=str, help='Minimum and maximum scale.')
        parser.add_argument('-rotation', default='%g,%g' % (-math.pi / 4, math.pi / 4), type=str, help='Minimum and maximum rotation.')
        parser.add_argument('-color', default=0.5, help='Minimum color value, maximum is 1.', type=float)

        # Some network parameters.
        parser.add_argument('-network_architecture', default='standard', help='Classifier architecture to use.', type=str)
        parser.add_argument('-network_activation', default='relu', help='Activation function to use.', type=str)
        parser.add_argument('-network_no_batch_normalization', default=False, help='Do not use batch normalization.', action='store_true')
        parser.add_argument('-network_channels', default=16, help='Channels of first convolutional layer, afterwards channels are doubled.', type=int)
        parser.add_argument('-network_dropout', default=False, action='store_true', help='Whether to use dropout.')
        parser.add_argument('-network_units', default='1024,1024,1024,1024', help='Units for MLP.')

        return parser

    def attack(self):
        """
        Test the model.
        """

        assert self.model is not None
        assert self.model.classifier.training is False
        #assert self.model.decoder.training is False

        batch_size = 1
        objective = self.objective_class()
        num_batches = int(math.ceil(self.args.max_samples/batch_size))
        self.data = numpy.zeros((self.args.max_samples, 200, 2))

        for i in range(num_batches):
            i_start = i * batch_size
            i_end = min((i + 1) * batch_size, self.args.max_samples)

            batch_classes = common.torch.as_variable(self.test_codes[i_start: i_end], self.args.use_gpu)
            batch_images = common.torch.as_variable(numpy.expand_dims(self.test_images[i_start: i_end], axis=0), self.args.use_gpu)
            batch_theta = common.torch.as_variable(numpy.zeros((batch_size, self.args.N_theta)).astype(numpy.float32), self.args.use_gpu)

            objective = self.objective_class()
            self.model.decoder.set_image(batch_images)
            reference_logits = self.model.forward(batch_theta)

            log('[Visualization] %d class=%d, predicted=%d' % (i, batch_classes[0], torch.max(reference_logits, dim=1)[1]))
            rotations = numpy.linspace(self.min_bound[5], self.max_bound[5], 200)
            for r in range(len(rotations)):
                rotation = rotations[r]
                batch_theta[:, 5] = rotation

                output_images = self.model.decoder(batch_theta)
                output_logits = self.model.classifier(output_images)
                f = objective.f(output_logits, reference_logits, batch_classes)

                #from matplotlib import pyplot
                #pyplot.imshow(output_images[0, 0].cpu().numpy())
                #pyplot.show()
                self.data[i, r, 0] = batch_theta[:, 5]
                self.data[i, r, 1] = f.item()
                log('[Visualization] %d rotation=%g, f=%g' % (i, rotation, f))

    def plot(self):
        """
        Plot.
        """

        if self.args.output_directory:
            utils.makedir(self.args.output_directory)
            for i in range(self.data.shape[0]):
                plot_file = paths.image_file('%s/%d' % (self.args.output_directory, i))
                plot.line(plot_file, self.data[i, :, 0], self.data[i, :, 1])

    def load_model(self):
        """
        Load the decoder.
        """

        assert self.args.N_theta > 0 and self.args.N_theta <= 9

        min_translation_x, max_translation_x = map(float, self.args.translation_x.split(','))
        min_translation_y, max_translation_y = map(float, self.args.translation_y.split(','))
        min_shear_x, max_shear_x = map(float, self.args.shear_x.split(','))
        min_shear_y, max_shear_y = map(float, self.args.shear_y.split(','))
        min_scale, max_scale = map(float, self.args.scale.split(','))
        min_rotation, max_rotation = map(float, self.args.rotation.split(','))
        min_color, max_color = self.args.color, 1

        self.min_bound = numpy.array([
            min_translation_x,
            min_translation_y,
            min_shear_x,
            min_shear_y,
            min_scale,
            min_rotation,
            min_color,
            min_color,
            min_color,
        ])
        self.max_bound = numpy.array([
            max_translation_x,
            max_translation_y,
            max_shear_x,
            max_shear_y,
            max_scale,
            max_rotation,
            max_color,
            max_color,
            max_color
        ])

        self.min_bound = self.min_bound[:self.args.N_theta].astype(numpy.float32)
        self.max_bound = self.max_bound[:self.args.N_theta].astype(numpy.float32)

        decoder = models.STNDecoder(self.args.N_theta)
        #decoder.eval()
        log('[Visualization] set up STN decoder')

        classifier = models.Classifier(self.N_class, resolution=(self.image_channels, self.resolution, self.resolution),
                                       architecture='standard',
                                       activation=self.args.network_activation,
                                       batch_normalization=not self.args.network_no_batch_normalization,
                                       start_channels=self.args.network_channels,
                                       dropout=self.args.network_dropout)

        assert os.path.exists(self.args.classifier_file), 'state file %s not found' % self.args.classifier_file
        state = State.load(self.args.classifier_file)
        log('[Visualization] read %s' % self.args.classifier_file)

        classifier.load_state_dict(state.model)
        if self.args.use_gpu and not cuda.is_cuda(classifier):
            log('[Visualization] classifier is not CUDA')
            classifier = classifier.cuda()
        classifier.eval()
        log('[Visualization] loaded classifier')

        self.model = models.DecoderClassifier(decoder, classifier)
        log('[Training] set up decoder classifier')

    def load_attack(self):
        """
        Load attack and objective:
        """

        self.objective_class = utils.get_class('attacks', self.args.objective)
        if not self.objective_class:
            log('[Error] could not find objective %s' % self.args.objective, LogLevel.ERROR)
            exit(1)
        log('[Visualization] found %s' % self.objective_class)

    def load_data(self):
        """
        Load data.
        """

        self.test_images = utils.read_hdf5(self.args.test_images_file).astype(numpy.float32)
        log('[Visualization] read %s' % self.args.test_images_file)

        self.test_codes = utils.read_hdf5(self.args.test_codes_file).astype(numpy.int)
        self.test_codes = self.test_codes[:, self.args.label_index]
        log('[Visualization] read %s' % self.args.test_codes_file)

        self.N_class = numpy.max(self.test_codes) + 1
        self.resolution = self.test_images.shape[1]
        self.image_channels = self.test_images.shape[3] if len(self.test_images.shape) > 3 else 1
        log('[Visualization] resolution %d' % self.resolution)

        if self.args.max_samples < 0:
            self.args.max_samples = self.test_codes.shape[0]
        else:
            self.args.max_samples = min(self.args.max_samples, self.test_codes.shape[0])

    def main(self):
        """
        Main.
        """

        self.load_data()
        self.load_model()
        self.load_attack()
        self.attack()
        self.plot()


if __name__ == '__main__':
    program = PlotAttackSTNClassifier()
    program.main()