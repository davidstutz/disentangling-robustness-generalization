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


class PlotAttackLearnedDecoderClassifier:
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

        self.test_theta = None
        """ (numpy.ndarray) Transformations for testing. """

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
        parser.add_argument('-test_theta_file', default=paths.results_file('test_theta'), help='HDF5 file containing theta.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-classifier_file', default=paths.state_file('classifier'), help='Snapshot state file of classifier.', type=str)
        parser.add_argument('-decoder_files', default=paths.state_file('decoder'), help='Decoder state file.', type=str)
        parser.add_argument('-output_directory', default=paths.experiment_dir('output'), help='Output directory.', type=str)
        parser.add_argument('-log_file', default=paths.log_file('learned_decoder/attacks'), help='Log file.', type=str)
        parser.add_argument('-attack', default='UntargetedBatchL2ClippedGradientDescent', help='Attack to try.', type=str)
        parser.add_argument('-objective', default='UntargetedF6', help='Objective to use.', type=str)
        parser.add_argument('-max_attempts', default=1, help='Maximum number of attempts per attack.', type=int)
        parser.add_argument('-max_samples', default=10, help='How many samples from the test set to attack.', type=int)
        parser.add_argument('-epsilon', default=0.1, help='Epsilon allowed for attacks.', type=float)
        parser.add_argument('-c_0', default=0., help='Weight of norm.', type=float)
        parser.add_argument('-c_1', default=0.1, help='Weight of bound, if not enforced through clipping or reparameterization.', type=float)
        parser.add_argument('-c_2', default=0.5, help='Weight of objective.', type=float)
        parser.add_argument('-max_iterations', default=100, help='Number of iterations for attack.', type=int)
        parser.add_argument('-max_projections', default=5, help='Number of projections for alternating projection.', type=int)
        parser.add_argument('-base_lr', default=0.005, help='Learning rate for attack.', type=float)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')
        parser.add_argument('-on_manifold', dest='on_manifold', action='store_true')

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

    def setup_attack(self, batch_inputs, batch_classes):
        """
        Setup attack.

        :param batch_inputs: input to attack
        :type batch_inputs: torch.autograd.Variable
        :param batch_classes: true classes
        :type batch_classes: torch.autograd.Variable
        :return: attack
        :rtype: attacks.UntargetedAttack
        """

        attack = self.attack_class(self.model, batch_inputs, batch_classes, self.args.epsilon)

        if self.args.on_manifold:
            attack.set_bound(torch.from_numpy(self.min_bound), torch.from_numpy(self.max_bound))
        else:
            attack.set_bound(None, None)

        if getattr(attack, 'set_c_0', None) is not None:
            attack.set_c_0(self.args.c_0)
        if getattr(attack, 'set_c_1', None) is not None:
            attack.set_c_1(self.args.c_1)
        if getattr(attack, 'set_c_2', None) is not None:
            attack.set_c_2(self.args.c_2)
        if getattr(attack, 'set_max_projections', None) is not None:
            attack.set_max_projections(self.args.max_projections)

        assert attack.training_mode is False

        attack.set_max_iterations(self.args.max_iterations)
        attack.set_base_lr(self.args.base_lr)
        attack.initialize_zero()

        return attack

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
            batch_theta = common.torch.as_variable(numpy.zeros((batch_size, self.args.latent_space_size), dtype=numpy.float32), self.args.use_gpu)

            objective = self.objective_class()
            self.model.decoder.set_code(batch_classes)
            reference_logits = self.model.forward(batch_theta)

            attack = self.setup_attack(batch_theta, batch_classes)
            success, perturbations, _, _, _ = attack.run(objective)

            steps = numpy.linspace(-1, 5, 200)
            log('[Visualization] %d class=%d, predicted=%d' % (i, batch_classes[0], torch.max(reference_logits, dim=1)[1]))
            log('[Visualization] %d success=%d' % (i, success[0]))

            for s in range(len(steps)):
                step = steps[s]
                batch_theta = common.torch.as_variable(numpy.expand_dims(step*perturbations[0].astype(numpy.float32), axis=0), self.args.use_gpu)

                output_images = self.model.decoder(batch_theta)
                output_logits = self.model.classifier(output_images)
                f = objective.f(output_logits, reference_logits, batch_classes)

                #from matplotlib import pyplot
                #pyplot.imshow(output_images[0, 0].cpu().detach().numpy())
                #pyplot.show()
                self.data[i, s, 0] = batch_theta[:, 5]
                self.data[i, s, 1] = f.item()
                log('[Visualization] %d step=%g, f=%g' % (i, step, f))

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

        assert self.args.decoder_files
        decoder_files = self.args.decoder_files.split(',')
        for decoder_file in decoder_files:
            assert os.path.exists(decoder_file), 'could not find %s' % decoder_file

        decoder_units = list(map(int, self.args.decoder_units.split(',')))
        log('[Visualization] using %d input channels' % self.image_channels)

        if len(decoder_files) > 1:
            log('[Visualization] loading multiple decoders')
            decoders = []
            for i in range(len(decoder_files)):
                decoder = models.LearnedDecoder(self.args.latent_space_size, resolution=(self.image_channels, self.resolution, self.resolution),
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
                log('[Training] loaded %s' % decoder_files[i])
            decoder = models.SelectiveDecoder(decoders, resolution=(self.image_channels, self.resolution, self.resolution))
        else:
            log('[Visualization] loading one decoder')
            decoder = models.LearnedDecoder(self.args.latent_space_size, resolution=(self.image_channels, self.resolution, self.resolution),
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

        self.attack_class = utils.get_class('attacks', self.args.attack)
        if not self.attack_class:
            log('[Error] could not find attack %s' % self.args.attack, LogLevel.ERROR)
            exit(1)
        log('[Visualization] found %s' % self.attack_class)
        # attack is instantiated per sample

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
    program = PlotAttackLearnedDecoderClassifier()
    program.main()