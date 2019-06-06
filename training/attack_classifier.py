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
import math
import torch
import numpy
import argparse


class AttackClassifier:
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
        """ (numpy.ndarray) Codes for testing. """

        self.attack_class = None
        """ (attacks.UntargetedAttack) Attack to use (as class). """

        self.objective_class = None
        """ (attacks.UntargetedObjective) Objective to use (as class). """

        self.model = None
        """ (encoder.Encoder) Model to train. """

        self.perturbations = None
        """ (numpy.ndarray) Perturbations per test image. """

        self.accuracy = None
        """ (numpy.ndarray) Success of classifier. """

        self.success = None
        """ (numpy.ndarray) Success per test image. """

        if self.args.log_file:
            utils.makedir(os.path.dirname(self.args.log_file))
            Log.get_instance().attach(open(self.args.log_file, 'w'))

        log('-- ' + self.__class__.__name__)
        for key in vars(self.args):
            log('[Attack] %s=%s' % (key, str(getattr(self.args, key))))

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

        parser = argparse.ArgumentParser(description='Attack classifier.')
        parser.add_argument('-test_images_file', default=paths.test_images_file(), help='HDF5 file containing dataset.', type=str)
        parser.add_argument('-test_codes_file', default=paths.test_codes_file(), help='HDF5 file containing codes.', type=str)
        parser.add_argument('-label_index', default=2, help='Label index.', type=int)
        parser.add_argument('-classifier_file', default=paths.state_file('classifier'), help='Snapshot state file of classifier.', type=str)
        parser.add_argument('-accuracy_file', default=paths.results_file('classifier/accuracy'), help='Correctly classified test samples of classifier.', type=str)
        parser.add_argument('-perturbations_file', default=paths.results_file('classifier/perturbations'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-success_file', default=paths.results_file('classifier/success'), help='HDF5 file containing perturbations.', type=str)
        parser.add_argument('-log_file', default=paths.log_file('classifier/attacks'), help='Log file.', type=str)
        parser.add_argument('-attack', default='UntargetedBatchL2ClippedGradientDescent', help='Attack to try.', type=str)
        parser.add_argument('-objective', default='UntargetedF6', help='Objective to use.', type=str)
        parser.add_argument('-max_attempts', default=1, help='Maximum number of attempts per attack.', type=int)
        parser.add_argument('-max_samples', default=20*128, help='How many samples from the test set to attack.', type=int)
        parser.add_argument('-batch_size', default=128, help='Batch size of attack.', type=int)
        parser.add_argument('-epsilon', default=1, help='Epsilon allowed for attacks.', type=float)
        parser.add_argument('-c_0', default=0., help='Weight of norm.', type=float)
        parser.add_argument('-c_1', default=0.1, help='Weight of bound, if not enforced through clipping or reparameterization.', type=float)
        parser.add_argument('-c_2', default=0.5, help='Weight of objective.', type=float)
        parser.add_argument('-max_iterations', default=250, help='Number of iterations for attack.', type=int)
        parser.add_argument('-max_projections', default=5, help='Number of projections for alternating projection.', type=int)
        parser.add_argument('-base_lr', default=0.005, help='Learning rate for attack.', type=float)
        parser.add_argument('-no_gpu', dest='use_gpu', action='store_false')
        parser.add_argument('-no_label_leaking', default=False, dest='no_label_leaking', action='store_true')
        parser.add_argument('-initialize_zero', default=False, action='store_true', help='Initialize attack at zero.')

        # Some network parameters.
        parser.add_argument('-network_architecture', default='standard', help='Classifier architecture to use.', type=str)
        parser.add_argument('-network_activation', default='relu', help='Activation function to use.', type=str)
        parser.add_argument('-network_no_batch_normalization', default=False, help='Do not use batch normalization.', action='store_true')
        parser.add_argument('-network_channels', default=16, help='Channels of first convolutional layer, afterwards channels are doubled.', type=int)
        parser.add_argument('-network_dropout', default=False, action='store_true', help='Whether to use dropout.')
        parser.add_argument('-network_units', default='1024,1024,1024,1024', help='Units for MLP.')

        return parser

    def setup_attack(self, batch_images, batch_classes):
        """
        Setup and initialize attack.

        :param batch_images: images to attack
        :type batch_images: torch.autograd.Variable
        :param batch_classes: true classes to attack
        :type batch_classes: torch.autograd.Variable
        """

        if self.args.no_label_leaking:
            attack = self.attack_class(self.model, batch_images, None, self.args.epsilon)
        else:
            attack = self.attack_class(self.model, batch_images, batch_classes, self.args.epsilon)

        if getattr(attack, 'set_c_0', None) is not None:
            attack.set_c_0(self.args.c_0)
        if getattr(attack, 'set_c_1', None) is not None:
            attack.set_c_1(self.args.c_1)
        if getattr(attack, 'set_c_2', None) is not None:
            attack.set_c_2(self.args.c_2)
        if getattr(attack, 'set_max_projections', None) is not None:
            attack.set_max_projections(self.args.max_projections)

        attack.set_max_iterations(self.args.max_iterations)
        attack.set_base_lr(self.args.base_lr)

        assert attack.training_mode is False

        if self.args.initialize_zero:
            attack.initialize_zero()
        else:
            attack.initialize_random()

        return attack

    def test(self):
        """
        Test classifier to identify valid samples to attack.
        """

        num_batches = int(math.ceil(self.test_images.shape[0] / self.args.batch_size))
        self.model.eval()
        assert self.model.training is False
        assert self.test_images.shape[0] == self.test_codes.shape[0], 'number of samples have to match'

        for b in range(num_batches):
            b_start = b * self.args.batch_size
            b_end = min((b + 1) * self.args.batch_size, self.test_images.shape[0])
            batch_images = common.torch.as_variable(self.test_images[b_start: b_end], self.args.use_gpu)
            batch_classes = common.torch.as_variable(self.test_codes[b_start: b_end], self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            output_classes = self.model(batch_images)
            values, indices = torch.max(torch.nn.functional.softmax(output_classes, dim=1), dim=1)
            errors = torch.abs(indices - batch_classes)

            self.accuracy = common.numpy.concatenate(self.accuracy, errors.data.cpu().numpy())

            if b % 100 == 0:
                log('[Attack] computing accuracy %d' % b)

        self.accuracy = self.accuracy == 0
        utils.write_hdf5(self.args.accuracy_file, self.accuracy)
        log('[Attack] wrote %s' % self.args.accuracy_file)

        accuracy = numpy.sum(self.accuracy)/float(self.accuracy.shape[0])
        log('[Attack] accuracy %g' % accuracy)
        accuracy = numpy.sum(self.accuracy[:self.args.max_samples]) / float(self.args.max_samples)
        log('[Attack] accuracy on %d samples %g' % (self.args.max_samples, accuracy))

    def attack(self):
        """
        Test the model.
        """

        assert self.model is not None
        assert self.model.training is False
        assert self.test_images.shape[0] == self.test_codes.shape[0], 'number of samples has to match'

        concatenate_axis = -1
        if os.path.exists(self.args.perturbations_file) and os.path.exists(self.args.success_file):
            self.original_perturbations = utils.read_hdf5(self.args.perturbations_file)
            if self.test_images.shape[3] > 1:
                assert len(self.original_perturbations.shape) == 5
            else:
                assert len(self.original_perturbations.shape) == 4
            log('[Attack] read %s' % self.args.perturbations_file)

            self.original_success = utils.read_hdf5(self.args.success_file)
            log('[Attack] read %s' % self.args.success_file)

            assert self.original_perturbations.shape[0] == self.original_success.shape[0]
            assert self.original_perturbations.shape[1] == self.original_success.shape[1]
            assert self.original_perturbations.shape[2] == self.test_images.shape[1]
            assert self.original_perturbations.shape[3] == self.test_images.shape[2]#

            if self.original_perturbations.shape[1] >= self.args.max_samples and self.original_perturbations.shape[0] >= self.args.max_attempts:
                log('[Attack] found %d attempts, %d samples, requested no more' % (self.original_perturbations.shape[0], self.original_perturbations.shape[1]))
                return
            elif self.original_perturbations.shape[0] == self.args.max_attempts or self.original_perturbations.shape[1] == self.args.max_samples:
                if self.original_perturbations.shape[0] == self.args.max_attempts:
                    self.test_images = self.test_images[self.original_perturbations.shape[1]:]
                    self.test_codes = self.test_codes[self.original_perturbations.shape[1]:]
                    self.args.max_samples = self.args.max_samples - self.original_perturbations.shape[1]
                    concatenate_axis = 1
                    log('[Attack] found %d attempts with %d perturbations, computing %d more perturbations' % (self.original_perturbations.shape[0], self.original_perturbations.shape[1], self.args.max_samples))
                elif self.original_perturbations.shape[1] == self.args.max_samples:
                    self.args.max_attempts = self.args.max_attempts - self.original_perturbations.shape[0]
                    concatenate_axis = 0
                    log('[Attack] found %d attempts with %d perturbations, computing %d more attempts' % (self.original_perturbations.shape[0], self.original_perturbations.shape[1], self.args.max_attempts))

        # can't squeeze here!
        if self.test_images.shape[3] > 1:
            self.perturbations = numpy.zeros((self.args.max_attempts, self.args.max_samples, self.test_images.shape[1], self.test_images.shape[2], self.test_images.shape[3]))
        else:
            self.perturbations = numpy.zeros((self.args.max_attempts, self.args.max_samples, self.test_images.shape[1], self.test_images.shape[2]))
        self.success = numpy.ones((self.args.max_attempts, self.args.max_samples), dtype=int) * -1

        if self.args.attack.find('Batch') >= 0:
            batch_size = min(self.args.batch_size, self.args.max_samples)
        else:
            batch_size = 1

        objective = self.objective_class()
        num_batches = int(math.ceil(self.args.max_samples/batch_size))

        for i in range(num_batches):  # self.test_images.shape[0]
            if i*batch_size == self.args.max_samples:
                break
                
            i_start = i*batch_size
            i_end = min((i+1)*batch_size, self.args.max_samples)

            batch_images = common.torch.as_variable(self.test_images[i_start: i_end], self.args.use_gpu)
            batch_classes = common.torch.as_variable(numpy.array(self.test_codes[i_start: i_end]), self.args.use_gpu)
            batch_images = batch_images.permute(0, 3, 1, 2)

            t = 0
            while t < self.args.max_attempts:
                attack = self.setup_attack(batch_images, batch_classes)
                success, perturbations, probabilities, norm, _ = attack.run(objective)
                assert not numpy.any(perturbations != perturbations), perturbations

                # Note that we save the perturbed image, not only the perturbation!
                self.perturbations[t][i_start: i_end] = numpy.squeeze(numpy.transpose(perturbations + batch_images.cpu().numpy(), (0, 2, 3, 1)))
                self.success[t][i_start: i_end] = success

                # IMPORTANT: The adversarial examples are not considering whether the classifier is
                # actually correct to start with.

                t += 1

            log('[Attack] %d: completed' % i)

        if concatenate_axis >= 0:
            if self.perturbations.shape[0] == self.args.max_attempts:
                self.perturbations = numpy.concatenate((self.original_perturbations, self.perturbations), axis=concatenate_axis)
                self.success = numpy.concatenate((self.original_success, self.success), axis=concatenate_axis)
                log('[Attack] concatenated')

        utils.write_hdf5(self.args.perturbations_file, self.perturbations)
        log('[Attack] wrote %s' % self.args.perturbations_file)
        utils.write_hdf5(self.args.success_file, self.success)
        log('[Attack] wrote %s' % self.args.success_file)

    def load_attack(self):
        """
        Load attack and objective.
        """

        self.attack_class = utils.get_class('attacks', self.args.attack)
        if not self.attack_class:
            log('[Error] could not find attack %s' % self.args.attack, LogLevel.ERROR)
            exit(1)
        log('[Attack] found %s' % self.attack_class)
        # attack is instantiated per sample

        self.objective_class = utils.get_class('attacks', self.args.objective)
        if not self.objective_class:
            log('[Error] could not find objective %s' % self.args.objective, LogLevel.ERROR)
            exit(1)
        log('[Attack] found %s' % self.objective_class)

    def load_models(self):
        """
        Load models.
        """

        N_class = numpy.max(self.test_codes) + 1
        network_units = list(map(int, self.args.network_units.split(',')))
        log('[Attack] using %d input channels' % self.test_images.shape[3])
        self.model = models.Classifier(N_class, resolution=(self.test_images.shape[3], self.test_images.shape[1], self.test_images.shape[2]),
                                       architecture=self.args.network_architecture,
                                       activation=self.args.network_activation,
                                       batch_normalization=not self.args.network_no_batch_normalization,
                                       start_channels=self.args.network_channels,
                                       dropout=self.args.network_dropout,
                                       units=network_units)
        assert os.path.exists(self.args.classifier_file), 'state file %s not found' % self.args.classifier_file
        state = State.load(self.args.classifier_file)
        log('[Attack] read %s' % self.args.classifier_file)

        self.model.load_state_dict(state.model)
        if self.args.use_gpu and not cuda.is_cuda(self.model):
            log('[Attack] classifier is not CUDA')
            self.model = self.model.cuda()
        log('[Attack] loaded classifier')

        # !
        self.model.eval()
        log('[Attack] set classifier to eval')

    def load_data(self):
        """
        Load data.
        """

        self.test_images = utils.read_hdf5(self.args.test_images_file).astype(numpy.float32)
        log('[Attack] read %s' % self.args.test_images_file)

        # For color and gray images.
        if len(self.test_images.shape) < 4:
            self.test_images = numpy.expand_dims(self.test_images, axis=3)

        self.test_codes = utils.read_hdf5(self.args.test_codes_file).astype(numpy.int)
        self.test_codes = self.test_codes[:, self.args.label_index]
        log('[Attack] read %s' % self.args.test_codes_file)

        if self.args.max_samples < 0:
            self.args.max_samples = self.test_images.shape[0]
        else:
            self.args.max_samples = min(self.args.max_samples, self.test_images.shape[0])

    def main(self):
        """
        Main.
        """

        self.load_data()
        self.load_models()
        self.load_attack()
        if not os.path.exists(self.args.accuracy_file):
            self.test()
        self.attack()


if __name__ == '__main__':
    program = AttackClassifier()
    program.main()
