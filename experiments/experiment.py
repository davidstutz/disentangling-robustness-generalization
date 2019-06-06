import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
from training.train_classifier import *
from training.train_classifier_augmented import *
from training.train_classifier_adversarially import *
from training.train_stn_classifier_augmented import *
from training.train_stn_classifier_adversarially import *
from training.train_stn_classifier_adversarially2 import *
from training.train_decoder_classifier_augmented import *
from training.train_decoder_classifier_adversarially import *
from training.train_decoder_classifier_adversarially2 import *
from training.train_learned_decoder_classifier_augmented import *
from training.train_learned_decoder_classifier_adversarially import *
from training.train_learned_decoder_classifier_adversarially2 import *
from training.test_classifier import *
from training.test_variational_auto_encoder import *
from training.attack_classifier import *
from training.test_attack_classifier import *
from training.attack_decoder_classifier import *
from training.test_attack_decoder_classifier import *
from training.attack_stn_classifier import *
from training.test_attack_stn_classifier import *
from training.attack_learned_decoder_classifier import *
from training.test_attack_learned_decoder_classifier import *
from training.test_perturbations import *
from training.compute_decoder_perturbations import *
from training.compute_learned_decoder_perturbations import *
from training.compute_stn_perturbations import *
from training.sample_variational_auto_encoder import *
if utils.display():
    from training.visualize_attack_classifier import *
    from training.visualize_attack_decoder_classifier import *
    from training.visualize_attack_learned_decoder_classifier import *
from training.detect_attack_classifier_nn import *
from training.detect_attack_decoder_classifier_nn import *
from training.detect_attack_learned_decoder_classifier_nn import *
from common.log import log, Log
from common import paths
from common import utils
from .options import *
import argparse
import shutil


class Experiment:
    """
    Prototypical experiment.
    """

    def __init__(self, args=None):
        """
        Constructor, also sets options.
        """

        self.args = None
        """ Arguments of program. """

        parser = self.get_parser()
        if args is not None:
            self.args = parser.parse_args(args)
        else:
            self.args = parser.parse_args()

        assert self.args.training_sizes is not None
        assert self.args.max_models is not None and self.args.max_models > 0
        assert self.args.mode is not None and self.args.mode in ['run', 'evaluate', 'visualize']

        self.labels = 10
        """ (int) Number of labels. """

        self.label_index = 2
        """ (int) Label index. """

        self.batch_size = 100
        """ (int) Batch size for computation, not training. """

        self.network_architecture = None
        """ ([str]) Network parameters. """

        self.decoder_architecture = None
        """ ([str]) Decoder parameters. """

        self.class_latent_space_size = None
        """ (int) Latent space size. """

        self.data_latent_space_size = None
        """ (int) Latent space size. """

        self.decoder_files = []
        """ ([str]) Decoder files for class manifolds. """

        self.decoder_file = None
        """ (str) Decoder file for data manifold. """

        self.encoder_files = []
        """ ([str]) Decoder files for class manifolds. """

        self.encoder_file = None
        """ (str) Decoder file for data manifold. """

        self.database_file = None
        """ (str) Database file. """

        self.train_images_file = None
        """ (str) Train images file. """

        self.test_images_file = None
        """ (str) Test images file. """

        self.train_codes_file = None
        """ (str) Train codes file. """

        self.test_codes_file = None
        """ (str) Test codes file. """

        self.train_theta_file = None
        """ (str) Train theta file. """

        self.test_theta_file = None
        """ (str) Test theta file. """

        self.class_train_images_file = None
        """ (str) Class manifold train images file. """

        self.class_test_images_file = None
        """ (str) Class manifold test images file. """

        self.data_train_images_file = None
        """ (str) Data manifold train images file. """

        self.data_test_images_file = None
        """ (str) Data manifold test images file. """

        self.class_train_theta_file = None
        """ (str) Class manifold train theta file. """

        self.class_test_theta_file = None
        """ (str) Class manifold test theta file. """

        self.data_train_theta_file = None
        """ (str) Data manifold train theta file. """

        self.data_test_theta_file = None
        """ (str) Data manifold test theta file. """

        self.class_sampled_images_file = None
        """ (str) Class manifold sampled images file. """

        self.class_sampled_codes_file = None
        """ (str) Class manifold sampled codes file. """

        assert self.args.training_sizes is not None
        training_sizes = list(map(int, self.args.training_sizes.split(',')))

        self.training_options = [TrainingOptions(training_size) for training_size in training_sizes]
        """ ([TrainingOptions]) Training options. """

        self.off_augmentation_options = None
        """ ([OffAugmentationOptions]) Augmentation options. """

        self.off_training_options = []
        """ ([OffAttackOptions]) Training options. """

        self.on_augmentation_options = None
        """ ([OnAugmentationOptions]) Augmentation options. """

        self.on_training_options = []
        """ ([OnAttackOptions]) Training options. """

        self.learned_on_class_augmentation_options = None
        """ ([LearnedOnClassAugmentationOptions]) Augmentation options. """

        self.learned_on_class_training_options = []
        """ ([LearnedOnClassAttackOptions]) Training options."""

        self.learned_on_data_augmentation_options = None
        """ ([LearnedOnClassAugmentationOptions]) Augmentation options. """

        self.learned_on_data_training_options = []
        """ ([LearnedOnDataAttackOptions]) Training options. """

        self.stn_augmentation_options = None
        """ ([STNAugmentationOptions]) Augmentation options. """

        self.stn_training_options = []
        """ ([STNAttackOptions]) Training options. """

        self.off_augmentation_options = None
        """ ([OffAugmentationOptions]) Augmentation options. """

        self.off_attack_options = []
        """ ([OffAttackOptions]) Attack options. """

        self.on_attack_options = []
        """ ([OnAttackOptions]) Attack options. """

        self.learned_on_class_attack_options = []
        """ ([LearnedOnClassAttackOptions]) Attack options. """

        self.learned_on_data_attack_options = []
        """ ([LearnedOnClassAttackOptions]) Attack options. """

        self.stn_attack_options = []
        """ ([STNAttackOptions]) Attack options. """

        self.results = dict()
        """ (dict) Results. """

        self.statistics = dict()
        """ (dict) Statistics. """

    def validate(self):
        """
        Validate.
        """

        assert len(self.decoder_files) == self.labels
        assert len(self.encoder_files) == self.labels

        for label in range(self.labels):
            assert os.path.exists(self.encoder_files[label]), self.encoder_files[label]
            assert os.path.exists(self.decoder_files[label]), self.decoder_files[label]
        assert self.encoder_file is None or os.path.exists(self.encoder_file), self.encoder_file
        assert self.decoder_file is None or os.path.exists(self.decoder_file), self.decoder_file

        assert self.database_file is None or os.path.exists(self.database_file), self.database_file
        assert self.train_theta_file is None or os.path.exists(self.train_theta_file), self.train_theta_file
        assert self.test_theta_file is None or os.path.exists(self.test_theta_file), self.test_theta_file

        assert os.path.exists(self.train_images_file)
        assert os.path.exists(self.test_images_file)
        assert os.path.exists(self.train_codes_file)
        assert os.path.exists(self.test_codes_file)

        paths.set_globals(experiment=self.experiment())
        log_file = paths.log_file('experiment_%s_%d_%s' % (self.args.training_sizes, self.args.max_models, self.args.mode))
        utils.makedir(os.path.dirname(log_file))
        Log.get_instance().attach(open(log_file, 'w'))
        log('-- ' + self.__class__.__name__)

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Experiment.')
        parser.add_argument('-mode', default='run', help='Mode.', type=str)
        parser.add_argument('-training_sizes', help='Training sizes.', type=str)
        parser.add_argument('-max_models', help='Number of models.', type=int)
        parser.add_argument('-start_model', default=0, help='Model to start with.', type=int)
        parser.add_argument('-reevaluate', action='store_true', default=False, help='Re-evaluate.')
        parser.add_argument('-revisualize', action='store_true', default=False, help='Re-visualize.')
        parser.add_argument('-reattack', action='store_true', default=False, help='Re-attack.')
        parser.add_argument('-evaluate_reconstructed', action='store_true', default=False, help='Evaluate on reconstructed.')
        parser.add_argument('-evaluate_sampled', action='store_true', default=False, help='Evaluate on reconstructed.')

        return parser

    def network_parameters(self):
        """
        Get network parameters.
        """

        if self.network_architecture is None:
            self.network_architecture = [
                '-network_architecture=standard',
                # '-network_no_batch_normalization',
                # '-network_dropout',
                '-network_activation=relu',
                '-network_channels=16',
                '-network_units=1024,1024,1024,1024',
            ]

        return self.network_architecture

    def decoder_parameters(self):
        """
        Get network parameters.
        """

        if self.decoder_architecture is None:
            self.decoder_architecture = [
                '-decoder_architecture=standard',
                # '-decoder_no_batch_normalization',
                # '-decoder_dropout',
                '-decoder_activation=relu',
                '-decoder_channels=64',  # !
                '-decoder_units=1024,1024,1024,1024',
            ]

        return self.decoder_architecture

    def experiment(self):
        """
        Experiment name.
        """

        raise NotImplementedError()

    def model_name(self, prefix, training_options, a1=None, a2=None):
        """
        Get model name.

        :param prefix: prefix
        :type prefix: str
        :param training_options: training options
        :type training_options: TrainingOptions
        :param a1: attack or augmentation
        :type a1: OffAttackOptions, OnAttackOptions, OffAugmentationOptions or OnAugmentationOptions
        :param a2: attack or augmentation
        :type a2: OffAttackOptions, OnAttackOptions, OffAugmentationOptions or OnAugmentationOptions
        """

        model_directory = '%s' % prefix

        if a1 is not None:
            model_directory += '_%d' % a1.max_iterations
            if a2 is not None:
                assert a1.max_iterations == a2.max_iterations

            # EMNIST FASHION FONTS
            if isinstance(a1, OffAttackMadryUnconstrainedOptions):
                model_directory += '_unconstrained'
            elif isinstance(a1, OnAttackMadryUnconstrainedOptions):
                model_directory += '_unconstrained'
            elif isinstance(a1, STNAttackMadryUnconstrainedOptions):
                model_directory += '_unconstrained'
            elif isinstance(a1, LearnedOnClassAttackMadryUnconstrainedOptions):
                model_directory += '_unconstrained'
            elif isinstance(a1, LearnedOnDataAttackMadryUnconstrainedOptions):
                model_directory += '_unconstrained'
            else:
                model_directory += '_%g' % a1.epsilon
                if a2 is not None:
                    assert a1.epsilon == a2.epsilon

            if hasattr(a1, 'bound'):
                model_directory += '_%g' % a1.bound
                if a2 is not None:
                    assert a1.bound == a1.bound

            if hasattr(a1, 'training_mode'):
                if a1.training_mode is True:
                    model_directory += '_training_mode'

            if hasattr(a1, 'full'):
                if a1.full is True:
                    model_directory += '_full'

        if training_options.suffix is not None:
            model_directory += '_%s' % training_options.suffix

        return model_directory

    def attack_directory(self, model_directory, attack):
        """
        Get attack directory name.

        :param model_directory: model directory to attack
        :type model_directory: str
        :param attack: attack
        :type attack: OnAttacKOptions, OffAttackOptions ...
        """

        attack_directory = '%s_%s_%d_%g' % (model_directory, attack.__class__.__name__, attack.max_iterations, attack.epsilon)
        if attack.suffix is not None:
            attack_directory += '_%s' % attack.suffix
        return attack_directory

    def compute_class_theta(self):
        """
        Compute theta.
        """

        decoder_architecture = []
        for option in self.decoder_parameters():
            decoder_architecture.append(option.replace('decoder', 'network'))

        for label in range(self.labels):
            assert os.path.exists(self.encoder_files[label]), self.encoder_files[label]
            assert os.path.exists(self.decoder_files[label]), self.decoder_files[label]

        self.class_train_images_file = paths.results_file('class_train_images')
        self.class_test_images_file = paths.results_file('class_test_images')
        self.class_train_theta_file = paths.results_file('class_train_theta')
        self.class_test_theta_file = paths.results_file('class_test_theta')
        self.class_sampled_images_file = paths.results_file('class_sampled_images')
        self.class_sampled_codes_file = paths.results_file('class_sampled_codes')

        for label in range(self.labels):
            train_theta_file = paths.results_file('train_theta_%d' % label)
            test_theta_file = paths.results_file('test_theta_%d' % label)
            train_images_file = paths.results_file('train_images_%d' % label)
            test_images_file = paths.results_file('test_images_%d' % label)

            log_file = paths.log_file('log_%d' % label)
            if not os.path.exists(train_theta_file) or not os.path.exists(test_theta_file) \
                    or not os.path.exists(train_images_file) or not os.path.exists(test_images_file):
                arguments = [
                    '-train_images_file=%s' % self.train_images_file,
                    '-test_images_file=%s' % self.test_images_file,
                    '-train_codes_file=%s' % self.train_codes_file,
                    '-test_codes_file=%s' % self.test_codes_file,
                    '-train_theta_file=%s' % train_theta_file,
                    '-test_theta_file=%s' % test_theta_file,
                    '-label_index=%d' % self.label_index,
                    '-label=%d' % label,
                    '-encoder_file=%s' % self.encoder_files[label],
                    '-decoder_file=%s' % self.decoder_files[label],
                    '-train_reconstruction_file=%s' % train_images_file,
                    '-reconstruction_file=%s' % test_images_file,
                    '-random_file=',
                    '-interpolation_file=',
                    '-batch_size=%d' % self.batch_size,
                    '-latent_space_size=%d' % self.class_latent_space_size,
                    '-results_file=',
                    '-output_directory=',
                    '-log_file=%s' % log_file
                ] + decoder_architecture
                test_variational_auto_encoder = TestVariationalAutoEncoder(arguments)
                test_variational_auto_encoder.main()

        if not os.path.exists(self.class_train_theta_file) or not os.path.exists(self.class_test_theta_file) \
                or not os.path.exists(self.class_train_images_file) or not os.path.exists(self.class_test_images_file):
            train_codes = utils.read_hdf5(self.train_codes_file)
            train_codes = train_codes[:, self.label_index]
            log('[Experiment] read %s' % self.train_codes_file)
            test_codes = utils.read_hdf5(self.test_codes_file)
            test_codes = test_codes[:, self.label_index]
            log('[Experiment] read %s' % self.test_codes_file)
            train_theta = None
            test_theta = None
            train_images = None
            test_images = None

            for label in range(self.labels):

                train_theta_file_c = paths.results_file('train_theta_%d' % label)
                test_theta_file_c = paths.results_file('test_theta_%d' % label)

                train_theta_c = utils.read_hdf5(train_theta_file_c)
                log('[Experiment] read %s' % train_theta_file_c)
                test_theta_c = utils.read_hdf5(test_theta_file_c)
                log('[Experiment] read %s' % test_theta_file_c)

                if train_theta is None:
                    train_theta = numpy.zeros((train_codes.shape[0], train_theta_c.shape[1]))
                if test_theta is None:
                    test_theta = numpy.zeros((test_codes.shape[0], test_theta_c.shape[1]))

                train_theta[train_codes == label] = train_theta_c
                test_theta[test_codes == label] = test_theta_c

                train_images_file_c = paths.results_file('train_images_%d' % label)
                test_images_file_c = paths.results_file('test_images_%d' % label)

                train_images_c = utils.read_hdf5(train_images_file_c)
                log('[Experiment] read %s' % train_images_file_c)
                test_images_c = utils.read_hdf5(test_images_file_c)
                log('[Experiment] read %s' % test_images_file_c)

                if train_images is None:
                    if len(train_images_c.shape) > 3:
                        train_images = numpy.zeros((train_codes.shape[0], train_images_c.shape[1], train_images_c.shape[2], train_images_c.shape[3]))
                    else:
                        train_images = numpy.zeros((train_codes.shape[0], train_images_c.shape[1], train_images_c.shape[2]))
                if test_images is None:
                    if len(test_images_c.shape) > 3:
                        test_images = numpy.zeros((test_codes.shape[0], test_images_c.shape[1], test_images_c.shape[2], test_images_c.shape[3]))
                    else:
                        test_images = numpy.zeros((test_codes.shape[0], test_images_c.shape[1], test_images_c.shape[2]))

                train_images[train_codes == label] = train_images_c
                test_images[test_codes == label] = test_images_c

            utils.write_hdf5(self.class_train_theta_file, train_theta)
            log('[Experiment] wrote %s' % self.class_train_theta_file)
            utils.write_hdf5(self.class_test_theta_file, test_theta)
            log('[Experiment] wrote %s' % self.class_test_theta_file)

            utils.write_hdf5(self.class_train_images_file, train_images)
            log('[Experiment] wrote %s' % self.class_train_images_file)
            utils.write_hdf5(self.class_test_images_file, test_images)
            log('[Experiment] wrote %s' % self.class_test_images_file)

        if not os.path.exists(self.class_sampled_images_file):
            sampled_images = None
            sampled_codes = None
            for label in range(self.labels):
                sampled_images_file_c = paths.results_file('sampled_images_%d' % label)
                if not os.path.exists(sampled_images_file_c):
                    arguments = [
                        '-decoder_file=%s' % self.decoder_files[label],
                        '-test_images_file=%s' % self.test_images_file,
                        '-images_file=%s' % sampled_images_file_c,
                        '-theta_file=',
                        '-N_samples=4000',
                        '-bound=2',
                        '-batch_size=%d' % self.batch_size,
                        '-latent_space_size=%s' % self.class_latent_space_size,
                    ] + decoder_architecture
                    sample_variational_auto_encoder = SampleVariationalAutoEncoder(arguments)
                    sample_variational_auto_encoder.main()

                sampled_images_c = utils.read_hdf5(sampled_images_file_c)
                log('[Experiment] read %s' % sampled_images_file_c)

                sampled_images = common.numpy.concatenate(sampled_images, sampled_images_c)
                sampled_codes = common.numpy.concatenate(sampled_codes, numpy.ones((sampled_images_c.shape[0], 1)) * label)

            utils.write_hdf5(self.class_sampled_images_file, sampled_images)
            log('[Experiment] wrote %s' % self.class_sampled_images_file)
            utils.write_hdf5(self.class_sampled_codes_file, sampled_codes)
            log('[Experiment] wrote %s' % self.class_sampled_codes_file)

        if self.test_theta_file is None:
            self.test_theta_file = self.class_test_theta_file
        if self.train_theta_file is None:
            self.train_theta_file = self.class_train_theta_file
        assert os.path.exists(self.train_theta_file)
        assert os.path.exists(self.test_theta_file)

    def compute_data_theta(self):
        """
        Compute theta.
        """

        decoder_architecture = []
        for option in self.decoder_parameters():
            decoder_architecture.append(option.replace('decoder', 'network'))

        assert os.path.exists(self.encoder_file), self.encoder_file
        assert os.path.exists(self.decoder_file), self.decoder_file

        self.data_train_images_file = paths.results_file('data_train_images')
        self.data_test_images_file = paths.results_file('data_test_images')
        self.data_train_theta_file = paths.results_file('data_train_theta')
        self.data_test_theta_file = paths.results_file('data_test_theta')

        if not os.path.exists(self.data_train_theta_file) or not os.path.exists(self.data_test_theta_file) \
                or not os.path.exists(self.data_train_images_file) or not os.path.exists(self.data_test_images_file):
            log_file = paths.log_file('log')
            arguments = [
                '-train_images_file=%s' % self.train_images_file,
                '-test_images_file=%s' % self.test_images_file,
                '-train_codes_file=%s' % self.train_codes_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-train_theta_file=%s' % self.data_train_theta_file,
                '-test_theta_file=%s' % self.data_test_theta_file,
                '-label_index=%d' % self.label_index,
                '-label=%d' % -1,
                '-encoder_file=%s' % self.encoder_file,
                '-decoder_file=%s' % self.decoder_file,
                '-train_reconstruction_file=%s' % self.data_train_images_file,
                '-reconstruction_file=%s' % self.data_test_images_file,
                '-random_file=',
                '-interpolation_file=',
                '-batch_size=%d' % self.batch_size,
                '-latent_space_size=%d' % self.data_latent_space_size,
                '-results_file=',
                '-output_directory=',
                '-log_file=%s' % log_file
            ] + decoder_architecture
            test_variational_auto_encoder = TestVariationalAutoEncoder(arguments)
            test_variational_auto_encoder.main()

    def test(self, model_directory):
        """
        Test.
        """

        state_file = paths.state_file('%s/classifier' % model_directory)
        accuracy_file = paths.results_file('%s/accuracy' % model_directory)
        results_file = paths.pickle_file('%s/test_results' % model_directory)
        log_file = paths.log_file('%s/test_results' % model_directory)

        if self.args.reevaluate or not os.path.exists(results_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-state_file=%s' % state_file,
                '-accuracy_file=%s' % accuracy_file,
                '-results_file=%s' % results_file,
                '-log_file=%s' % log_file
            ] + self.network_parameters()

            program = TestClassifier(arguments)
            program.main()
            test_results = program.results
        else:
            test_results = utils.read_pickle(results_file)
            log('[Experiment] read %s' % results_file)

        if self.args.evaluate_reconstructed:
            results_file = paths.pickle_file('%s/reconstructed_results' % model_directory)
            log_file = paths.log_file('%s/reconstructed_results' % model_directory)

            if self.args.reevaluate or not os.path.exists(results_file):
                arguments = [
                    '-test_images_file=%s' % self.class_test_images_file,
                    '-test_codes_file=%s' % self.test_codes_file,
                    '-label_index=%d' % self.label_index,
                    '-state_file=%s' % state_file,
                    '-accuracy_file=%s' % accuracy_file,
                    '-results_file=%s' % results_file,
                    '-log_file=%s' % log_file
                ] + self.network_parameters()

                program = TestClassifier(arguments)
                program.main()
                reconstructed_results = program.results
            else:
                reconstructed_results = utils.read_pickle(results_file)
                log('[Experiment] read %s' % results_file)
        else:
            reconstructed_results = test_results

        if self.args.evaluate_sampled:
            results_file = paths.pickle_file('%s/sampled_results' % model_directory)
            log_file = paths.log_file('%s/sampled_results' % model_directory)

            if self.args.reevaluate or not os.path.exists(results_file):
                arguments = [
                    '-test_images_file=%s' % self.class_sampled_images_file,
                    '-test_codes_file=%s' % self.class_sampled_codes_file,
                    '-label_index=0',
                    '-state_file=%s' % state_file,
                    '-accuracy_file=%s' % accuracy_file,
                    '-results_file=%s' % results_file,
                    '-log_file=%s' % log_file
                ] + self.network_parameters()

                program = TestClassifier(arguments)
                program.main()
                sampled_results = program.results
            else:
                sampled_results = utils.read_pickle(results_file)
                log('[Experiment] read %s' % results_file)
        else:
            sampled_results = test_results

        return test_results, reconstructed_results, sampled_results

    def train_normal(self, t):
        """
        Train normal model.
        """

        training = self.training_options[t]
        model_name = self.model_name('normal', training)
        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)

        state_file = paths.state_file('%s/classifier' % model_directory)
        log_file = paths.log_file('%s/log' % model_directory)
        training_file = paths.results_file('%s/training' % model_directory)
        testing_file = paths.results_file('%s/testing' % model_directory)
        loss_file = paths.image_file('%s/loss' % model_directory)
        debug_directory = paths.experiment_dir('%s/debug/' % model_directory)
        gradient_file = paths.image_file('%s/gradient' % model_directory)
        error_file = paths.image_file('%s/error' % model_directory)

        arguments = [
            '-state_file=%s' % state_file,
            '-log_file=%s' % log_file,
            '-training_file=%s' % training_file,
            '-testing_file=%s' % testing_file,
            '-loss_file=%s' % loss_file,
            '-error_file=%s' % error_file,
            '-gradient_file=%s' % gradient_file,
            #
            '-train_images_file=%s' % self.train_images_file,
            '-train_codes_file=%s' % self.train_codes_file,
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-label_index=%d' % self.label_index,
            '-validation_samples=%d' % training.validation_samples,
            '-training_samples=%d' % training.training_samples,
            '-test_samples=%d' % training.test_samples,
            '-epochs=%d' % training.epochs,
            '-early_stopping',
            # '-random_samples',
            '-batch_size=%d' % training.batch_size,
            '-weight_decay=%g' % training.weight_decay,
            # '-logit_decay',
            # '-drop_labels',
            # '-no_gpu',
            '-lr=%g' % training.lr,
            '-lr_decay=%g' % training.lr_decay,
            '-results_file=',
            '-debug_directory=%s' % debug_directory,
            #
        ] + self.network_parameters()

        if not os.path.exists(state_file):
            program = TrainClassifier(arguments)
            program.main()

        test_results, reconstructed_results, sampled_results = self.test(model_directory)

        if not model_name in self.results.keys():
            self.results[model_name] = numpy.zeros((len(self.training_options), self.args.max_models, 7))

        self.results[model_name][t, training.model, 0] = training.training_samples
        self.results[model_name][t, training.model, 1] = test_results['error']
        self.results[model_name][t, training.model, 2] = test_results['loss']
        self.results[model_name][t, training.model, 3] = reconstructed_results['error']
        self.results[model_name][t, training.model, 4] = reconstructed_results['loss']
        self.results[model_name][t, training.model, 5] = sampled_results['error']
        self.results[model_name][t, training.model, 6] = sampled_results['loss']

        return model_name

    def train_off_manifold_augmented(self, t, augmentation):
        """
        Train off-manifold data augmentation.
        """

        assert t >= 0
        assert isinstance(augmentation, OffAugmentationOptions)

        training = self.training_options[t]
        model_name = self.model_name('off_manifold_augmented', training, augmentation)
        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)

        state_file = paths.state_file('%s/classifier' % model_directory)
        log_file = paths.log_file('%s/log' % model_directory)
        training_file = paths.results_file('%s/training' % model_directory)
        testing_file = paths.results_file('%s/testing' % model_directory)
        loss_file = paths.image_file('%s/loss' % model_directory)
        debug_directory = paths.experiment_dir('%s/debug/' % model_directory)
        gradient_file = paths.image_file('%s/gradient' % model_directory)
        error_file = paths.image_file('%s/error' % model_directory)

        arguments = [
            '-state_file=%s' % state_file,
            '-log_file=%s' % log_file,
            '-training_file=%s' % training_file,
            '-testing_file=%s' % testing_file,
            '-loss_file=%s' % loss_file,
            '-error_file=%s' % error_file,
            '-gradient_file=%s' % gradient_file,
            #
            '-train_images_file=%s' % self.train_images_file,
            '-train_codes_file=%s' % self.train_codes_file,
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-label_index=%d' % self.label_index,
            '-validation_samples=%d' % training.validation_samples,
            '-training_samples=%d' % training.training_samples,
            '-test_samples=%d' % training.test_samples,
            '-epochs=%d' % training.epochs,
            '-early_stopping',
            # '-random_samples',
            '-batch_size=%d' % training.batch_size,
            '-weight_decay=%g' % training.weight_decay,
            # '-logit_decay',
            # '-drop_labels',
            # '-no_gpu',
            '-lr=%g' % training.lr,
            '-lr_decay=%g' % training.lr_decay,
            '-results_file=',
            '-debug_directory=%s' % debug_directory,
            #
            '-norm=%s' % augmentation.norm,
            '-epsilon=%g' % augmentation.epsilon,
            '-max_iterations=%d' % augmentation.max_iterations,
            # '-full_variant',
        ] + self.network_parameters()

        if augmentation.strong_variant:
            arguments += ['-strong_variant']

        if not os.path.exists(state_file):
            program = TrainClassifierAugmented(arguments)
            program.main()

        test_results, reconstructed_results, sampled_results = self.test(model_directory)

        if not model_name in self.results.keys():
            self.results[model_name] = numpy.zeros((len(self.training_options), self.args.max_models, 7))

        self.results[model_name][t, training.model, 0] = training.training_samples
        self.results[model_name][t, training.model, 1] = test_results['error']
        self.results[model_name][t, training.model, 2] = test_results['loss']
        self.results[model_name][t, training.model, 3] = reconstructed_results['error']
        self.results[model_name][t, training.model, 4] = reconstructed_results['loss']
        self.results[model_name][t, training.model, 5] = sampled_results['error']
        self.results[model_name][t, training.model, 6] = sampled_results['loss']

        return model_name

    def train_off_manifold_adversarial(self, t, attack):
        """
        Train off-manifold data augmentation.
        """

        assert t >= 0
        assert isinstance(attack, OffAttackOptions), 'class %s not instance of OffAttackOptions' % attack.__class__.__name__

        training = self.training_options[t]
        model_name = self.model_name('off_manifold_adversarial', training, attack)
        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)

        state_file = paths.state_file('%s/classifier' % model_directory)
        log_file = paths.log_file('%s/log' % model_directory)
        training_file = paths.results_file('%s/training' % model_directory)
        testing_file = paths.results_file('%s/testing' % model_directory)
        loss_file = paths.image_file('%s/loss' % model_directory)
        debug_directory = paths.experiment_dir('%s/debug/' % model_directory)
        gradient_file = paths.image_file('%s/gradient' % model_directory)
        error_file = paths.image_file('%s/error' % model_directory)
        success_file = paths.image_file('%s/success' % model_directory)

        arguments = [
            '-state_file=%s' % state_file,
            '-log_file=%s' % log_file,
            '-training_file=%s' % training_file,
            '-testing_file=%s' % testing_file,
            '-loss_file=%s' % loss_file,
            '-error_file=%s' % error_file,
            '-gradient_file=%s' % gradient_file,
            '-success_file=%s' % success_file,
            #
            '-train_images_file=%s' % self.train_images_file,
            '-train_codes_file=%s' % self.train_codes_file,
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-label_index=%d' % self.label_index,
            '-validation_samples=%d' % training.validation_samples,
            '-training_samples=%d' % training.training_samples,
            '-test_samples=%d' % training.test_samples,
            '-epochs=%d' % training.epochs,
            '-early_stopping',
            # '-random_samples',
            '-batch_size=%d' % training.batch_size,
            '-weight_decay=%g' % training.weight_decay,
            # '-logit_decay',
            # '-drop_labels',
            # '-no_gpu',
            '-lr=%g' % training.lr,
            '-lr_decay=%g' % training.lr_decay,
            '-results_file=',
            '-debug_directory=%s' % debug_directory,
            #
            '-attack=%s' % attack.attack,
            '-objective=%s' % attack.objective,
            '-epsilon=%g' % attack.epsilon,
            '-c_0=%g' % attack.c_0,
            '-c_1=%g' % attack.c_1,
            '-c_2=%g' % attack.c_2,
            '-max_iterations=%d' % attack.max_iterations,
            '-max_projections=%d' % attack.max_projections,
            '-base_lr=%g' % attack.base_lr,
        ] + self.network_parameters()

        if attack.training_mode:
            arguments += ['-training_mode']
            log('[Experiment] using training mode')

        if attack.full:
            arguments += ['-full_variant']
            log('[Experiment] using full variant')

        if not os.path.exists(state_file):
            program = TrainClassifierAdversarially(arguments)
            program.main()

        test_results, reconstructed_results, sampled_results = self.test(model_directory)

        if not model_name in self.results.keys():
            self.results[model_name] = numpy.zeros((len(self.training_options), self.args.max_models, 7))

        self.results[model_name][t, training.model, 0] = training.training_samples
        self.results[model_name][t, training.model, 1] = test_results['error']
        self.results[model_name][t, training.model, 2] = test_results['loss']
        self.results[model_name][t, training.model, 3] = reconstructed_results['error']
        self.results[model_name][t, training.model, 4] = reconstructed_results['loss']
        self.results[model_name][t, training.model, 5] = sampled_results['error']
        self.results[model_name][t, training.model, 6] = sampled_results['loss']

        return model_name

    def train_regular_augmented(self, t, augmentation):
        """
        Train with regular augmentation.
        """

        assert t >= 0
        assert isinstance(augmentation, STNAugmentationOptions)

        training = self.training_options[t]
        model_name = self.model_name('regular_augmented', training, augmentation)
        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)

        state_file = paths.state_file('%s/classifier' % model_directory)
        log_file = paths.log_file('%s/log' % model_directory)
        training_file = paths.results_file('%s/training' % model_directory)
        testing_file = paths.results_file('%s/testing' % model_directory)
        loss_file = paths.image_file('%s/loss' % model_directory)
        debug_directory = paths.experiment_dir('%s/debug/' % model_directory)
        gradient_file = paths.image_file('%s/gradient' % model_directory)
        error_file = paths.image_file('%s/error' % model_directory)

        arguments = [
            '-state_file=%s' % state_file,
            '-log_file=%s' % log_file,
            '-training_file=%s' % training_file,
            '-testing_file=%s' % testing_file,
            '-loss_file=%s' % loss_file,
            '-error_file=%s' % error_file,
            '-gradient_file=%s' % gradient_file,
            #
            '-train_images_file=%s' % self.train_images_file,
            '-train_codes_file=%s' % self.train_codes_file,
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-label_index=%d' % self.label_index,
            '-validation_samples=%d' % training.validation_samples,
            '-training_samples=%d' % training.training_samples,
            '-test_samples=%d' % training.test_samples,
            '-epochs=%d' % training.epochs,
            '-early_stopping',
            # '-random_samples',
            '-batch_size=%d' % training.batch_size,
            '-weight_decay=%g' % training.weight_decay,
            # '-logit_decay',
            # '-drop_labels',
            # '-no_gpu',
            '-lr=%g' % training.lr,
            '-lr_decay=%g' % training.lr_decay,
            '-results_file=',
            '-debug_directory=%s' % debug_directory,
            #
            '-norm=%s' % augmentation.norm,
            '-epsilon=%g' % augmentation.epsilon,
            '-max_iterations=%g' % augmentation.max_iterations,
            '-N_theta=%d' % augmentation.N_theta,
            '-translation_x=%s' % augmentation.translation_x,
            '-translation_y=%s' % augmentation.translation_y,
            '-shear_x=%s' % augmentation.shear_x,
            '-shear_y=%s' % augmentation.shear_y,
            '-scale=%s' % augmentation.scale,
            '-rotation=%s' % augmentation.rotation,
            '-color=%s' % augmentation.color,
            # '-full_variant',
        ] + self.network_parameters()

        if augmentation.strong_variant:
            arguments += ['-strong_variant']

        if not os.path.exists(state_file):
            program = TrainSTNClassifierAugmented(arguments)
            program.main()

        test_results, reconstructed_results, sampled_results = self.test(model_directory)

        if not model_name in self.results.keys():
            self.results[model_name] = numpy.zeros((len(self.training_options), self.args.max_models, 7))

        self.results[model_name][t, training.model, 0] = training.training_samples
        self.results[model_name][t, training.model, 1] = test_results['error']
        self.results[model_name][t, training.model, 2] = test_results['loss']
        self.results[model_name][t, training.model, 3] = reconstructed_results['error']
        self.results[model_name][t, training.model, 4] = reconstructed_results['loss']
        self.results[model_name][t, training.model, 5] = sampled_results['error']
        self.results[model_name][t, training.model, 6] = sampled_results['loss']

        return model_name

    def train_adversarial_augmented(self, t, attack):
        """
        Train with regular augmentation.
        """

        assert t >= 0
        assert isinstance(attack, STNAttackOptions), 'class %s not instance of STNAttackOptions' % attack.__class__.__name__

        training = self.training_options[t]
        model_name = self.model_name('adversarial_augmented', training, attack)
        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)

        state_file = paths.state_file('%s/classifier' % model_directory)
        log_file = paths.log_file('%s/log' % model_directory)
        training_file = paths.results_file('%s/training' % model_directory)
        testing_file = paths.results_file('%s/testing' % model_directory)
        loss_file = paths.image_file('%s/loss' % model_directory)
        debug_directory = paths.experiment_dir('%s/debug/' % model_directory)
        gradient_file = paths.image_file('%s/gradient' % model_directory)
        error_file = paths.image_file('%s/error' % model_directory)
        success_file = paths.image_file('%s/success' % model_directory)

        arguments = [
            '-state_file=%s' % state_file,
            '-log_file=%s' % log_file,
            '-training_file=%s' % training_file,
            '-testing_file=%s' % testing_file,
            '-loss_file=%s' % loss_file,
            '-error_file=%s' % error_file,
            '-gradient_file=%s' % gradient_file,
            '-success_file=%s' % success_file,
            #
            '-train_images_file=%s' % self.train_images_file,
            '-train_codes_file=%s' % self.train_codes_file,
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-label_index=%d' % self.label_index,
            '-validation_samples=%d' % training.validation_samples,
            '-training_samples=%d' % training.training_samples,
            '-test_samples=%d' % training.test_samples,
            '-epochs=%d' % training.epochs,
            '-early_stopping',
            # '-random_samples',
            '-batch_size=%d' % training.batch_size,
            '-weight_decay=%g' % training.weight_decay,
            # '-logit_decay',
            # '-drop_labels',
            # '-no_gpu',
            '-lr=%g' % training.lr,
            '-lr_decay=%g' % training.lr_decay,
            '-results_file=',
            '-debug_directory=%s' % debug_directory,
            #
            '-attack=%s' % attack.attack,
            '-objective=%s' % attack.objective,
            '-epsilon=%g' % attack.epsilon,
            '-c_0=%g' % attack.c_0,
            '-c_1=%g' % attack.c_1,
            '-c_2=%g' % attack.c_2,
            '-max_iterations=%d' % attack.max_iterations,
            '-max_projections=%d' % attack.max_projections,
            '-base_lr=%g' % attack.base_lr,
            '-N_theta=%d' % attack.N_theta,
            '-translation_x=%s' % attack.translation_x,
            '-translation_y=%s' % attack.translation_y,
            '-shear_x=%s' % attack.shear_x,
            '-shear_y=%s' % attack.shear_y,
            '-scale=%s' % attack.scale,
            '-rotation=%s' % attack.rotation,
            '-color=%s' % attack.color,
        ] + self.network_parameters()

        if attack.training_mode:
            arguments += ['-training_mode']
            log('[Experiment] using training mode')

        if attack.full:
            arguments += ['-full_variant']
            log('[Experiment] using full variant')

        if not os.path.exists(state_file):
            program = TrainSTNClassifierAdversarially(arguments)
            program.main()

        test_results, reconstructed_results, sampled_results = self.test(model_directory)

        if not model_name in self.results.keys():
            self.results[model_name] = numpy.zeros((len(self.training_options), self.args.max_models, 7))

        self.results[model_name][t, training.model, 0] = training.training_samples
        self.results[model_name][t, training.model, 1] = test_results['error']
        self.results[model_name][t, training.model, 2] = test_results['loss']
        self.results[model_name][t, training.model, 3] = reconstructed_results['error']
        self.results[model_name][t, training.model, 4] = reconstructed_results['loss']
        self.results[model_name][t, training.model, 5] = sampled_results['error']
        self.results[model_name][t, training.model, 6] = sampled_results['loss']

        return model_name

    def train_augmented_off_manifold_adversarial(self, t, attack, decoder_attack):
        """
        Train learned on and off manifold adversarial.
        """

        assert t >= 0
        assert isinstance(attack, OffAttackOptions), 'class %s not instance of OffAttackOptions' % attack.__class__.__name__
        assert isinstance(decoder_attack, STNAttackOptions), 'class %s not instance of STNAttackOptions' % decoder_attack.__class__.__name__

        training = self.training_options[t]
        model_name = self.model_name('train_augmented_off_manifold_adversarial', training, attack)
        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)

        state_file = paths.state_file('%s/classifier' % model_directory)
        log_file = paths.log_file('%s/log' % model_directory)
        training_file = paths.results_file('%s/training' % model_directory)
        testing_file = paths.results_file('%s/testing' % model_directory)
        loss_file = paths.image_file('%s/loss' % model_directory)
        debug_directory = paths.experiment_dir('%s/debug/' % model_directory)
        gradient_file = paths.image_file('%s/gradient' % model_directory)
        error_file = paths.image_file('%s/error' % model_directory)
        success_file = paths.image_file('%s/success' % model_directory)

        arguments = [
            '-state_file=%s' % state_file,
            '-log_file=%s' % log_file,
            '-training_file=%s' % training_file,
            '-testing_file=%s' % testing_file,
            '-loss_file=%s' % loss_file,
            '-error_file=%s' % error_file,
            '-gradient_file=%s' % gradient_file,
            '-success_file=%s' % success_file,
            #
            '-train_images_file=%s' % self.train_images_file,
            '-train_codes_file=%s' % self.train_codes_file,
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-label_index=%d' % self.label_index,
            '-training_samples=%d' % training.training_samples,
            '-validation_samples=%d' % training.validation_samples,
            '-test_samples=%d' % training.test_samples,
            '-epochs=%d' % training.epochs,
            '-early_stopping',
            # '-random_samples',
            '-batch_size=%d' % training.batch_size,
            '-weight_decay=%g' % training.weight_decay,
            '-weight_decay=%g' % training.weight_decay,
            # '-logit_decay',
            # '-drop_labels',
            # '-no_gpu',
            '-lr=%g' % training.lr,
            '-lr_decay=%g' % training.lr_decay,
            '-results_file=',
            '-debug_directory=%s' % debug_directory,
            #
            '-attack=%s' % attack.attack,
            '-objective=%s' % attack.objective,
            '-epsilon=%g' % attack.epsilon,
            '-c_0=%g' % attack.c_0,
            '-c_1=%g' % attack.c_1,
            '-c_2=%g' % attack.c_2,
            '-max_iterations=%d' % attack.max_iterations,
            '-max_projections=%d' % attack.max_projections,
            '-base_lr=%g' % attack.base_lr,
            '-decoder_epsilon=%g' % decoder_attack.epsilon,
            '-decoder_c_0=%g' % decoder_attack.c_0,
            '-decoder_c_1=%g' % decoder_attack.c_1,
            '-decoder_c_2=%g' % decoder_attack.c_2,
            '-decoder_max_iterations=%d' % decoder_attack.max_iterations,
            '-decoder_max_projections=%d' % decoder_attack.max_projections,
            '-decoder_base_lr=%g' % decoder_attack.base_lr,
            '-N_theta=%d' % decoder_attack.N_theta,
            '-translation_x=%s' % decoder_attack.translation_x,
            '-translation_y=%s' % decoder_attack.translation_y,
            '-shear_x=%s' % decoder_attack.shear_x,
            '-shear_y=%s' % decoder_attack.shear_y,
            '-scale=%s' % decoder_attack.scale,
            '-rotation=%s' % decoder_attack.rotation,
            '-color=%s' % decoder_attack.color,
        ] + self.network_parameters()

        if attack.training_mode:
            assert attack.training_mode == decoder_attack.training_mode
            arguments += ['-training_mode']
            log('[Experiment] using training mode')

        if attack.full:
            arguments += ['-full_variant']
            log('[Experiment] using full variant')

        if not os.path.exists(state_file):
            program = TrainSTNClassifierAdversarially2(arguments)
            program.main()

        test_results, reconstructed_results, sampled_results = self.test(model_directory)

        if not model_name in self.results.keys():
            self.results[model_name] = numpy.zeros((len(self.training_options), self.args.max_models, 7))

        self.results[model_name][t, training.model, 0] = training.training_samples
        self.results[model_name][t, training.model, 1] = test_results['error']
        self.results[model_name][t, training.model, 2] = test_results['loss']
        self.results[model_name][t, training.model, 3] = reconstructed_results['error']
        self.results[model_name][t, training.model, 4] = reconstructed_results['loss']
        self.results[model_name][t, training.model, 5] = sampled_results['error']
        self.results[model_name][t, training.model, 6] = sampled_results['loss']

        return model_name

    def train_on_manifold_augmented(self, t, augmentation):
        """
        Train off-manifold data augmentation.
        """

        assert t >= 0
        assert isinstance(augmentation, OnAugmentationOptions)

        training = self.training_options[t]
        model_name = self.model_name('on_manifold_augmented', training, augmentation)
        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)

        state_file = paths.state_file('%s/classifier' % model_directory)
        log_file = paths.log_file('%s/log' % model_directory)
        training_file = paths.results_file('%s/training' % model_directory)
        testing_file = paths.results_file('%s/testing' % model_directory)
        loss_file = paths.image_file('%s/loss' % model_directory)
        debug_directory = paths.experiment_dir('%s/debug/' % model_directory)
        gradient_file = paths.image_file('%s/gradient' % model_directory)
        error_file = paths.image_file('%s/error' % model_directory)

        arguments = [
            '-state_file=%s' % state_file,
            '-log_file=%s' % log_file,
            '-training_file=%s' % training_file,
            '-testing_file=%s' % testing_file,
            '-loss_file=%s' % loss_file,
            '-error_file=%s' % error_file,
            '-gradient_file=%s' % gradient_file,
            #
            '-database_file=%s' % self.database_file,
            '-train_images_file=%s' % self.train_images_file,
            '-train_codes_file=%s' % self.train_codes_file,
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-train_theta_file=%s' % self.train_theta_file,
            '-test_theta_file=%s' % self.test_theta_file,  # EMNIST FASHION FONTS
            '-label_index=%d' % self.label_index,
            '-training_samples=%d' % training.training_samples,
            '-validation_samples=%d' % training.validation_samples,
            '-test_samples=%d' % training.test_samples,
            '-epochs=%d' % training.epochs,
            '-early_stopping',
            # '-random_samples',
            '-batch_size=%d' % training.batch_size,
            '-weight_decay=%g' % training.weight_decay,
            '-weight_decay=%g' % training.weight_decay,
            # '-logit_decay',
            # '-drop_labels',
            # '-no_gpu',
            '-lr=%g' % training.lr,
            '-lr_decay=%g' % training.lr_decay,
            '-results_file=',
            '-debug_directory=%s' % debug_directory,
            #
            '-norm=%s' % augmentation.norm,
            '-epsilon=%g' % augmentation.epsilon,
            '-max_iterations=%d' % augmentation.max_iterations,
            # '-full_variant',
        ] + self.network_parameters()

        if augmentation.strong_variant:
            arguments += ['-strong_variant']

        if not os.path.exists(state_file):
            program = TrainDecoderClassifierAugmented(arguments)
            program.main()

        test_results, reconstructed_results, sampled_results = self.test(model_directory)

        if not model_name in self.results.keys():
            self.results[model_name] = numpy.zeros((len(self.training_options), self.args.max_models, 7))

        self.results[model_name][t, training.model, 0] = training.training_samples
        self.results[model_name][t, training.model, 1] = test_results['error']
        self.results[model_name][t, training.model, 2] = test_results['loss']
        self.results[model_name][t, training.model, 3] = reconstructed_results['error']
        self.results[model_name][t, training.model, 4] = reconstructed_results['loss']
        self.results[model_name][t, training.model, 5] = sampled_results['error']
        self.results[model_name][t, training.model, 6] = sampled_results['loss']

        return model_name

    def train_on_manifold_adversarial(self, t, attack):
        """
        Train off-manifold data augmentation.
        """

        assert t >= 0
        assert isinstance(attack, OnAttackOptions), 'class %s not instance of OnAttackOptions' % attack.__class__.__name__

        training = self.training_options[t]
        model_name = self.model_name('on_manifold_adversarial', training, attack)
        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)

        state_file = paths.state_file('%s/classifier' % model_directory)
        log_file = paths.log_file('%s/log' % model_directory)
        training_file = paths.results_file('%s/training' % model_directory)
        testing_file = paths.results_file('%s/testing' % model_directory)
        loss_file = paths.image_file('%s/loss' % model_directory)
        debug_directory = paths.experiment_dir('%s/debug/' % model_directory)
        gradient_file = paths.image_file('%s/gradient' % model_directory)
        error_file = paths.image_file('%s/error' % model_directory)
        success_file = paths.image_file('%s/success' % model_directory)

        arguments = [
            '-state_file=%s' % state_file,
            '-log_file=%s' % log_file,
            '-training_file=%s' % training_file,
            '-testing_file=%s' % testing_file,
            '-loss_file=%s' % loss_file,
            '-error_file=%s' % error_file,
            '-gradient_file=%s' % gradient_file,
            '-success_file=%s' % success_file,
            #
            '-database_file=%s' % self.database_file,
            '-train_images_file=%s' % self.train_images_file,
            '-train_codes_file=%s' % self.train_codes_file,
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-train_theta_file=%s' % self.train_theta_file,
            '-test_theta_file=%s' % self.test_theta_file,  # EMNIST FASHION FONTS
            '-label_index=%d' % self.label_index,
            '-training_samples=%d' % training.training_samples,
            '-validation_samples=%d' % training.validation_samples,
            '-test_samples=%d' % training.test_samples,
            '-epochs=%d' % training.epochs,
            '-early_stopping',
            # '-random_samples',
            '-batch_size=%d' % training.batch_size,
            '-weight_decay=%g' % training.weight_decay,
            '-weight_decay=%g' % training.weight_decay,
            # '-logit_decay',
            # '-drop_labels',
            # '-no_gpu',
            '-lr=%g' % training.lr,
            '-lr_decay=%g' % training.lr_decay,
            '-results_file=',
            '-debug_directory=%s' % debug_directory,
            #
            '-attack=%s' % attack.attack,
            '-objective=%s' % attack.objective,
            '-epsilon=%g' % attack.epsilon,
            '-c_0=%g' % attack.c_0,
            '-c_1=%g' % attack.c_1,
            '-c_2=%g' % attack.c_2,
            '-max_iterations=%d' % attack.max_iterations,
            '-max_projections=%d' % attack.max_projections,
            '-base_lr=%g' % attack.base_lr,
        ] + self.network_parameters()

        if attack.training_mode:
            arguments += ['-training_mode']
            log('[Experiment] using training mode')

        if attack.full:
            arguments += ['-full_variant']
            log('[Experiment] using full variant')

        if not os.path.exists(state_file):
            program = TrainDecoderClassifierAdversarially(arguments)
            program.main()

        test_results, reconstructed_results, sampled_results = self.test(model_directory)

        if not model_name in self.results.keys():
            self.results[model_name] = numpy.zeros((len(self.training_options), self.args.max_models, 7))

        self.results[model_name][t, training.model, 0] = training.training_samples
        self.results[model_name][t, training.model, 1] = test_results['error']
        self.results[model_name][t, training.model, 2] = test_results['loss']
        self.results[model_name][t, training.model, 3] = reconstructed_results['error']
        self.results[model_name][t, training.model, 4] = reconstructed_results['loss']
        self.results[model_name][t, training.model, 5] = sampled_results['error']
        self.results[model_name][t, training.model, 6] = sampled_results['loss']

        return model_name

    def train_on_off_manifold_adversarial(self, t, attack, decoder_attack):
        """
        Train off-manifold data augmentation.
        """

        assert t >= 0
        assert isinstance(attack, OffAttackOptions), 'class %s not instance of OnAttackOptions' % attack.__class__.__name__
        assert isinstance(decoder_attack, OnAttackOptions), 'class %s not instance of OnAttackOptions' % decoder_attack.__class__.__name__

        training = self.training_options[t]
        model_name = self.model_name('on_off_manifold_adversarial', training, attack, decoder_attack)
        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)

        state_file = paths.state_file('%s/classifier' % model_directory)
        log_file = paths.log_file('%s/log' % model_directory)
        training_file = paths.results_file('%s/training' % model_directory)
        testing_file = paths.results_file('%s/testing' % model_directory)
        loss_file = paths.image_file('%s/loss' % model_directory)
        debug_directory = paths.experiment_dir('%s/debug/' % model_directory)
        gradient_file = paths.image_file('%s/gradient' % model_directory)
        error_file = paths.image_file('%s/error' % model_directory)
        success_file = paths.image_file('%s/success' % model_directory)

        arguments = [
            '-state_file=%s' % state_file,
            '-log_file=%s' % log_file,
            '-training_file=%s' % training_file,
            '-testing_file=%s' % testing_file,
            '-loss_file=%s' % loss_file,
            '-error_file=%s' % error_file,
            '-gradient_file=%s' % gradient_file,
            '-success_file=%s' % success_file,
            #
            '-database_file=%s' % self.database_file,
            '-train_images_file=%s' % self.train_images_file,
            '-train_codes_file=%s' % self.train_codes_file,
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-train_theta_file=%s' % self.train_theta_file,
            '-test_theta_file=%s' % self.test_theta_file,  # EMNIST FASHION FONTS
            '-label_index=%d' % self.label_index,
            '-training_samples=%d' % training.training_samples,
            '-validation_samples=%d' % training.validation_samples,
            '-test_samples=%d' % training.test_samples,
            '-epochs=%d' % training.epochs,
            '-early_stopping',
            # '-random_samples',
            '-batch_size=%d' % training.batch_size,
            '-weight_decay=%g' % training.weight_decay,
            '-weight_decay=%g' % training.weight_decay,
            # '-logit_decay',
            # '-drop_labels',
            # '-no_gpu',
            '-lr=%g' % training.lr,
            '-lr_decay=%g' % training.lr_decay,
            '-results_file=',
            '-debug_directory=%s' % debug_directory,
            #
            '-attack=%s' % attack.attack,
            '-objective=%s' % attack.objective,
            '-epsilon=%g' % attack.epsilon,
            '-c_0=%g' % attack.c_0,
            '-c_1=%g' % attack.c_1,
            '-c_2=%g' % attack.c_2,
            '-max_iterations=%d' % attack.max_iterations,
            '-max_projections=%d' % attack.max_projections,
            '-base_lr=%g' % attack.base_lr,
            '-decoder_epsilon=%g' % decoder_attack.epsilon,
            '-decoder_c_0=%g' % decoder_attack.c_0,
            '-decoder_c_1=%g' % decoder_attack.c_1,
            '-decoder_c_2=%g' % decoder_attack.c_2,
            '-decoder_max_iterations=%d' % decoder_attack.max_iterations,
            '-decoder_max_projections=%d' % decoder_attack.max_projections,
            '-decoder_base_lr=%g' % decoder_attack.base_lr,
        ] + self.network_parameters()

        if attack.training_mode:
            assert attack.training_mode == decoder_attack.training_mode
            arguments += ['-training_mode']
            log('[Experiment] using training mode')

        if attack.full:
            arguments += ['-full_variant']
            log('[Experiment] using full variant')

        if not os.path.exists(state_file):
            program = TrainDecoderClassifierAdversarially2(arguments)
            program.main()

        test_results, reconstructed_results, sampled_results = self.test(model_directory)

        if not model_name in self.results.keys():
            self.results[model_name] = numpy.zeros((len(self.training_options), self.args.max_models, 7))

        self.results[model_name][t, training.model, 0] = training.training_samples
        self.results[model_name][t, training.model, 1] = test_results['error']
        self.results[model_name][t, training.model, 2] = test_results['loss']
        self.results[model_name][t, training.model, 3] = reconstructed_results['error']
        self.results[model_name][t, training.model, 4] = reconstructed_results['loss']
        self.results[model_name][t, training.model, 5] = sampled_results['error']
        self.results[model_name][t, training.model, 6] = sampled_results['loss']

        return model_name

    def train_learned_on_class_manifold_augmented(self, t, augmentation):
        """
        Train on class manifold augmented.
        """

        assert t >= 0
        assert isinstance(augmentation, LearnedOnClassAugmentationOptions)

        training = self.training_options[t]
        model_name = self.model_name('learned_on_class_manifold_augmented', training, augmentation)
        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)

        state_file = paths.state_file('%s/classifier' % model_directory)
        log_file = paths.log_file('%s/log' % model_directory)
        training_file = paths.results_file('%s/training' % model_directory)
        testing_file = paths.results_file('%s/testing' % model_directory)
        loss_file = paths.image_file('%s/loss' % model_directory)
        debug_directory = paths.experiment_dir('%s/debug/' % model_directory)
        gradient_file = paths.image_file('%s/gradient' % model_directory)
        error_file = paths.image_file('%s/error' % model_directory)

        arguments = [
            '-state_file=%s' % state_file,
            '-log_file=%s' % log_file,
            '-training_file=%s' % training_file,
            '-testing_file=%s' % testing_file,
            '-loss_file=%s' % loss_file,
            '-error_file=%s' % error_file,
            '-gradient_file=%s' % gradient_file,
            #
            '-train_images_file=%s' % self.train_images_file,
            '-train_codes_file=%s' % self.train_codes_file,
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-train_theta_file=%s' % self.class_train_theta_file,
            '-test_theta_file=%s' % self.class_test_theta_file,
            '-decoder_files=%s' % ','.join(self.decoder_files),
            '-latent_space_size=%d' % self.class_latent_space_size,
            '-label_index=%d' % self.label_index,
            '-training_samples=%d' % training.training_samples,
            '-validation_samples=%d' % training.validation_samples,
            '-test_samples=%d' % training.test_samples,
            '-epochs=%d' % training.epochs,
            '-early_stopping',
            # '-random_samples',
            '-batch_size=%d' % training.batch_size,
            '-weight_decay=%g' % training.weight_decay,
            '-weight_decay=%g' % training.weight_decay,
            # '-logit_decay',
            # '-drop_labels',
            # '-no_gpu',
            '-lr=%g' % training.lr,
            '-lr_decay=%g' % training.lr_decay,
            '-results_file=',
            '-debug_directory=%s' % debug_directory,
            #
            '-norm=%s' % augmentation.norm,
            '-epsilon=%g' % augmentation.epsilon,
            '-max_iterations=%d' % augmentation.max_iterations,
            '-bound=%g' % augmentation.bound,
            # '-full_variant',
        ] + self.network_parameters() + self.decoder_parameters()

        if augmentation.strong_variant:
            arguments += ['-strong_variant']

        if not os.path.exists(state_file):
            program = TrainLearnedDecoderClassifierAugmented(arguments)
            program.main()

        test_results, reconstructed_results, sampled_results = self.test(model_directory)

        if not model_name in self.results.keys():
            self.results[model_name] = numpy.zeros((len(self.training_options), self.args.max_models, 7))

        self.results[model_name][t, training.model, 0] = training.training_samples
        self.results[model_name][t, training.model, 1] = test_results['error']
        self.results[model_name][t, training.model, 2] = test_results['loss']
        self.results[model_name][t, training.model, 3] = reconstructed_results['error']
        self.results[model_name][t, training.model, 4] = reconstructed_results['loss']
        self.results[model_name][t, training.model, 5] = sampled_results['error']
        self.results[model_name][t, training.model, 6] = sampled_results['loss']

        return model_name

    def train_learned_on_class_manifold_adversarial(self, t, attack):
        """
        Train on class manifold adversarial.
        """

        assert t >= 0
        assert isinstance(attack, LearnedOnClassAttackOptions), 'class %s not instance of LearnedOnClassAttackOptions' % attack.__class__.__name__

        training = self.training_options[t]
        model_name = self.model_name('learned_on_class_manifold_adversarial', training, attack)
        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)

        state_file = paths.state_file('%s/classifier' % model_directory)
        log_file = paths.log_file('%s/log' % model_directory)
        training_file = paths.results_file('%s/training' % model_directory)
        testing_file = paths.results_file('%s/testing' % model_directory)
        loss_file = paths.image_file('%s/loss' % model_directory)
        debug_directory = paths.experiment_dir('%s/debug/' % model_directory)
        gradient_file = paths.image_file('%s/gradient' % model_directory)
        error_file = paths.image_file('%s/error' % model_directory)
        success_file = paths.image_file('%s/success' % model_directory)

        arguments = [
            '-state_file=%s' % state_file,
            '-log_file=%s' % log_file,
            '-training_file=%s' % training_file,
            '-testing_file=%s' % testing_file,
            '-loss_file=%s' % loss_file,
            '-error_file=%s' % error_file,
            '-gradient_file=%s' % gradient_file,
            '-success_file=%s' % success_file,
            #
            '-train_images_file=%s' % self.train_images_file,
            '-train_codes_file=%s' % self.train_codes_file,
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-train_theta_file=%s' % self.class_train_theta_file,
            '-test_theta_file=%s' % self.class_test_theta_file,
            '-decoder_files=%s' % ','.join(self.decoder_files),
            '-latent_space_size=%d' % self.class_latent_space_size,
            '-label_index=%d' % self.label_index,
            '-training_samples=%d' % training.training_samples,
            '-validation_samples=%d' % training.validation_samples,
            '-test_samples=%d' % training.test_samples,
            '-epochs=%d' % training.epochs,
            '-early_stopping',
            # '-random_samples',
            '-batch_size=%d' % training.batch_size,
            '-weight_decay=%g' % training.weight_decay,
            '-weight_decay=%g' % training.weight_decay,
            # '-logit_decay',
            # '-drop_labels',
            # '-no_gpu',
            '-lr=%g' % training.lr,
            '-lr_decay=%g' % training.lr_decay,
            '-results_file=',
            '-debug_directory=%s' % debug_directory,
            #
            '-attack=%s' % attack.attack,
            '-objective=%s' % attack.objective,
            '-epsilon=%g' % attack.epsilon,
            '-c_0=%g' % attack.c_0,
            '-c_1=%g' % attack.c_1,
            '-c_2=%g' % attack.c_2,
            '-max_iterations=%d' % attack.max_iterations,
            '-max_projections=%d' % attack.max_projections,
            '-base_lr=%g' % attack.base_lr,
            '-bound=%g' % attack.bound,
            '-safe'
        ] + self.network_parameters() + self.decoder_parameters()

        if attack.training_mode:
            arguments += ['-training_mode']
            log('[Experiment] using training mode')

        if attack.full:
            arguments += ['-full_variant']
            log('[Experiment] using full variant')

        if not os.path.exists(state_file):
            program = TrainLearnedDecoderClassifierAdversarially(arguments)
            program.main()

        test_results, reconstructed_results, sampled_results = self.test(model_directory)

        if not model_name in self.results.keys():
            self.results[model_name] = numpy.zeros((len(self.training_options), self.args.max_models, 7))

        self.results[model_name][t, training.model, 0] = training.training_samples
        self.results[model_name][t, training.model, 1] = test_results['error']
        self.results[model_name][t, training.model, 2] = test_results['loss']
        self.results[model_name][t, training.model, 3] = reconstructed_results['error']
        self.results[model_name][t, training.model, 4] = reconstructed_results['loss']
        self.results[model_name][t, training.model, 5] = sampled_results['error']
        self.results[model_name][t, training.model, 6] = sampled_results['loss']

        return model_name

    def train_learned_on_off_class_manifold_adversarial(self, t, attack, decoder_attack):
        """
        Train learned on and off manifold adversarial.
        """

        assert t >= 0
        assert isinstance(attack, OffAttackOptions), 'class %s not instance of OffAttackOptions' % attack.__class__.__name__
        assert isinstance(decoder_attack, LearnedOnClassAttackOptions), 'class %s not instance of LearnedOnClassAttackOptions' % decoder_attack.__class__.__name__

        training = self.training_options[t]
        model_name = self.model_name('learned_on_off_class_manifold_adversarial', training, attack)
        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)

        state_file = paths.state_file('%s/classifier' % model_directory)
        log_file = paths.log_file('%s/log' % model_directory)
        training_file = paths.results_file('%s/training' % model_directory)
        testing_file = paths.results_file('%s/testing' % model_directory)
        loss_file = paths.image_file('%s/loss' % model_directory)
        debug_directory = paths.experiment_dir('%s/debug/' % model_directory)
        gradient_file = paths.image_file('%s/gradient' % model_directory)
        error_file = paths.image_file('%s/error' % model_directory)
        success_file = paths.image_file('%s/success' % model_directory)

        arguments = [
            '-state_file=%s' % state_file,
            '-log_file=%s' % log_file,
            '-training_file=%s' % training_file,
            '-testing_file=%s' % testing_file,
            '-loss_file=%s' % loss_file,
            '-error_file=%s' % error_file,
            '-gradient_file=%s' % gradient_file,
            '-success_file=%s' % success_file,
            #
            '-train_images_file=%s' % self.train_images_file,
            '-train_codes_file=%s' % self.train_codes_file,
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-train_theta_file=%s' % self.class_train_theta_file,
            '-test_theta_file=%s' % self.class_test_theta_file,
            '-decoder_files=%s' % ','.join(self.decoder_files),
            '-latent_space_size=%d' % self.class_latent_space_size,
            '-label_index=%d' % self.label_index,
            '-training_samples=%d' % training.training_samples,
            '-validation_samples=%d' % training.validation_samples,
            '-test_samples=%d' % training.test_samples,
            '-epochs=%d' % training.epochs,
            '-early_stopping',
            # '-random_samples',
            '-batch_size=%d' % training.batch_size,
            '-weight_decay=%g' % training.weight_decay,
            '-weight_decay=%g' % training.weight_decay,
            # '-logit_decay',
            # '-drop_labels',
            # '-no_gpu',
            '-lr=%g' % training.lr,
            '-lr_decay=%g' % training.lr_decay,
            '-results_file=',
            '-debug_directory=%s' % debug_directory,
            #
            '-attack=%s' % attack.attack,
            '-objective=%s' % attack.objective,
            '-epsilon=%g' % attack.epsilon,
            '-c_0=%g' % attack.c_0,
            '-c_1=%g' % attack.c_1,
            '-c_2=%g' % attack.c_2,
            '-max_iterations=%d' % attack.max_iterations,
            '-max_projections=%d' % attack.max_projections,
            '-base_lr=%g' % attack.base_lr,
            '-bound=%g' % decoder_attack.bound,
            '-decoder_epsilon=%g' % decoder_attack.epsilon,
            '-decoder_c_0=%g' % decoder_attack.c_0,
            '-decoder_c_1=%g' % decoder_attack.c_1,
            '-decoder_c_2=%g' % decoder_attack.c_2,
            '-decoder_max_iterations=%d' % decoder_attack.max_iterations,
            '-decoder_max_projections=%d' % decoder_attack.max_projections,
            '-decoder_base_lr=%g' % decoder_attack.base_lr,
            '-safe'
        ] + self.network_parameters() + self.decoder_parameters()

        if attack.training_mode:
            assert attack.training_mode == decoder_attack.training_mode
            arguments += ['-training_mode']
            log('[Experiment] using training mode')

        if attack.full:
            arguments += ['-full_variant']
            log('[Experiment] using full variant')

        if not os.path.exists(state_file):
            program = TrainLearnedDecoderClassifierAdversarially2(arguments)
            program.main()

        test_results, reconstructed_results, sampled_results = self.test(model_directory)

        if not model_name in self.results.keys():
            self.results[model_name] = numpy.zeros((len(self.training_options), self.args.max_models, 7))

        self.results[model_name][t, training.model, 0] = training.training_samples
        self.results[model_name][t, training.model, 1] = test_results['error']
        self.results[model_name][t, training.model, 2] = test_results['loss']
        self.results[model_name][t, training.model, 3] = reconstructed_results['error']
        self.results[model_name][t, training.model, 4] = reconstructed_results['loss']
        self.results[model_name][t, training.model, 5] = sampled_results['error']
        self.results[model_name][t, training.model, 6] = sampled_results['loss']

        return model_name

    def train_learned_on_data_manifold_augmented(self, t, augmentation):
        """
        Train on class manifold augmented.
        """

        assert t >= 0
        assert isinstance(augmentation, LearnedOnDataAugmentationOptions)

        training = self.training_options[t]
        model_name = self.model_name('learned_on_data_manifold_augmented', training, augmentation)
        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)

        state_file = paths.state_file('%s/classifier' % model_directory)
        log_file = paths.log_file('%s/log' % model_directory)
        training_file = paths.results_file('%s/training' % model_directory)
        testing_file = paths.results_file('%s/testing' % model_directory)
        loss_file = paths.image_file('%s/loss' % model_directory)
        debug_directory = paths.experiment_dir('%s/debug/' % model_directory)
        gradient_file = paths.image_file('%s/gradient' % model_directory)
        error_file = paths.image_file('%s/error' % model_directory)

        arguments = [
            '-state_file=%s' % state_file,
            '-log_file=%s' % log_file,
            '-training_file=%s' % training_file,
            '-testing_file=%s' % testing_file,
            '-loss_file=%s' % loss_file,
            '-error_file=%s' % error_file,
            '-gradient_file=%s' % gradient_file,
            #
            '-train_images_file=%s' % self.train_images_file,
            '-train_codes_file=%s' % self.train_codes_file,
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-train_theta_file=%s' % self.data_train_theta_file,
            '-test_theta_file=%s' % self.data_test_theta_file,
            '-decoder_files=%s' % self.decoder_file,
            '-latent_space_size=%d' % self.data_latent_space_size,
            '-label_index=%d' % self.label_index,
            '-training_samples=%d' % training.training_samples,
            '-validation_samples=%d' % training.validation_samples,
            '-test_samples=%d' % training.test_samples,
            '-epochs=%d' % training.epochs,
            '-early_stopping',
            # '-random_samples',
            '-batch_size=%d' % training.batch_size,
            '-weight_decay=%g' % training.weight_decay,
            '-weight_decay=%g' % training.weight_decay,
            # '-logit_decay',
            # '-drop_labels',
            # '-no_gpu',
            '-lr=%g' % training.lr,
            '-lr_decay=%g' % training.lr_decay,
            '-results_file=',
            '-debug_directory=%s' % debug_directory,
            #
            '-norm=%s' % augmentation.norm,
            '-epsilon=%g' % augmentation.epsilon,
            '-max_iterations=%d' % augmentation.max_iterations,
            '-bound=%g' % augmentation.bound,
            # '-full_variant',
        ] + self.network_parameters() + self.decoder_parameters()

        if augmentation.strong_variant:
            arguments += ['-strong_variant']

        if not os.path.exists(state_file):
            program = TrainLearnedDecoderClassifierAugmented(arguments)
            program.main()

        test_results, reconstructed_results, sampled_results = self.test(model_directory)

        if not model_name in self.results.keys():
            self.results[model_name] = numpy.zeros((len(self.training_options), self.args.max_models, 7))

        self.results[model_name][t, training.model, 0] = training.training_samples
        self.results[model_name][t, training.model, 1] = test_results['error']
        self.results[model_name][t, training.model, 2] = test_results['loss']
        self.results[model_name][t, training.model, 3] = reconstructed_results['error']
        self.results[model_name][t, training.model, 4] = reconstructed_results['loss']
        self.results[model_name][t, training.model, 5] = sampled_results['error']
        self.results[model_name][t, training.model, 6] = sampled_results['loss']

        return model_name

    def train_learned_on_data_manifold_adversarial(self, t, attack):
        """
        Train on class manifold adversarial.
        """

        assert t >= 0
        assert isinstance(attack, LearnedOnDataAttackOptions), 'class %s not instance of LearnedOnDataAttackOptions' % attack.__class__.__name__

        training = self.training_options[t]
        model_name = self.model_name('learned_on_data_manifold_adversarial', training, attack)
        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)

        state_file = paths.state_file('%s/classifier' % model_directory)
        log_file = paths.log_file('%s/log' % model_directory)
        training_file = paths.results_file('%s/training' % model_directory)
        testing_file = paths.results_file('%s/testing' % model_directory)
        loss_file = paths.image_file('%s/loss' % model_directory)
        debug_directory = paths.experiment_dir('%s/debug/' % model_directory)
        gradient_file = paths.image_file('%s/gradient' % model_directory)
        error_file = paths.image_file('%s/error' % model_directory)
        success_file = paths.image_file('%s/success' % model_directory)

        arguments = [
            '-state_file=%s' % state_file,
            '-log_file=%s' % log_file,
            '-training_file=%s' % training_file,
            '-testing_file=%s' % testing_file,
            '-loss_file=%s' % loss_file,
            '-error_file=%s' % error_file,
            '-gradient_file=%s' % gradient_file,
            '-success_file=%s' % success_file,
            #
            '-train_images_file=%s' % self.train_images_file,
            '-train_codes_file=%s' % self.train_codes_file,
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-train_theta_file=%s' % self.data_train_theta_file,
            '-test_theta_file=%s' % self.data_test_theta_file,
            '-decoder_files=%s' % self.decoder_file,
            '-latent_space_size=%d' % self.data_latent_space_size,
            '-label_index=%d' % self.label_index,
            '-training_samples=%d' % training.training_samples,
            '-validation_samples=%d' % training.validation_samples,
            '-test_samples=%d' % training.test_samples,
            '-epochs=%d' % training.epochs,
            '-early_stopping',
            # '-random_samples',
            '-batch_size=%d' % training.batch_size,
            '-weight_decay=%g' % training.weight_decay,
            '-weight_decay=%g' % training.weight_decay,
            # '-logit_decay',
            # '-drop_labels',
            # '-no_gpu',
            '-lr=%g' % training.lr,
            '-lr_decay=%g' % training.lr_decay,
            '-results_file=',
            '-debug_directory=%s' % debug_directory,
            #
            '-attack=%s' % attack.attack,
            '-objective=%s' % attack.objective,
            '-epsilon=%g' % attack.epsilon,
            '-c_0=%g' % attack.c_0,
            '-c_1=%g' % attack.c_1,
            '-c_2=%g' % attack.c_2,
            '-max_iterations=%d' % attack.max_iterations,
            '-max_projections=%d' % attack.max_projections,
            '-base_lr=%g' % attack.base_lr,
            '-bound=%g' % attack.bound,
            '-safe'
        ] + self.network_parameters() + self.decoder_parameters()

        if attack.training_mode:
            arguments += ['-training_mode']
            log('[Experiment] using training mode')

        if attack.full:
            arguments += ['-full_variant']
            log('[Experiment] using full variant')

        if not os.path.exists(state_file):
            program = TrainLearnedDecoderClassifierAdversarially(arguments)
            program.main()

        test_results, reconstructed_results, sampled_results = self.test(model_directory)

        if not model_name in self.results.keys():
            self.results[model_name] = numpy.zeros((len(self.training_options), self.args.max_models, 7))

        self.results[model_name][t, training.model, 0] = training.training_samples
        self.results[model_name][t, training.model, 1] = test_results['error']
        self.results[model_name][t, training.model, 2] = test_results['loss']
        self.results[model_name][t, training.model, 3] = reconstructed_results['error']
        self.results[model_name][t, training.model, 4] = reconstructed_results['loss']
        self.results[model_name][t, training.model, 5] = sampled_results['error']
        self.results[model_name][t, training.model, 6] = sampled_results['loss']

        return model_name

    def train_learned_on_off_data_manifold_adversarial(self, t, attack, decoder_attack):
        """
        Train learned on and off manifold adversarial.
        """

        assert t >= 0
        assert isinstance(attack, OffAttackOptions), 'class %s not instance of OffAttackOptions' % attack.__class__.__name__
        assert isinstance(decoder_attack, LearnedOnDataAttackOptions), 'class %s not instance of LearnedOnDataAttackOptions' % decoder_attack.__class__.__name__

        training = self.training_options[t]
        model_name = self.model_name('learned_on_off_data_manifold_adversarial', training, attack)
        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)

        state_file = paths.state_file('%s/classifier' % model_directory)
        log_file = paths.log_file('%s/log' % model_directory)
        training_file = paths.results_file('%s/training' % model_directory)
        testing_file = paths.results_file('%s/testing' % model_directory)
        loss_file = paths.image_file('%s/loss' % model_directory)
        debug_directory = paths.experiment_dir('%s/debug/' % model_directory)
        gradient_file = paths.image_file('%s/gradient' % model_directory)
        error_file = paths.image_file('%s/error' % model_directory)
        success_file = paths.image_file('%s/success' % model_directory)

        arguments = [
            '-state_file=%s' % state_file,
            '-log_file=%s' % log_file,
            '-training_file=%s' % training_file,
            '-testing_file=%s' % testing_file,
            '-loss_file=%s' % loss_file,
            '-error_file=%s' % error_file,
            '-gradient_file=%s' % gradient_file,
            '-success_file=%s' % success_file,
            #
            '-train_images_file=%s' % self.train_images_file,
            '-train_codes_file=%s' % self.train_codes_file,
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-train_theta_file=%s' % self.data_train_theta_file,
            '-test_theta_file=%s' % self.data_test_theta_file,
            '-decoder_files=%s' % self.decoder_file,
            '-latent_space_size=%d' % self.data_latent_space_size,
            '-label_index=%d' % self.label_index,
            '-training_samples=%d' % training.training_samples,
            '-validation_samples=%d' % training.validation_samples,
            '-test_samples=%d' % training.test_samples,
            '-epochs=%d' % training.epochs,
            '-early_stopping',
            # '-random_samples',
            '-batch_size=%d' % training.batch_size,
            '-weight_decay=%g' % training.weight_decay,
            '-weight_decay=%g' % training.weight_decay,
            # '-logit_decay',
            # '-drop_labels',
            # '-no_gpu',
            '-lr=%g' % training.lr,
            '-lr_decay=%g' % training.lr_decay,
            '-results_file=',
            '-debug_directory=%s' % debug_directory,
            #
            '-attack=%s' % attack.attack,
            '-objective=%s' % attack.objective,
            '-epsilon=%g' % attack.epsilon,
            '-c_0=%g' % attack.c_0,
            '-c_1=%g' % attack.c_1,
            '-c_2=%g' % attack.c_2,
            '-max_iterations=%d' % attack.max_iterations,
            '-max_projections=%d' % attack.max_projections,
            '-base_lr=%g' % attack.base_lr,
            '-bound=%g' % decoder_attack.bound,
            '-decoder_epsilon=%g' % decoder_attack.epsilon,
            '-decoder_c_0=%g' % decoder_attack.c_0,
            '-decoder_c_1=%g' % decoder_attack.c_1,
            '-decoder_c_2=%g' % decoder_attack.c_2,
            '-decoder_max_iterations=%d' % decoder_attack.max_iterations,
            '-decoder_max_projections=%d' % decoder_attack.max_projections,
            '-decoder_base_lr=%g' % decoder_attack.base_lr,
            '-safe'
        ] + self.network_parameters() + self.decoder_parameters()

        if attack.training_mode:
            assert attack.training_mode == decoder_attack.training_mode
            arguments += ['-training_mode']
            log('[Experiment] using training mode')

        if attack.full:
            arguments += ['-full_variant']
            log('[Experiment] using full variant')

        if not os.path.exists(state_file):
            program = TrainLearnedDecoderClassifierAdversarially2(arguments)
            program.main()

        test_results, reconstructed_results, sampled_results = self.test(model_directory)

        if not model_name in self.results.keys():
            self.results[model_name] = numpy.zeros((len(self.training_options), self.args.max_models, 7))

        self.results[model_name][t, training.model, 0] = training.training_samples
        self.results[model_name][t, training.model, 1] = test_results['error']
        self.results[model_name][t, training.model, 2] = test_results['loss']
        self.results[model_name][t, training.model, 3] = reconstructed_results['error']
        self.results[model_name][t, training.model, 4] = reconstructed_results['loss']
        self.results[model_name][t, training.model, 5] = sampled_results['error']
        self.results[model_name][t, training.model, 6] = sampled_results['loss']

        return model_name

    def attack_off_manifold(self, model_name, t, a):
        """
        Attack off manifold.
        """

        assert self.off_attack_options is not None
        assert a >= 0 and a < len(self.off_attack_options)

        training = self.training_options[t]
        attack = self.off_attack_options[a]
        assert isinstance(attack, OffAttackOptions), 'class %s not instance of OffAttackOptions' % attack.__class__.__name__

        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)
        base_directory = self.attack_directory(model_directory, attack)
        assert os.path.exists(paths.experiment_dir(model_directory)), model_directory

        classifier_file = paths.experiment_file('%s/classifier' % model_directory, ext='.pth.tar')
        perturbations_file = paths.results_file('%s/perturbations' % base_directory)
        success_file = paths.results_file('%s/success' % base_directory)
        accuracy_file = paths.results_file('%s/accuracy' % base_directory)
        log_file = paths.log_file('%s/log' % base_directory)

        results_file = paths.pickle_file('%s/success' % base_directory)
        plot_directory = paths.experiment_dir('%s/' % base_directory)

        if self.args.reattack:
            if os.path.exists(perturbations_file):
                os.unlink(perturbations_file)
            if os.path.exists(success_file):
                os.unlink(success_file)
            if os.path.exists(accuracy_file):
                os.unlink(accuracy_file)
            if os.path.exists(results_file):
                os.unlink(results_file)

        arguments = [
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-label_index=%d' % self.label_index,
            '-classifier_file=%s' % classifier_file,
            '-accuracy_file=%s' % accuracy_file,
            '-perturbations_file=%s' % perturbations_file,
            '-success_file=%s' % success_file,
            '-log_file=%s' % log_file,
            '-attack=%s' % attack.attack,
            '-objective=%s' % attack.objective,
            '-max_attempts=%d' % attack.max_attempts,
            '-max_samples=%d' % attack.max_samples,
            '-batch_size=%d' % self.batch_size,
            '-epsilon=%g' % attack.epsilon,
            '-c_0=%g' % attack.c_0,
            '-c_1=%g' % attack.c_1,
            '-c_2=%g' % attack.c_2,
            '-max_iterations=%d' % attack.max_iterations,
            '-max_projections=%d' % attack.max_projections,
            '-base_lr=%g' % attack.base_lr,
            # '-no_gpu',
            # '-no_label_leaking',
        ] + self.network_parameters()

        if not os.path.exists(perturbations_file) or not os.path.exists(success_file):
            attack_classifier = AttackClassifier(arguments)
            attack_classifier.main()

        if self.args.reattack or self.args.reevaluate or not os.path.exists(results_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-train_images_file=%s' % self.train_images_file,
                '-test_theta_file=%s' % self.test_theta_file,  # EMNIST FASHION FONTS
                '-train_theta_file=%s' % self.train_theta_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-accuracy_file=%s' % accuracy_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-results_file=%s' % results_file,
                '-plot_directory=%s' % plot_directory,
                # '-plot_manifolds'
            ]

            test_attack_classifier = TestAttackClassifier(arguments)
            test_attack_classifier.main()
            results = test_attack_classifier.results
        else:
            results = utils.read_pickle(results_file)
            log('[Experiment] read %s' % results_file)

        key = model_name + '_off'
        if not key in self.results.keys():
            self.results[key] = numpy.zeros((len(self.training_options), self.args.max_models, len(self.off_attack_options), 3, 4))

        for n in range(3):
            self.results[key][t, training.model, a, n, 0] = results[n]['raw_success']
            self.results[key][t, training.model, a, n, 1] = results[n]['raw_iteration']
            self.results[key][t, training.model, a, n, 2] = results[n]['raw_average']
            self.results[key][t, training.model, a, n, 3] = results[n]['raw_latent']

    def attack_on_manifold(self, model_name, t, a):
        """
        Attack on manifold.
        """

        assert self.on_attack_options is not None
        assert a >= 0 and a < len(self.on_attack_options)

        training = self.training_options[t]
        attack = self.on_attack_options[a]
        assert isinstance(attack, OnAttackOptions), 'class %s not instance of OnAttackOptions' % attack.__class__.__name__

        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)
        base_directory = self.attack_directory(model_directory, attack)
        assert os.path.exists(paths.experiment_dir(model_directory)), model_directory

        classifier_file = paths.experiment_file('%s/classifier' % model_directory, ext='.pth.tar')
        perturbations_file = paths.results_file('%s/perturbations' % base_directory)
        success_file = paths.results_file('%s/success' % base_directory)
        accuracy_file = paths.results_file('%s/accuracy' % base_directory)
        log_file = paths.log_file('%s/log' % base_directory)

        results_file = paths.pickle_file('%s/success' % base_directory)
        plot_directory = paths.experiment_dir('%s/' % base_directory)

        if self.args.reattack:
            if os.path.exists(perturbations_file):
                os.unlink(perturbations_file)
            if os.path.exists(success_file):
                os.unlink(success_file)
            if os.path.exists(accuracy_file):
                os.unlink(accuracy_file)
            if os.path.exists(results_file):
                os.unlink(results_file)

        arguments = [
            '-database_file=%s' % self.database_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-test_theta_file=%s' % self.test_theta_file,  # EMNIST FASHION FONTS
            '-classifier_file=%s' % classifier_file,
            '-accuracy_file=%s' % accuracy_file,
            '-perturbations_file=%s' % perturbations_file,
            '-success_file=%s' % success_file,
            '-log_file=%s' % log_file,
            '-attack=%s' % attack.attack,
            '-objective=%s' % attack.objective,
            '-max_attempts=%d' % attack.max_attempts,
            '-max_samples=%d' % attack.max_samples,
            '-batch_size=%d' % self.batch_size,
            '-epsilon=%g' % attack.epsilon,
            '-c_0=%g' % attack.c_0,
            '-c_1=%g' % attack.c_1,
            '-c_2=%g' % attack.c_2,
            '-max_iterations=%d' % attack.max_iterations,
            '-max_projections=%d' % attack.max_projections,
            '-base_lr=%g' % attack.base_lr,
            # '-no_gpu',
            # '-no_label_leaking',
            '-on_manifold',
        ] + self.network_parameters()

        if not os.path.exists(perturbations_file) or not os.path.exists(success_file):
            attack_classifier = AttackDecoderClassifier(arguments)
            attack_classifier.main()

        if self.args.reattack or self.args.reevaluate or not os.path.exists(results_file):
            arguments = [
                '-database_file=%s' % self.database_file,
                '-test_images_file=%s' % self.test_images_file,
                '-train_images_file=%s' % self.train_images_file,
                '-test_theta_file=%s' % self.test_theta_file,  # EMNIST FASHION FONTS
                '-train_theta_file=%s' % self.train_theta_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-accuracy_file=%s' % accuracy_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-plot_directory=%s' % plot_directory,
                '-results_file=%s' % results_file,
                '-batch_size=%d' % self.batch_size,
                # '-plot_manifolds',
                # '-no_gpu'
            ]

            test_attack_classifier = TestAttackDecoderClassifier(arguments)
            test_attack_classifier.main()
            results = test_attack_classifier.results
        else:
            results = utils.read_pickle(results_file)
            log('[Experiment] read %s' % results_file)

        key = model_name + '_on'
        if not key in self.results.keys():
            self.results[key] = numpy.zeros((len(self.training_options), self.args.max_models, len(self.on_attack_options), 3, 4))

        for n in range(3):
            self.results[key][t, training.model, a, n, 0] = results[n]['raw_success']
            self.results[key][t, training.model, a, n, 1] = results[n]['raw_iteration']
            self.results[key][t, training.model, a, n, 2] = results[n]['raw_average']
            self.results[key][t, training.model, a, n, 3] = results[n]['raw_image']

    def attack_stn(self, model_name, t, a):
        """
        Attack STN.
        """

        assert self.stn_attack_options is not None
        assert a >= 0 and a < len(self.stn_attack_options)

        training = self.training_options[t]
        attack = self.stn_attack_options[a]
        assert isinstance(attack, STNAttackOptions), 'class %s not instance of STNAttackOptions' % attack.__class__.__name__

        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)
        base_directory = self.attack_directory(model_directory, attack)
        assert os.path.exists(paths.experiment_dir(model_directory)), model_directory

        classifier_file = paths.experiment_file('%s/classifier' % model_directory, ext='.pth.tar')
        perturbations_file = paths.results_file('%s/perturbations' % base_directory)
        success_file = paths.results_file('%s/success' % base_directory)
        accuracy_file = paths.results_file('%s/accuracy' % base_directory)
        log_file = paths.log_file('%s/log' % base_directory)

        results_file = paths.pickle_file('%s/success' % base_directory)
        plot_directory = paths.experiment_dir('%s/' % base_directory)

        if self.args.reattack:
            if os.path.exists(perturbations_file):
                os.unlink(perturbations_file)
            if os.path.exists(success_file):
                os.unlink(success_file)
            if os.path.exists(accuracy_file):
                os.unlink(accuracy_file)
            if os.path.exists(results_file):
                os.unlink(results_file)

        arguments = [
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-label_index=%d' % self.label_index,
            '-classifier_file=%s' % classifier_file,
            '-accuracy_file=%s' % accuracy_file,
            '-perturbations_file=%s' % perturbations_file,
            '-success_file=%s' % success_file,
            '-log_file=%s' % log_file,
            '-attack=%s' % attack.attack,
            '-objective=%s' % attack.objective,
            '-max_attempts=%d' % attack.max_attempts,
            '-max_samples=%d' % attack.max_samples,
            '-batch_size=%d' % self.batch_size,
            '-epsilon=%g' % attack.epsilon,
            '-c_0=%g' % attack.c_0,
            '-c_1=%g' % attack.c_1,
            '-c_2=%g' % attack.c_2,
            '-max_iterations=%d' % attack.max_iterations,
            '-max_projections=%d' % attack.max_projections,
            '-base_lr=%g' % attack.base_lr,
            # '-no_gpu',
            # '-no_label_leaking',
            '-N_theta=%d' % attack.N_theta,
            '-translation_x=%s' % attack.translation_x,
            '-translation_y=%s' % attack.translation_y,
            '-shear_x=%s' % attack.shear_x,
            '-shear_y=%s' % attack.shear_y,
            '-scale=%s' % attack.scale,
            '-rotation=%s' % attack.rotation,
            '-color=%s' % attack.color,
        ] + self.network_parameters()

        if not os.path.exists(perturbations_file) or not os.path.exists(success_file):
            attack_classifier = AttackSTNClassifier(arguments)
            attack_classifier.main()

        if self.args.reattack or self.args.reevaluate or not os.path.exists(results_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-train_images_file=%s' % self.train_images_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-accuracy_file=%s' % accuracy_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-plot_directory=%s' % plot_directory,
                '-results_file=%s' % results_file,
                '-batch_size=%d' % self.batch_size,
                # '-plot_manifolds',
                # '-no_gpu'
            ]

            test_attack_classifier = TestAttackSTNClassifier(arguments)
            test_attack_classifier.main()
            results = test_attack_classifier.results
        else:
            results = utils.read_pickle(results_file)
            log('[Experiment] read %s' % results_file)

        key = model_name + '_stn'
        if not key in self.results.keys():
            self.results[key] = numpy.zeros((len(self.training_options), self.args.max_models, len(self.stn_attack_options), 3, 4))

        for n in range(3):
            self.results[key][t, training.model, a, n, 0] = results[n]['raw_success']
            self.results[key][t, training.model, a, n, 1] = results[n]['raw_iteration']
            self.results[key][t, training.model, a, n, 2] = results[n]['raw_average']
            self.results[key][t, training.model, a, n, 3] = results[n]['raw_image']

    def attack_learned_on_class_manifold(self, model_name, t, a):
        """
        Attack learned on class manifold.
        """

        assert self.learned_on_class_attack_options is not None
        assert a >= 0 and a < len(self.learned_on_class_attack_options)

        training = self.training_options[t]
        attack = self.learned_on_class_attack_options[a]
        assert isinstance(attack, LearnedOnClassAttackOptions), 'class %s not instance of LearnedOnClassAttackOptions' % attack.__class__.__name__

        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)
        base_directory = self.attack_directory(model_directory, attack)
        assert os.path.exists(paths.experiment_dir(model_directory)), model_directory

        classifier_file = paths.experiment_file('%s/classifier' % model_directory, ext='.pth.tar')
        perturbations_file = paths.results_file('%s/perturbations' % base_directory)
        success_file = paths.results_file('%s/success' % base_directory)
        accuracy_file = paths.results_file('%s/accuracy' % base_directory)
        log_file = paths.log_file('%s/log' % base_directory)

        results_file = paths.pickle_file('%s/success' % base_directory)
        plot_directory = paths.experiment_dir('%s/' % base_directory)

        if self.args.reattack:
            if os.path.exists(perturbations_file):
                os.unlink(perturbations_file)
            if os.path.exists(success_file):
                os.unlink(success_file)
            if os.path.exists(accuracy_file):
                os.unlink(accuracy_file)
            if os.path.exists(results_file):
                os.unlink(results_file)

        arguments = [
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-test_theta_file=%s' % self.class_test_theta_file,
            '-label_index=%d' % self.label_index,
            '-decoder_files=%s' % ','.join(self.decoder_files),
            '-latent_space_size=%d' % self.class_latent_space_size,
            '-classifier_file=%s' % classifier_file,
            '-accuracy_file=%s' % accuracy_file,
            '-perturbations_file=%s' % perturbations_file,
            '-success_file=%s' % success_file,
            '-log_file=%s' % log_file,
            '-attack=%s' % attack.attack,
            '-objective=%s' % attack.objective,
            '-max_attempts=%d' % attack.max_attempts,
            '-max_samples=%d' % attack.max_samples,
            '-batch_size=%d' % self.batch_size,
            '-epsilon=%g' % attack.epsilon,
            '-c_0=%g' % attack.c_0,
            '-c_1=%g' % attack.c_1,
            '-c_2=%g' % attack.c_2,
            '-max_iterations=%d' % attack.max_iterations,
            '-max_projections=%d' % attack.max_projections,
            '-base_lr=%g' % attack.base_lr,
            # '-no_gpu',
            # '-no_label_leaking',
            '-on_manifold',
        ] + self.network_parameters() + self.decoder_parameters()

        if not os.path.exists(perturbations_file) or not os.path.exists(success_file):
            attack_classifier = AttackLearnedDecoderClassifier(arguments)
            attack_classifier.main()

        if self.args.reattack or self.args.reevaluate or not os.path.exists(results_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-train_images_file=%s' % self.train_images_file,
                '-test_theta_file=%s' % self.class_test_theta_file,
                '-train_theta_file=%s' % self.class_train_theta_file,
                '-decoder_files=%s' % ','.join(self.decoder_files),
                '-latent_space_size=%d' % self.class_latent_space_size,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-accuracy_file=%s' % accuracy_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-plot_directory=%s' % plot_directory,
                '-results_file=%s' % results_file,
                '-batch_size=%d' % self.batch_size,
                # '-plot_manifolds',
                # '-no_gpu'
                '-bound=%g' % attack.bound,
            ] + self.decoder_parameters()

            test_attack_classifier = TestAttackLearnedDecoderClassifier(arguments)
            test_attack_classifier.main()
            results = test_attack_classifier.results
        else:
            results = utils.read_pickle(results_file)

        key = model_name + '_learned_on_class'
        if not key in self.results.keys():
            self.results[key] = numpy.zeros((len(self.training_options), self.args.max_models, len(self.learned_on_class_attack_options), 3, 4))

        for n in range(3):
            self.results[key][t, training.model, a, n, 0] = results[n]['raw_success']
            self.results[key][t, training.model, a, n, 1] = results[n]['raw_iteration']
            self.results[key][t, training.model, a, n, 2] = results[n]['raw_average']
            self.results[key][t, training.model, a, n, 3] = results[n]['raw_image']

    def attack_learned_on_data_manifold(self, model_name, t, a):
        """
        Attack learned on class manifold.
        """

        assert self.learned_on_data_attack_options is not None
        assert a >= 0 and a < len(self.learned_on_data_attack_options)

        training = self.training_options[t]
        attack = self.learned_on_data_attack_options[a]
        assert isinstance(attack, LearnedOnDataAttackOptions), 'class %s not instance of LearnedOnDataAttackOptions' % attack.__class__.__name__

        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)
        base_directory = self.attack_directory(model_directory, attack)
        assert os.path.exists(paths.experiment_dir(model_directory)), model_directory

        classifier_file = paths.experiment_file('%s/classifier' % model_directory, ext='.pth.tar')
        perturbations_file = paths.results_file('%s/perturbations' % base_directory)
        success_file = paths.results_file('%s/success' % base_directory)
        accuracy_file = paths.results_file('%s/accuracy' % base_directory)
        log_file = paths.log_file('%s/log' % base_directory)

        results_file = paths.pickle_file('%s/success' % base_directory)
        plot_directory = paths.experiment_dir('%s/' % base_directory)

        if self.args.reattack:
            if os.path.exists(perturbations_file):
                os.unlink(perturbations_file)
            if os.path.exists(success_file):
                os.unlink(success_file)
            if os.path.exists(accuracy_file):
                os.unlink(accuracy_file)
            if os.path.exists(results_file):
                os.unlink(results_file)

        arguments = [
            '-test_images_file=%s' % self.test_images_file,
            '-test_codes_file=%s' % self.test_codes_file,
            '-test_theta_file=%s' % self.data_test_theta_file,
            '-label_index=%d' % self.label_index,
            '-decoder_files=%s' % self.decoder_file,
            '-latent_space_size=%d' % self.data_latent_space_size,
            '-classifier_file=%s' % classifier_file,
            '-accuracy_file=%s' % accuracy_file,
            '-perturbations_file=%s' % perturbations_file,
            '-success_file=%s' % success_file,
            '-log_file=%s' % log_file,
            '-attack=%s' % attack.attack,
            '-objective=%s' % attack.objective,
            '-max_attempts=%d' % attack.max_attempts,
            '-max_samples=%d' % attack.max_samples,
            '-batch_size=%d' % self.batch_size,
            '-epsilon=%g' % attack.epsilon,
            '-c_0=%g' % attack.c_0,
            '-c_1=%g' % attack.c_1,
            '-c_2=%g' % attack.c_2,
            '-max_iterations=%d' % attack.max_iterations,
            '-max_projections=%d' % attack.max_projections,
            '-base_lr=%g' % attack.base_lr,
            # '-no_gpu',
            # '-no_label_leaking',
            '-on_manifold',
        ] + self.network_parameters() + self.decoder_parameters()

        if not os.path.exists(perturbations_file) or not os.path.exists(success_file):
            attack_classifier = AttackLearnedDecoderClassifier(arguments)
            attack_classifier.main()

        if self.args.reevaluate or not os.path.exists(results_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-train_images_file=%s' % self.train_images_file,
                '-test_theta_file=%s' % self.data_test_theta_file,
                '-train_theta_file=%s' % self.data_train_theta_file,
                '-decoder_files=%s' % self.decoder_file,
                '-latent_space_size=%d' % self.data_latent_space_size,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-accuracy_file=%s' % accuracy_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-plot_directory=%s' % plot_directory,
                '-results_file=%s' % results_file,
                '-batch_size=%d' % self.batch_size,
                # '-plot_manifolds',
                # '-no_gpu'
                '-bound=%g' % attack.bound,
            ] + self.decoder_parameters()

            test_attack_classifier = TestAttackLearnedDecoderClassifier(arguments)
            test_attack_classifier.main()
            results = test_attack_classifier.results
        else:
            results = utils.read_pickle(results_file)

        key = model_name + '_learned_on_data'
        if not key in self.results.keys():
            self.results[key] = numpy.zeros((len(self.training_options), self.args.max_models, len(self.learned_on_data_attack_options), 3, 4))

        for n in range(3):
            self.results[key][t, training.model, a, n, 0] = results[n]['raw_success']
            self.results[key][t, training.model, a, n, 1] = results[n]['raw_iteration']
            self.results[key][t, training.model, a, n, 2] = results[n]['raw_average']
            self.results[key][t, training.model, a, n, 3] = results[n]['raw_image']

    def attack_off_manifold_transfer(self, transfer_model_directory, target_model_name, t, a):
        """
        Attack off manifold.
        """

        assert self.off_attack_options is not None
        assert a >= 0 and a < len(self.off_attack_options)

        training = self.training_options[t]
        attack = self.off_attack_options[a]
        assert isinstance(attack, OffAttackOptions), 'class %s not instance of OffAttackOptions' % attack.__class__.__name__

        transfer_directory = self.attack_directory(transfer_model_directory, attack)
        assert os.path.exists(paths.experiment_dir(transfer_model_directory)), transfer_model_directory
        assert os.path.exists(paths.experiment_dir(transfer_directory)), transfer_directory

        target_model_directory = '%s_%d_%d' % (target_model_name, training.model, training.training_samples)
        target_directory = self.attack_directory(target_model_directory, attack) + '_Transfer'
        assert os.path.exists(paths.experiment_dir(target_model_directory)), target_model_directory

        classifier_file = paths.experiment_file('%s/classifier' % target_model_directory, ext='.pth.tar')
        perturbations_file = paths.results_file('%s/perturbations' % transfer_directory)
        original_success_file = paths.results_file('%s/success' % transfer_directory)
        transfer_success_file = paths.results_file('%s/success' % target_directory)
        original_accuracy_file = paths.results_file('%s/accuracy' % transfer_directory)
        transfer_accuracy_file = paths.results_file('%s/accuracy' % target_directory)
        log_file = paths.log_file('%s/log' % target_directory)

        results_file = paths.pickle_file('%s/success' % target_directory)
        plot_directory = paths.experiment_dir('%s/' % target_directory)

        if self.args.reattack:
            if os.path.exists(transfer_success_file):
                os.unlink(transfer_success_file)
            if os.path.exists(results_file):
                os.unlink(results_file)

        if not os.path.exists(transfer_success_file) or not os.path.exists(transfer_accuracy_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-classifier_file=%s' % classifier_file,
                '-perturbations_file=%s' % perturbations_file,
                '-original_success_file=%s' % original_success_file,
                '-transfer_success_file=%s' % transfer_success_file,
                '-original_accuracy_file=%s' % original_accuracy_file,
                '-transfer_accuracy_file=%s' % transfer_accuracy_file,
                '-log_file=%s' % log_file,
                '-batch_size=%d' % self.batch_size,
            ]
            test_perturbations = TestPerturbations(arguments)
            test_perturbations.main()

        if self.args.reevaluate or not os.path.exists(results_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-train_images_file=%s' % self.train_images_file,
                '-test_theta_file=%s' % self.test_theta_file,  # EMNIST FASHION FONTS
                '-train_theta_file=%s' % self.train_theta_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-accuracy_file=%s' % transfer_accuracy_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % transfer_success_file,
                '-results_file=%s' % results_file,
                '-plot_directory=%s' % plot_directory,
                # '-plot_manifolds'
            ]

            test_attack_classifier = TestAttackClassifier(arguments)
            test_attack_classifier.main()
            results = test_attack_classifier.results
        else:
            results = utils.read_pickle(results_file)
            log('[Experiment] read %s' % results_file)

        key = target_model_name + '_off_transfer'
        if not key in self.results.keys():
            self.results[key] = numpy.zeros((len(self.training_options), self.args.max_models, len(self.off_attack_options), 3, 4))

        for n in range(3):
            self.results[key][t, training.model, a, n, 0] = results[n]['raw_success']
            self.results[key][t, training.model, a, n, 1] = results[n]['raw_iteration']
            self.results[key][t, training.model, a, n, 2] = results[n]['raw_average']
            self.results[key][t, training.model, a, n, 3] = results[n]['raw_latent']

    def attack_on_manifold_transfer(self, transfer_model_directory, target_model_name, t, a):
        """
        Attack transfer.
        """

        assert self.on_attack_options is not None
        assert a >= 0 and a < len(self.on_attack_options)

        training = self.training_options[t]
        attack = self.on_attack_options[a]
        assert isinstance(attack, OnAttackOptions), 'class %s not instance of OnAttackOptions' % attack.__class__.__name__

        transfer_directory = self.attack_directory(transfer_model_directory, attack)
        assert os.path.exists(paths.experiment_dir(transfer_model_directory)), transfer_model_directory
        assert os.path.exists(paths.experiment_dir(transfer_directory)), transfer_directory

        target_model_directory = '%s_%d_%d' % (target_model_name, training.model, training.training_samples)
        target_directory = self.attack_directory(target_model_directory, attack) + '_Transfer'
        assert os.path.exists(paths.experiment_dir(target_model_directory)), target_model_directory

        classifier_file = paths.experiment_file('%s/classifier' % target_model_directory, ext='.pth.tar')
        perturbations_file = paths.results_file('%s/perturbations' % transfer_directory)
        perturbation_images_file = paths.results_file('%s/perturbations' % target_directory)
        original_success_file = paths.results_file('%s/success' % transfer_directory)
        transfer_success_file = paths.results_file('%s/success' % target_directory)
        original_accuracy_file = paths.results_file('%s/accuracy' % transfer_directory)
        transfer_accuracy_file = paths.results_file('%s/accuracy' % target_directory)
        log_file = paths.log_file('%s/log' % target_directory)

        results_file = paths.pickle_file('%s/success' % target_directory)
        plot_directory = paths.experiment_dir('%s/' % target_directory)

        if self.args.reattack:
            if os.path.exists(transfer_success_file):
                os.unlink(transfer_success_file)
            if os.path.exists(results_file):
                os.unlink(results_file)
            if os.path.exists(perturbation_images_file):
                os.unlink(perturbation_images_file)

        if not os.path.exists(perturbation_images_file):
            arguments = [
                '-database_file=%s' % self.database_file,
                '-test_theta_file=%s' % self.test_theta_file,  # EMNIST FASHION FONTS
                '-test_codes_file=%s' % self.test_codes_file,
                #'-label_index=%d' % self.label_index,
                '-perturbations_file=%s' % perturbations_file,
                '-perturbation_images_file=%s' % perturbation_images_file,
                '-log_file=',
                '-batch_size=%d' % self.batch_size
            ]
            compute_decoder_perturbations = ComputeDecoderPerturbations(arguments)
            compute_decoder_perturbations.main()

        if not os.path.exists(transfer_success_file) or not os.path.exists(transfer_accuracy_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-classifier_file=%s' % classifier_file,
                '-perturbations_file=%s' % perturbation_images_file,
                '-original_success_file=%s' % original_success_file,
                '-transfer_success_file=%s' % transfer_success_file,
                '-original_accuracy_file=%s' % original_accuracy_file,
                '-transfer_accuracy_file=%s' % transfer_accuracy_file,
                '-log_file=%s' % log_file,
                '-batch_size=%d' % self.batch_size
            ]
            test_perturbations = TestPerturbations(arguments)
            test_perturbations.main()

        if self.args.reevaluate or not os.path.exists(results_file):
            arguments = [
                '-database_file=%s' % self.database_file,
                '-test_images_file=%s' % self.test_images_file,
                '-train_images_file=%s' % self.train_images_file,
                '-test_theta_file=%s' % self.test_theta_file,  # EMNIST FASHION FONTS
                '-train_theta_file=%s' % self.train_theta_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-accuracy_file=%s' % transfer_accuracy_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % transfer_success_file,
                '-plot_directory=%s' % plot_directory,
                '-results_file=%s' % results_file,
                '-batch_size=%d' % self.batch_size,
                # '-plot_manifolds',
                # '-no_gpu'
            ]

            test_attack_classifier = TestAttackDecoderClassifier(arguments)
            test_attack_classifier.main()
            results = test_attack_classifier.results
        else:
            results = utils.read_pickle(results_file)
            log('[Experiment] read %s' % results_file)

        key = target_model_name + '_on_transfer'
        if not key in self.results.keys():
            self.results[key] = numpy.zeros((len(self.training_options), self.args.max_models, len(self.on_attack_options), 3, 4))

        for n in range(3):
            self.results[key][t, training.model, a, n, 0] = results[n]['raw_success']
            self.results[key][t, training.model, a, n, 1] = results[n]['raw_iteration']
            self.results[key][t, training.model, a, n, 2] = results[n]['raw_average']
            self.results[key][t, training.model, a, n, 3] = results[n]['raw_image']

    def attack_learned_on_class_manifold_transfer(self, transfer_model_directory, target_model_name, t, a):
        """
        Attack transfer.
        """

        assert self.learned_on_class_attack_options is not None
        assert a >= 0 and a < len(self.learned_on_class_attack_options)

        training = self.training_options[t]
        attack = self.learned_on_class_attack_options[a]
        assert isinstance(attack, LearnedOnClassAttackOptions), 'class %s not instance of LearnedOnClassAttackOptions' % attack.__class__.__name__

        transfer_directory = self.attack_directory(transfer_model_directory, attack)
        assert os.path.exists(paths.experiment_dir(transfer_model_directory)), transfer_model_directory
        assert os.path.exists(paths.experiment_dir(transfer_directory)), transfer_directory

        target_model_directory = '%s_%d_%d' % (target_model_name, training.model, training.training_samples)
        target_directory = self.attack_directory(target_model_directory, attack) + '_Transfer'
        assert os.path.exists(paths.experiment_dir(target_model_directory)), target_model_directory

        classifier_file = paths.experiment_file('%s/classifier' % target_model_directory, ext='.pth.tar')
        perturbations_file = paths.results_file('%s/perturbations' % transfer_directory)
        perturbation_images_file = paths.results_file('%s/perturbations' % target_directory)
        original_success_file = paths.results_file('%s/success' % transfer_directory)
        transfer_success_file = paths.results_file('%s/success' % target_directory)
        original_accuracy_file = paths.results_file('%s/accuracy' % transfer_directory)
        transfer_accuracy_file = paths.results_file('%s/accuracy' % target_directory)
        log_file = paths.log_file('%s/log' % target_directory)

        results_file = paths.pickle_file('%s/success' % target_directory)
        plot_directory = paths.experiment_dir('%s/' % target_directory)

        if self.args.reattack:
            if os.path.exists(transfer_success_file):
                os.unlink(transfer_success_file)
            if os.path.exists(results_file):
                os.unlink(results_file)
            if os.path.exists(perturbation_images_file):
                os.unlink(perturbation_images_file)

        if not os.path.exists(perturbation_images_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-decoder_files=%s' % ','.join(self.decoder_files),
                '-latent_space_size=%d' % self.class_latent_space_size,
                '-perturbations_file=%s' % perturbations_file,
                '-perturbation_images_file=%s' % perturbation_images_file,
                '-log_file=',
                '-batch_size=%d' % self.batch_size
            ] + self.decoder_parameters()
            compute_decoder_perturbations = ComputeLearnedDecoderPerturbations(arguments)
            compute_decoder_perturbations.main()

        if not os.path.exists(transfer_success_file) or not os.path.exists(transfer_accuracy_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-classifier_file=%s' % classifier_file,
                '-perturbations_file=%s' % perturbation_images_file,
                '-original_success_file=%s' % original_success_file,
                '-transfer_success_file=%s' % transfer_success_file,
                '-original_accuracy_file=%s' % original_accuracy_file,
                '-transfer_accuracy_file=%s' % transfer_accuracy_file,
                '-log_file=%s' % log_file,
                '-batch_size=%d' % self.batch_size
            ]
            test_perturbations = TestPerturbations(arguments)
            test_perturbations.main()

        if self.args.reevaluate or not os.path.exists(results_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-train_images_file=%s' % self.train_images_file,
                '-test_theta_file=%s' % self.class_test_theta_file,
                '-train_theta_file=%s' % self.class_train_theta_file,
                '-decoder_files=%s' % ','.join(self.decoder_files),
                '-latent_space_size=%d' % self.class_latent_space_size,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-accuracy_file=%s' % transfer_accuracy_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % transfer_success_file,
                '-plot_directory=%s' % plot_directory,
                '-results_file=%s' % results_file,
                '-batch_size=%d' % self.batch_size,
                # '-plot_manifolds',
                # '-no_gpu'
                '-bound=%g' % attack.bound,
            ] + self.decoder_parameters()

            test_attack_classifier = TestAttackLearnedDecoderClassifier(arguments)
            test_attack_classifier.main()
            results = test_attack_classifier.results
        else:
            results = utils.read_pickle(results_file)

        key = target_model_name + '_learned_on_class_transfer'
        if not key in self.results.keys():
            self.results[key] = numpy.zeros((len(self.training_options), self.args.max_models, len(self.learned_on_class_attack_options), 3, 4))

        for n in range(3):
            self.results[key][t, training.model, a, n, 0] = results[n]['raw_success']
            self.results[key][t, training.model, a, n, 1] = results[n]['raw_iteration']
            self.results[key][t, training.model, a, n, 2] = results[n]['raw_average']
            self.results[key][t, training.model, a, n, 3] = results[n]['raw_image']

    def attack_learned_on_data_manifold_transfer(self, transfer_model_directory, target_model_name, t, a):
        """
        Attack transfer.
        """

        assert self.learned_on_data_attack_options is not None
        assert a >= 0 and a < len(self.learned_on_data_attack_options)

        training = self.training_options[t]
        attack = self.learned_on_data_attack_options[a]
        assert isinstance(attack, LearnedOnDataAttackOptions), 'class %s not instance of LearnedOnDataAttackOptions' % attack.__class__.__name__

        transfer_directory = self.attack_directory(transfer_model_directory, attack)
        assert os.path.exists(paths.experiment_dir(transfer_model_directory)), transfer_model_directory
        assert os.path.exists(paths.experiment_dir(transfer_directory)), transfer_directory

        target_model_directory = '%s_%d_%d' % (target_model_name, training.model, training.training_samples)
        target_directory = self.attack_directory(target_model_directory, attack) + '_Transfer'
        assert os.path.exists(paths.experiment_dir(target_model_directory)), target_model_directory

        classifier_file = paths.experiment_file('%s/classifier' % target_model_directory, ext='.pth.tar')
        perturbations_file = paths.results_file('%s/perturbations' % transfer_directory)
        perturbation_images_file = paths.results_file('%s/perturbations' % target_directory)
        original_success_file = paths.results_file('%s/success' % transfer_directory)
        transfer_success_file = paths.results_file('%s/success' % target_directory)
        original_accuracy_file = paths.results_file('%s/accuracy' % transfer_directory)
        transfer_accuracy_file = paths.results_file('%s/accuracy' % target_directory)
        log_file = paths.log_file('%s/log' % target_directory)

        results_file = paths.pickle_file('%s/success' % target_directory)
        plot_directory = paths.experiment_dir('%s/' % target_directory)

        if self.args.reattack:
            if os.path.exists(transfer_success_file):
                os.unlink(transfer_success_file)
            if os.path.exists(results_file):
                os.unlink(results_file)
            if os.path.exists(perturbation_images_file):
                os.unlink(perturbation_images_file)

        if not os.path.exists(perturbation_images_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-decoder_files=%s' % self.decoder_file,
                '-latent_space_size=%d' % self.data_latent_space_size,
                '-perturbations_file=%s' % perturbations_file,
                '-perturbation_images_file=%s' % perturbation_images_file,
                '-log_file=',
                '-batch_size=%d' % self.batch_size
            ] + self.decoder_parameters()
            compute_decoder_perturbations = ComputeLearnedDecoderPerturbations(arguments)
            compute_decoder_perturbations.main()

        if not os.path.exists(transfer_success_file) or not os.path.exists(transfer_accuracy_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-classifier_file=%s' % classifier_file,
                '-perturbations_file=%s' % perturbation_images_file,
                '-original_success_file=%s' % original_success_file,
                '-transfer_success_file=%s' % transfer_success_file,
                '-original_accuracy_file=%s' % original_accuracy_file,
                '-transfer_accuracy_file=%s' % transfer_accuracy_file,
                '-log_file=%s' % log_file,
                '-batch_size=%d' % self.batch_size
            ]
            test_perturbations = TestPerturbations(arguments)
            test_perturbations.main()

        if self.args.reevaluate or not os.path.exists(results_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-train_images_file=%s' % self.train_images_file,
                '-test_theta_file=%s' % self.data_test_theta_file,
                '-train_theta_file=%s' % self.data_train_theta_file,
                '-decoder_files=%s' % self.decoder_file,
                '-latent_space_size=%d' % self.data_latent_space_size,
                '-test_codes_file=%s' % self.test_codes_file,
                '-accuracy_file=%s' % transfer_accuracy_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % transfer_success_file,
                '-plot_directory=%s' % plot_directory,
                '-results_file=%s' % results_file,
                '-batch_size=%d' % self.batch_size,
                # '-plot_manifolds',
                # '-no_gpu'
                '-bound=%g' % attack.bound,
            ] + self.decoder_parameters()

            test_attack_classifier = TestAttackLearnedDecoderClassifier(arguments)
            test_attack_classifier.main()
            results = test_attack_classifier.results
        else:
            results = utils.read_pickle(results_file)

        key = target_model_name + '_learned_on_data_transfer'
        if not key in self.results.keys():
            self.results[key] = numpy.zeros((len(self.training_options), self.args.max_models, len(self.learned_on_data_attack_options), 3, 4))

        for n in range(3):
            self.results[key][t, training.model, a, n, 0] = results[n]['raw_success']
            self.results[key][t, training.model, a, n, 1] = results[n]['raw_iteration']
            self.results[key][t, training.model, a, n, 2] = results[n]['raw_average']
            self.results[key][t, training.model, a, n, 3] = results[n]['raw_image']

    def attack_stn_transfer(self, transfer_model_directory, target_model_name, t, a):
        """
        Attack transfer.
        """

        assert self.stn_attack_options is not None
        assert a >= 0 and a < len(self.stn_attack_options)

        training = self.training_options[t]
        attack = self.stn_attack_options[a]
        assert isinstance(attack, STNAttackOptions), 'class %s not instance of OnAttackOptions' % attack.__class__.__name__

        transfer_directory = self.attack_directory(transfer_model_directory, attack)
        assert os.path.exists(paths.experiment_dir(transfer_model_directory)), transfer_model_directory
        assert os.path.exists(paths.experiment_dir(transfer_directory)), transfer_directory

        target_model_directory = '%s_%d_%d' % (target_model_name, training.model, training.training_samples)
        target_directory = self.attack_directory(target_model_directory, attack) + '_Transfer'
        assert os.path.exists(paths.experiment_dir(target_model_directory)), target_model_directory

        classifier_file = paths.experiment_file('%s/classifier' % target_model_directory, ext='.pth.tar')
        perturbations_file = paths.results_file('%s/perturbations' % transfer_directory)
        perturbation_images_file = paths.results_file('%s/perturbations' % target_directory)
        original_success_file = paths.results_file('%s/success' % transfer_directory)
        transfer_success_file = paths.results_file('%s/success' % target_directory)
        original_accuracy_file = paths.results_file('%s/accuracy' % transfer_directory)
        transfer_accuracy_file = paths.results_file('%s/accuracy' % target_directory)
        log_file = paths.log_file('%s/log' % target_directory)

        results_file = paths.pickle_file('%s/success' % target_directory)
        plot_directory = paths.experiment_dir('%s/' % target_directory)

        if self.args.reattack:
            if os.path.exists(transfer_success_file):
                os.unlink(transfer_success_file)
            if os.path.exists(results_file):
                os.unlink(results_file)
            if os.path.exists(perturbation_images_file):
                os.unlink(perturbation_images_file)

        if not os.path.exists(perturbation_images_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-perturbations_file=%s' % perturbations_file,
                '-perturbation_images_file=%s' % perturbation_images_file,
                '-log_file=',
                '-batch_size=%d' % self.batch_size,
            ]
            compute_decoder_perturbations = ComputeSTNPerturbations(arguments)
            compute_decoder_perturbations.main()

        if not os.path.exists(transfer_success_file) or not os.path.exists(transfer_accuracy_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-classifier_file=%s' % classifier_file,
                '-perturbations_file=%s' % perturbation_images_file,
                '-original_success_file=%s' % original_success_file,
                '-transfer_success_file=%s' % transfer_success_file,
                '-original_accuracy_file=%s' % original_accuracy_file,
                '-transfer_accuracy_file=%s' % transfer_accuracy_file,
                '-log_file=%s' % log_file,
                '-batch_size=%d' % self.batch_size,
            ]
            test_perturbations = TestPerturbations(arguments)
            test_perturbations.main()

        if self.args.reevaluate or not os.path.exists(results_file):
            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-train_images_file=%s' % self.train_images_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-accuracy_file=%s' % transfer_accuracy_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % transfer_success_file,
                '-plot_directory=%s' % plot_directory,
                '-results_file=%s' % results_file,
                '-batch_size=%d' % self.batch_size,
                # '-plot_manifolds',
                # '-no_gpu'
            ]

            test_attack_classifier = TestAttackSTNClassifier(arguments)
            test_attack_classifier.main()
            results = test_attack_classifier.results
        else:
            results = utils.read_pickle(results_file)
            log('[Experiment] read %s' % results_file)

        key = target_model_name + '_stn_transfer'
        if not key in self.results.keys():
            self.results[key] = numpy.zeros((len(self.training_options), self.args.max_models, len(self.stn_attack_options), 3, 4))

        for n in range(3):
            self.results[key][t, training.model, a, n, 0] = results[n]['raw_success']
            self.results[key][t, training.model, a, n, 1] = results[n]['raw_iteration']
            self.results[key][t, training.model, a, n, 2] = results[n]['raw_average']
            self.results[key][t, training.model, a, n, 3] = results[n]['raw_image']

    def visualize_off_manifold(self, model_name, selection_file='', t=0, a=0):
        """
        Test.
        """

        # Guard
        if not utils.display():
            return

        training = self.training_options[t]
        attack = self.off_attack_options[a]
        assert isinstance(attack, OffAttackOptions)

        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)
        base_directory = self.attack_directory(model_directory, attack)
        assert os.path.exists(paths.experiment_dir(model_directory)), model_directory

        classifier_file = paths.experiment_file('%s/classifier' % model_directory, ext='.pth.tar')
        perturbations_file = paths.results_file('%s/perturbations' % base_directory)
        success_file = paths.results_file('%s/success' % base_directory)
        accuracy_file = paths.results_file('%s/accuracy' % base_directory)
        output_directory = paths.experiment_dir('%s/perturbations/' % base_directory)

        if not os.path.exists(output_directory) or self.args.revisualize:

            # Remove if already exists!
            if self.args.revisualize and os.path.exists(output_directory):
                shutil.rmtree(output_directory)

            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%s' % self.label_index,
                '-classifier_file=%s' % classifier_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-selection_file=%s' % selection_file,
                '-accuracy_file=%s' % accuracy_file,
                '-output_directory=%s' % output_directory,
                '-batch_size=%d' % self.batch_size,
            ] + self.network_parameters()

            visualize_attack_classifier = VisualizeAttackClassifier(arguments)
            visualize_attack_classifier.main()
        else:
            log('[Experiment] %s exists' % output_directory)

    def detect_off_manifold(self, model_name, t=0, a=0):
        """
        Test.
        """

        training = self.training_options[t]
        attack = self.off_attack_options[a]
        assert isinstance(attack, OffAttackOptions)

        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)
        base_directory = self.attack_directory(model_directory, attack)
        assert os.path.exists(paths.experiment_dir(model_directory)), model_directory

        perturbations_file = paths.results_file('%s/perturbations' % base_directory)
        success_file = paths.results_file('%s/success' % base_directory)
        accuracy_file = paths.results_file('%s/accuracy' % base_directory)
        plot_directory = paths.experiment_dir('%s/detection/' % base_directory)

        if not os.path.exists(os.path.join(plot_directory, 'true_50_10')) or self.args.revisualize:
            if self.database_file:
                arguments = [
                    '-mode=true',
                    '-database_file=%s' % self.database_file,
                    '-train_images_file=%s' % self.train_images_file,
                    '-test_images_file=%s' % self.test_images_file,
                    '-test_theta_file=%s' % self.test_theta_file,
                    '-test_codes_file=%s' % self.test_codes_file,
                    '-perturbations_file=%s' % perturbations_file,
                    '-success_file=%s' % success_file,
                    '-accuracy_file=%s' % accuracy_file,
                    '-batch_size=%d' % self.batch_size,
                    '-pre_pca=40',
                    '-n_nearest_neighbors=50',
                    '-n_fit=240000',
                    '-plot_directory=%s' % plot_directory,
                    '-n_pca=10',
                    '-max_samples=5000'
                ]

                detect_attack_classifier_nn = DetectAttackClassifierNN(arguments)
                detect_attack_classifier_nn.main()

        if not os.path.exists(os.path.join(plot_directory, 'appr_50_10')) or self.args.revisualize:
            arguments = [
                '-mode=appr',
                '-train_images_file=%s' % self.train_images_file,
                '-test_images_file=%s' % self.test_images_file,
                '-test_theta_file=%s' % self.class_test_theta_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-perturbations_file=%s' % perturbations_file,
                '-decoder_files=%s' % ','.join(self.decoder_files),
                '-latent_space_size=%s' % self.class_latent_space_size,
                '-success_file=%s' % success_file,
                '-accuracy_file=%s' % accuracy_file,
                '-batch_size=%d' % self.batch_size,
                '-pre_pca=40',
                '-n_nearest_neighbors=50',
                '-n_fit=240000',
                '-plot_directory=%s' % plot_directory,
                '-n_pca=10',
                '-max_samples=5000'
            ] + self.decoder_parameters()

            #detect_attack_classifier_nn = DetectAttackClassifierNN(arguments)
            #detect_attack_classifier_nn.main()

        if not os.path.exists(os.path.join(plot_directory, 'nn_50_10')) or self.args.revisualize:
            arguments = [
                '-mode=nn',
                '-train_images_file=%s' % self.train_images_file,
                '-test_images_file=%s' % self.test_images_file,
                '-test_theta_file=%s' % self.test_theta_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-accuracy_file=%s' % accuracy_file,
                '-batch_size=%d' % self.batch_size,
                '-pre_pca=40',
                '-n_nearest_neighbors=50',
                '-n_fit=240000',
                '-plot_directory=%s' % plot_directory,
                '-n_pca=10',
                '-max_samples=5000'
            ]

            detect_attack_classifier_nn = DetectAttackClassifierNN(arguments)
            detect_attack_classifier_nn.main()

        if not os.path.exists(os.path.join(plot_directory, 'inclusive_nn_50_10')) or self.args.revisualize:
            arguments = [
                '-mode=inclusive_nn',
                '-train_images_file=%s' % self.train_images_file,
                '-test_images_file=%s' % self.test_images_file,
                '-test_theta_file=%s' % self.test_theta_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-accuracy_file=%s' % accuracy_file,
                '-batch_size=%d' % self.batch_size,
                '-pre_pca=40',
                '-n_nearest_neighbors=50',
                '-n_fit=240000',
                '-plot_directory=%s' % plot_directory,
                '-n_pca=10',
                '-max_samples=5000'
            ]

            detect_attack_classifier_nn = DetectAttackClassifierNN(arguments)
            detect_attack_classifier_nn.main()

    def visualize_on_manifold(self, model_name, t=0, a=0):
        """
        Visualize on manifold.
        """

        # Guard
        if not utils.display():
            return

        training = self.training_options[t]
        attack = self.on_attack_options[a]
        assert isinstance(attack, OnAttackOptions)

        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)
        base_directory = self.attack_directory(model_directory, attack)
        assert os.path.exists(paths.experiment_dir(model_directory)), model_directory

        classifier_file = paths.experiment_file('%s/classifier' % model_directory, ext='.pth.tar')
        perturbations_file = paths.results_file('%s/perturbations' % base_directory)
        success_file = paths.results_file('%s/success' % base_directory)
        accuracy_file = paths.results_file('%s/accuracy' % base_directory)
        output_directory = paths.experiment_dir('%s/perturbations/' % base_directory)

        if not os.path.exists(output_directory) or self.args.revisualize:

            # Remove if already exists!
            if self.args.revisualize and os.path.exists(output_directory):
                shutil.rmtree(output_directory)

            arguments = [
                '-database_file=%s' % self.database_file,
                '-test_theta_file=%s' % self.test_theta_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-test_images_file=%s' % self.test_images_file,
                '-label_index=%s' % self.label_index,
                '-classifier_file=%s' % classifier_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-accuracy_file=%s' % accuracy_file,
                '-output_directory=%s' % output_directory,
                '-batch_size=%d' % self.batch_size,
            ] + self.network_parameters()

            visualize_attack_classifier = VisualizeAttackDecoderClassifier(arguments)
            visualize_attack_classifier.main()
        else:
            log('[Experiment] %s exists' % output_directory)

        return success_file

    def detect_on_manifold(self, model_name, t=0, a=0):
        """
        Visualize on manifold.
        """

        assert self.database_file

        training = self.training_options[t]
        attack = self.on_attack_options[a]
        assert isinstance(attack, OnAttackOptions)

        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)
        base_directory = self.attack_directory(model_directory, attack)
        assert os.path.exists(paths.experiment_dir(model_directory)), model_directory

        perturbations_file = paths.results_file('%s/perturbations' % base_directory)
        success_file = paths.results_file('%s/success' % base_directory)
        accuracy_file = paths.results_file('%s/accuracy' % base_directory)
        plot_directory = paths.experiment_dir('%s/detection/' % base_directory)

        if not os.path.exists(os.path.join(plot_directory, 'true_50_10')) or self.args.revisualize:
            arguments = [
                '-mode=true',
                '-database_file=%s' % self.database_file,
                '-train_images_file=%s' % self.train_images_file,
                '-test_images_file=%s' % self.test_images_file,
                '-train_codes_file=%s' % self.train_codes_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-test_theta_file=%s' % self.test_theta_file,
                '-label_index=%d' % self.label_index,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-accuracy_file=%s' % accuracy_file,
                '-batch_size=%d' % self.batch_size,
                '-pre_pca=40',
                '-n_nearest_neighbors=50',
                '-n_fit=240000',
                '-plot_directory=%s' % plot_directory,
                '-n_pca=10',
                '-max_samples=10000'
            ]

            detect_attack_classifier_nn = DetectAttackDecoderClassifierNN(arguments)
            detect_attack_classifier_nn.main()

        if not os.path.exists(os.path.join(plot_directory, 'appr_50_10')) or self.args.revisualize:
            arguments = [
                '-mode=appr',
                '-train_images_file=%s' % self.train_images_file,
                '-test_images_file=%s' % self.test_images_file,
                '-train_codes_file=%s' % self.train_codes_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-test_theta_file=%s' % self.class_test_theta_file,
                '-decoder_files=%s' % ','.join(self.decoder_files),
                '-latent_space_size=%s' % self.class_latent_space_size,
                '-label_index=%d' % self.label_index,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-accuracy_file=%s' % accuracy_file,
                '-batch_size=%d' % self.batch_size,
                '-pre_pca=40',
                '-n_nearest_neighbors=50',
                '-n_fit=240000',
                '-plot_directory=%s' % plot_directory,
                '-n_pca=10',
                '-max_samples=10000'
            ] + self.decoder_parameters()

            #detect_attack_classifier_nn = DetectAttackDecoderClassifierNN(arguments)
            #detect_attack_classifier_nn.main()

        if not os.path.exists(os.path.join(plot_directory, 'nn_50_10')) or self.args.revisualize:
            arguments = [
                '-mode=nn',
                '-database_file=%s' % self.database_file,
                '-train_images_file=%s' % self.train_images_file,
                '-test_images_file=%s' % self.test_images_file,
                '-train_codes_file=%s' % self.train_codes_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-test_theta_file=%s' % self.test_theta_file,
                '-label_index=%d' % self.label_index,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-accuracy_file=%s' % accuracy_file,
                '-batch_size=%d' % self.batch_size,
                '-pre_pca=40',
                '-n_nearest_neighbors=50',
                '-n_fit=240000',
                '-plot_directory=%s' % plot_directory,
                '-n_pca=10',
                '-max_samples=10000'
            ]

            detect_attack_classifier_nn = DetectAttackDecoderClassifierNN(arguments)
            detect_attack_classifier_nn.main()

        if not os.path.exists(os.path.join(plot_directory, 'unclusive_nn_50_10')) or self.args.revisualize:
            arguments = [
                '-mode=inclusive_nn',
                '-database_file=%s' % self.database_file,
                '-train_images_file=%s' % self.train_images_file,
                '-test_images_file=%s' % self.test_images_file,
                '-train_codes_file=%s' % self.train_codes_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-test_theta_file=%s' % self.test_theta_file,
                '-label_index=%d' % self.label_index,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-accuracy_file=%s' % accuracy_file,
                '-batch_size=%d' % self.batch_size,
                '-pre_pca=40',
                '-n_nearest_neighbors=50',
                '-n_fit=240000',
                '-plot_directory=%s' % plot_directory,
                '-n_pca=10',
                '-max_samples=10000'
            ]

            detect_attack_classifier_nn = DetectAttackDecoderClassifierNN(arguments)
            detect_attack_classifier_nn.main()

    def visualize_learned_on_class_manifold(self, model_name, t=0, a=0):
        """
        Test.
        """

        # Guard
        if not utils.display():
            return

        training = self.training_options[t]
        attack = self.learned_on_class_attack_options[a]
        assert isinstance(attack, LearnedOnClassAttackOptions)

        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)
        base_directory = self.attack_directory(model_directory, attack)
        assert os.path.exists(paths.experiment_dir(model_directory)), model_directory

        classifier_file = paths.experiment_file('%s/classifier' % model_directory, ext='.pth.tar')
        perturbations_file = paths.results_file('%s/perturbations' % base_directory)
        success_file = paths.results_file('%s/success' % base_directory)
        accuracy_file = paths.results_file('%s/accuracy' % base_directory)
        output_directory = paths.experiment_dir('%s/perturbations/' % base_directory)

        if not os.path.exists(output_directory) or self.args.revisualize:

            # Remove if already exists!
            if self.args.revisualize and os.path.exists(output_directory):
                shutil.rmtree(output_directory)

            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-test_theta_file=%s' % self.class_test_theta_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-decoder_files=%s' % ','.join(self.decoder_files),
                '-classifier_file=%s' % classifier_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-accuracy_file=%s' % accuracy_file,
                '-output_directory=%s' % output_directory,
                '-batch_size=%d' % self.batch_size,
                '-latent_space_size=%d' % self.class_latent_space_size
            ] + self.network_parameters() + self.decoder_parameters()

            visualize_attack_classifier = VisualizeAttackLearnedDecoderClassifier(arguments)
            visualize_attack_classifier.main()
        else:
            log('[Experiment] %s exists' % output_directory)

        return success_file

    def detect_learned_on_class_manifold(self, model_name, t=0, a=0):
        """
        Learned on class manifold.
        """

        training = self.training_options[t]
        attack = self.learned_on_class_attack_options[a]
        assert isinstance(attack, LearnedOnClassAttackOptions)

        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)
        base_directory = self.attack_directory(model_directory, attack)
        assert os.path.exists(paths.experiment_dir(model_directory)), model_directory

        perturbations_file = paths.results_file('%s/perturbations' % base_directory)
        success_file = paths.results_file('%s/success' % base_directory)
        accuracy_file = paths.results_file('%s/accuracy' % base_directory)
        plot_directory = paths.experiment_dir('%s/detection/' % base_directory)

        if self.database_file:
            if not os.path.exists(os.path.join(plot_directory, 'true_50_10')) or self.args.revisualize:
                arguments = [
                    '-mode=true',
                    '-database_file=%s' % self.database_file,
                    '-train_images_file=%s' % self.train_images_file,
                    '-test_images_file=%s' % self.test_images_file,
                    '-train_codes_file=%s' % self.train_codes_file,
                    '-test_codes_file=%s' % self.test_codes_file,
                    '-test_theta_file=%s' % self.test_theta_file,
                    '-label_index=%d' % self.label_index,
                    '-perturbations_file=%s' % perturbations_file,
                    '-success_file=%s' % success_file,
                    '-accuracy_file=%s' % accuracy_file,
                    '-batch_size=%d' % self.batch_size,
                    '-pre_pca=40',
                    '-n_nearest_neighbors=50',
                    '-n_fit=240000',
                    '-plot_directory=%s' % plot_directory,
                    '-n_pca=10',
                    '-max_samples=10000'
                ]

                detect_attack_classifier_nn = DetectAttackDecoderClassifierNN(arguments)
                detect_attack_classifier_nn.main()

        if not os.path.exists(os.path.join(plot_directory, 'appr_50_10')) or self.args.revisualize:
            arguments = [
                '-mode=appr',
                '-train_images_file=%s' % self.train_images_file,
                '-test_images_file=%s' % self.test_images_file,
                '-train_codes_file=%s' % self.train_codes_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-test_theta_file=%s' % self.class_test_theta_file,
                '-label_index=%d' % self.label_index,
                '-decoder_files=%s' % ','.join(self.decoder_files),
                '-latent_space_size=%s' % self.class_latent_space_size,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-accuracy_file=%s' % accuracy_file,
                '-batch_size=%d' % self.batch_size,
                '-pre_pca=40',
                '-n_nearest_neighbors=50',
                '-n_fit=240000',
                '-plot_directory=%s' % plot_directory,
                '-n_pca=10',
                '-max_samples=10000'
            ] + self.decoder_parameters()

            #detect_attack_classifier_nn = DetectAttackLearnedDecoderClassifierNN(arguments)
            #detect_attack_classifier_nn.main()

        if not os.path.exists(os.path.join(plot_directory, 'nn_50_10')) or self.args.revisualize:
            arguments = [
                '-mode=nn',
                '-train_images_file=%s' % self.train_images_file,
                '-test_images_file=%s' % self.test_images_file,
                '-train_codes_file=%s' % self.train_codes_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-test_theta_file=%s' % self.class_test_theta_file,
                '-label_index=%d' % self.label_index,
                '-decoder_files=%s' % ','.join(self.decoder_files),
                '-latent_space_size=%s' % self.class_latent_space_size,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-accuracy_file=%s' % accuracy_file,
                '-batch_size=%d' % self.batch_size,
                '-pre_pca=40',
                '-n_nearest_neighbors=50',
                '-n_fit=240000',
                '-plot_directory=%s' % plot_directory,
                '-n_pca=10',
                '-max_samples=10000'
            ] + self.decoder_parameters()

            detect_attack_classifier_nn = DetectAttackLearnedDecoderClassifierNN(arguments)
            detect_attack_classifier_nn.main()

        if not os.path.exists(os.path.join(plot_directory, 'inclusive_nn_50_10')) or self.args.revisualize:
            arguments = [
                '-mode=inclusive_nn',
                '-train_images_file=%s' % self.train_images_file,
                '-test_images_file=%s' % self.test_images_file,
                '-train_codes_file=%s' % self.train_codes_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-test_theta_file=%s' % self.class_test_theta_file,
                '-label_index=%d' % self.label_index,
                '-decoder_files=%s' % ','.join(self.decoder_files),
                '-latent_space_size=%s' % self.class_latent_space_size,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-accuracy_file=%s' % accuracy_file,
                '-batch_size=%d' % self.batch_size,
                '-pre_pca=40',
                '-n_nearest_neighbors=50',
                '-n_fit=240000',
                '-plot_directory=%s' % plot_directory,
                '-n_pca=10',
                '-max_samples=10000'
            ] + self.decoder_parameters()

            detect_attack_classifier_nn = DetectAttackLearnedDecoderClassifierNN(arguments)
            detect_attack_classifier_nn.main()

    def visualize_learned_on_data_manifold(self, model_name, t=0, a=0):
        """
        Visualize on manifold.
        """

        # Guard
        if not utils.display():
            return

        training = self.training_options[t]
        attack = self.learned_on_data_attack_options[a]
        assert isinstance(attack, LearnedOnDataAttackOptions)

        model_directory = '%s_%d_%d' % (model_name, training.model, training.training_samples)
        base_directory = self.attack_directory(model_directory, attack)
        assert os.path.exists(paths.experiment_dir(model_directory)), model_directory

        classifier_file = paths.experiment_file('%s/classifier' % model_directory, ext='.pth.tar')
        perturbations_file = paths.results_file('%s/perturbations' % base_directory)
        success_file = paths.results_file('%s/success' % base_directory)
        accuracy_file = paths.results_file('%s/accuracy' % base_directory)
        output_directory = paths.experiment_dir('%s/perturbations/' % base_directory)

        if not os.path.exists(output_directory) or self.args.revisualize:

            # Remove if already exists!
            if self.args.revisualize and os.path.exists(output_directory):
                shutil.rmtree(output_directory)

            arguments = [
                '-test_images_file=%s' % self.test_images_file,
                '-test_theta_file=%s' % self.data_test_theta_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-decoder_files=%s' % self.decoder_file,
                '-classifier_file=%s' % classifier_file,
                '-perturbations_file=%s' % perturbations_file,
                '-success_file=%s' % success_file,
                '-accuracy_file=%s' % accuracy_file,
                '-output_directory=%s' % output_directory,
                '-batch_size=%d' % self.batch_size,
                '-latent_space_size=%d' % self.data_latent_space_size
            ] + self.network_parameters() + self.decoder_parameters()

            visualize_attack_classifier = VisualizeAttackLearnedDecoderClassifier(arguments)
            visualize_attack_classifier.main()
        else:
            log('[Experiment] %s exists' % output_directory)

        return success_file

    def detect_learned_on_data_manifold(self, model_name, t=0, a=0):
        """
        Visualize on manifold.
        """

        raise NotImplementedError()

    def run(self):
        """
        Run.
        """

        raise NotImplementedError()

    def evaluate(self, models):
        """
        Evaluate.
        """

        raise NotImplementedError()

    def visualize(self, models):
        """
        Visualize.
        """

        raise NotImplementedError()

    def main(self):
        """
        Main.
        """

        for training_options in self.training_options:
            training_options.model = self.args.start_model

        self.validate()
        if self.args.mode == 'run':
            self.run()
        elif self.args.mode == 'evaluate':
            models = self.run()
            self.evaluate(models)
        elif self.args.mode == 'visualize':
            models = self.run()
            self.visualize(models)
        else:
            raise NotImplementedError()
