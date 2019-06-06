import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from training.train_vae_gan2 import *
from training.train_vae_gan3 import *
from training.train_vae_gan4 import *
from training.train_vae_gan5 import *
from training.train_vae_gan6 import *
from training.test_variational_auto_encoder import *
from training.train_variational_auto_encoder import *
from tools.visualize_mosaic import *
from tools.visualize_individual import *
from common import paths


class LearnManifolds:
    """
    Test classifier for celebA.
    """

    def __init__(self, args=None):
        """
        Constructor.
        """

        paths.set_globals(experiment=self.experiment())
        paths.set_globals(experiment=self.experiment())
        self.train_images_file = paths.fashion_train_images_file()
        self.train_codes_file = paths.fashion_train_labels_file()
        self.test_images_file = paths.fashion_test_images_file()
        self.test_codes_file = paths.fashion_test_labels_file()
        self.label_index = 0
        self.results = dict()

        self.args = None
        """ Arguments of program. """

        parser = self.get_parser()
        if args is not None:
            self.args = parser.parse_args(args)
        else:
            self.args = parser.parse_args()

        # self.betas = [
        #     2.75,
        #     2.75,
        #     2.75,
        #     2.75,
        #     2.75,
        #     2.75,
        #     2.75,
        #     2.75,
        #     2.75,
        #     2.75,
        #     2.75,
        # ]
        #
        # self.gammas = [
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        # ]

        # self.classifier_channels = 32
        # self.network_channels = 128

        self.classifier_channels = 64
        self.network_channels = 64

        self.training_parameters = [
            '-base_lr=0.005',
            '-weight_decay=0.0001',
            '-base_lr_decay=0.9',
            '-batch_size=100',
            '-absolute_error',
        ]

        self.classifier_parameters = [
            '-classifier_architecture=standard',
            '-classifier_activation=relu',
            '-classifier_channels=%d' % self.classifier_channels,
            '-classifier_units=1024,1024,1024,1024'
        ]

        self.network_parameters = [
            '-network_architecture=standard',
            '-network_activation=relu',
            '-network_channels=%d' % self.network_channels,
            '-network_units=1024,1024,1024,1024',
        ]

        log('-- ' + self.__class__.__name__)
        for key in vars(self.args):
            log('[Experiment] %s=%s' % (key, str(getattr(self.args, key))))

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Test generalization methods.')
        parser.add_argument('-method', help='Manifold learning method.', type=str)
        parser.add_argument('-label', help='Label to learn.', type=str)
        parser.add_argument('-reevaluate', dest='reevaluate', action='store_true')
        parser.add_argument('-latent_space_size', default=10, help='Latent space size.', type=int)
        parser.add_argument('-epochs', default=60, help='Training epochs.', type=int)
        parser.add_argument('-eta', default=0.0, help='Training epochs.', type=float)
        parser.add_argument('-beta', default=2.75, help='Training epochs.', type=float)
        parser.add_argument('-gamma', default=1, help='Training epochs.', type=float)

        return parser

    def experiment(self):
        """
        Experiment.
        """

        return 'Manifolds/Fashion/'

    def train(self, base_directory, method, label):
        """
        Train VAE-GAN.
        """

        encoder_file = paths.state_file('%s/encoder' % base_directory)
        decoder_file = paths.state_file('%s/decoder' % base_directory)
        classifier_file = paths.state_file('%s/classifier' % base_directory)
        reconstruction_file = paths.results_file('%s/reconstructions' % base_directory)
        interpolation_file = paths.results_file('%s/interpolation' % base_directory)
        random_file = paths.results_file('%s/random' % base_directory)
        log_file = paths.log_file('%s/vaegan' % base_directory)
        results_file = paths.pickle_file('%s/results' % base_directory)
        training_file = paths.results_file('%s/training' % base_directory)
        testing_file = paths.results_file('%s/testing' % base_directory)
        error_file = paths.image_file('%s/error' % base_directory)
        gradient_file = paths.image_file('%s/gradient' % base_directory)
        test_results_file = paths.pickle_file('%s/test_results' % base_directory)

        if not os.path.exists(encoder_file) or not os.path.exists(decoder_file):
            if method.find('vaegan') >= 0:
                arguments = [
                    '-train_images_file=%s' % self.train_images_file,
                    '-train_codes_file=%s' % self.train_codes_file,
                    '-test_images_file=%s' % self.test_images_file,
                    '-test_codes_file=%s' % self.test_codes_file,
                    '-label_index=%d' % self.label_index,
                    '-label=%d' % label,
                    '-encoder_file=%s' % encoder_file,
                    '-decoder_file=%s' % decoder_file,
                    '-classifier_file=%s' % classifier_file,
                    '-reconstruction_file=%s' % reconstruction_file,
                    '-interpolation_file=%s' % interpolation_file,
                    '-random_file=%s' % random_file,
                    '-log_file=%s' % log_file,
                    '-results_file=%s' % results_file,
                    '-training_file=%s' % training_file,
                    '-testing_file=%s' % testing_file,
                    '-error_file=%s' % error_file,
                    '-gradient_file=%s' % gradient_file,
                    '-latent_space_size=%d' % self.args.latent_space_size,
                    '-beta=%g' % self.args.beta,
                    '-gamma=%g' % self.args.gamma,
                    '-epochs=%d' % self.args.epochs,
                    '-eta=%g' % self.args.eta,
                ] + self.training_parameters + self.network_parameters + self.classifier_parameters
                log(arguments)
                if method =='vaegan2':
                    train = TrainVAEGAN2(arguments)
                elif method == 'vaegan3':
                    train = TrainVAEGAN3(arguments)
                elif method == 'vaegan4':
                    train = TrainVAEGAN4(arguments)
                elif method == 'vaegan5':
                    train = TrainVAEGAN5(arguments)
                elif method == 'vaegan6':
                    train = TrainVAEGAN6(arguments)
                else:
                    raise NotImplementedError()
                train.main()
            elif method == 'vae':
                arguments = [
                    '-train_images_file=%s' % self.train_images_file,
                    '-train_codes_file=%s' % self.train_codes_file,
                    '-test_images_file=%s' % self.test_images_file,
                    '-test_codes_file=%s' % self.test_codes_file,
                    '-label_index=%d' % self.label_index,
                    '-label=%d' % label,
                    '-encoder_file=%s' % encoder_file,
                    '-decoder_file=%s' % decoder_file,
                    '-reconstruction_file=%s' % reconstruction_file,
                    '-interpolation_file=%s' % interpolation_file,
                    '-random_file=%s' % random_file,
                    '-log_file=%s' % log_file,
                    '-results_file=%s' % results_file,
                    '-training_file=%s' % training_file,
                    '-testing_file=%s' % testing_file,
                    '-error_file=%s' % error_file,
                    '-latent_space_size=%d' % self.args.latent_space_size,
                    '-beta=%g' % self.args.beta,
                    '-epochs=%d' % self.args.epochs,
                ] + self.training_parameters + self.network_parameters
                train = TrainVariationalAutoEncoder(arguments)
                train.main()
            else:
                raise NotImplementedError()

        train_theta_file = paths.results_file('%s/train_theta' % base_directory)
        test_theta_file = paths.results_file('%s/test_theta' % base_directory)

        if self.args.reevaluate or not os.path.exists(test_results_file):
            test = TestVariationalAutoEncoder([
                '-train_images_file=%s' % self.train_images_file,
                '-test_images_file=%s' % self.test_images_file,
                '-train_codes_file=%s' % self.train_codes_file,
                '-test_codes_file=%s' % self.test_codes_file,
                '-train_theta_file=%s' % train_theta_file,
                '-test_theta_file=%s' % test_theta_file,
                '-label_index=%d' % self.label_index,
                '-label=%d' % label,
                '-encoder_file=%s' % encoder_file,
                '-decoder_file=%s' % decoder_file,
                '-batch_size=256',
                '-results_file=%s' % test_results_file,
                '-reconstruction_file=',
                '-random_file=',
                '-interpolation_file=',
                '-output_directory=',
                '-latent_space_size=%d' % self.args.latent_space_size,
            ] + self.network_parameters)
            test.main()
            results = test.results
        else:
            results = utils.read_pickle(test_results_file)
        self.results[base_directory] = (results['reconstruction_error'], results['code_mean'], results['code_var'])

        if utils.display():
            random_directory = paths.experiment_dir('%s/random/' % base_directory, experiment=self.experiment())
            if self.args.reevaluate or not os.path.exists(random_directory):
                visualize_mosaic = VisualizeMosaic([
                    '-images_file=%s' % random_file,
                    '-output_directory=%s' % random_directory,
                ])
                visualize_mosaic.main()

            random_directory = paths.experiment_dir('%s/random_/' % base_directory, experiment=self.experiment())
            # if self.args.reevaluate or not os.path.exists(random_directory):
            visualize_individual = VisualizeIndividual([
                '-images_file=%s' % random_file,
                '-output_directory=%s' % random_directory,
            ])
            visualize_individual.main()

        if utils.display():
            reconstruction_directory = paths.experiment_dir('%s/reconstruction/' % base_directory, experiment=self.experiment())
            if self.args.reevaluate or not os.path.exists(reconstruction_directory):
                visualize_mosaic = VisualizeMosaic([
                    '-images_file=%s' % reconstruction_file,
                    '-output_directory=%s' % reconstruction_directory,
                ])
                visualize_mosaic.main()

            reconstruction_directory = paths.experiment_dir('%s/reconstruction_/' % base_directory, experiment=self.experiment())
            # if self.args.reevaluate or not os.path.exists(reconstruction_directory):
            visualize_individual = VisualizeIndividual([
                '-images_file=%s' % reconstruction_file,
                '-output_directory=%s' % reconstruction_directory,
            ])
            visualize_individual.main()

        if utils.display():
            interpolation_directory = paths.experiment_dir('%s/interpolation/' % base_directory, experiment=self.experiment())
            if self.args.reevaluate or not os.path.exists(interpolation_directory):
                visualize_mosaic = VisualizeMosaic([
                    '-images_file=%s' % interpolation_file,
                    '-output_directory=%s' % interpolation_directory,
                ])
                visualize_mosaic.main()

            interpolation_directory = paths.experiment_dir('%s/interpolation_/' % base_directory, experiment=self.experiment())
            # if self.args.reevaluate or not os.path.exists(reconstruction_directory):
            visualize_individual = VisualizeIndividual([
                '-images_file=%s' % interpolation_file,
                '-output_directory=%s' % interpolation_directory,
            ])
            visualize_individual.main()

        if utils.display():
            original_directory = paths.experiment_dir('%s/original/' % base_directory, experiment=self.experiment())
            if self.args.reevaluate or not os.path.exists(original_directory):
                visualize_mosaic = VisualizeMosaic([
                    '-images_file=%s' % self.test_images_file,
                    '-codes_file=%s' % self.test_codes_file,
                    '-label_index=%d' % self.label_index,
                    '-label=%d' % label,
                    '-output_directory=%s' % original_directory,
                ])
                visualize_mosaic.main()

            original_directory = paths.experiment_dir('%s/original_/' % base_directory, experiment=self.experiment())
            # if self.args.reevaluate or not os.path.exists(reconstruction_directory):
            visualize_individual = VisualizeIndividual([
                '-images_file=%s' % self.test_images_file,
                '-codes_file=%s' % self.test_codes_file,
                '-label_index=%d' % self.label_index,
                '-label=%d' % label,
                '-output_directory=%s' % original_directory,
            ])
            visualize_individual.main()

    def main(self):
        """
        Learn manifolds.
        """

        labels = map(int, self.args.label.split(','))

        for label in labels:
            self.train('%s_%d_%d_abs_%d_%d_%d_%g_%g_%g_manual/' % (self.args.method, self.args.latent_space_size, label, self.args.epochs, self.network_channels, self.classifier_channels, self.args.beta, self.args.gamma, self.args.eta), self.args.method, label)


if __name__ == '__main__':
    program = LearnManifolds()
    program.main()