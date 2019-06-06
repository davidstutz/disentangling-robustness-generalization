import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from experiments.experiment import *
from experiments.options import *
if utils.display():
    from common import plot
    from common import latex
    import scipy.stats


class VerifyLongResNetHypotheses(Experiment):
    """
    Explore number of epochs for models.
    """

    def __init__(self, args=None):
        """
        Constructor, also sets options.
        """

        super(VerifyLongResNetHypotheses, self).__init__(args)

        assert self.args.suffix in ['Hard', 'Moderate', 'Easy']

        self.class_latent_space_size = 10
        """ (int) Latent space size. """

        self.data_latent_space_size = 10
        """ (int) Latent space size. """

        decoder_file = 'Manifolds/Fonts/vaegan2_%d_%d_abs_10_64_64_%g_%g_0_manual_' + self.args.suffix + '/decoder'
        encoder_file = 'Manifolds/Fonts/vaegan2_%d_%d_abs_10_64_64_%g_%g_0_manual_' + self.args.suffix + '/encoder'

        self.betas = [3]*11
        """ ([float]) Betas. """

        self.gammas = [1]*11
        """ ([float]) Gammas. """

        self.decoder_files = []
        """ ([str]) Decoder files for class manifolds. """

        for label in range(self.labels):
            self.decoder_files.append(
                paths.state_file(decoder_file % (self.class_latent_space_size, label, self.betas[label], self.gammas[label]), experiment=''))

        self.decoder_file = paths.state_file(decoder_file % (self.data_latent_space_size, -1, self.betas[-1], self.gammas[-1]), experiment='')
        """ (str) Decoder file for data manifold. """

        self.encoder_files = []
        """ ([str]) Decoder files for class manifolds. """

        for label in range(self.labels):
            self.encoder_files.append(
                paths.state_file(encoder_file % (self.class_latent_space_size, label, self.betas[label], self.gammas[label]), experiment=''))

        self.encoder_file = paths.state_file(encoder_file % (self.data_latent_space_size, -1, self.betas[-1], self.gammas[-1]), experiment='')
        """ (str) Decoder file for data manifold. """

        self.manifold_directory = 'Manifolds/Fonts/vaegan2_' + self.args.suffix
        """ (str) Manifold directory. """

        paths.set_globals(characters='ABCDEFGHIJ', fonts=1000, transformations=6, size=28, suffix=self.args.suffix)
        assert 'fonts' in paths.get_globals().keys() and paths.get_globals()['fonts'] == 1000
        assert 'characters' in paths.get_globals().keys() and paths.get_globals()['characters'] == 'ABCDEFGHIJ'
        assert 'transformations' in paths.get_globals().keys() and paths.get_globals()['transformations'] == 6
        assert 'size' in paths.get_globals().keys() and paths.get_globals()['size'] == 28
        assert 'suffix' in paths.get_globals().keys() and paths.get_globals()['suffix'] == self.args.suffix

        self.database_file = paths.database_file()
        """ (str) Database file. """

        self.train_images_file = paths.train_images_file()
        """ (str) Train images file. """

        self.test_images_file = paths.test_images_file()
        """ (str) Test images file. """

        self.train_codes_file = paths.train_codes_file()
        """ (str) Train codes file. """

        self.test_codes_file = paths.test_codes_file()
        """ (str) Test codes file. """

        self.train_theta_file = paths.train_theta_file()
        """ (str) Train theta file. """

        self.test_theta_file = paths.test_theta_file()
        """ (str) Test theta file. """

        self.max_iterations = 40
        """ (int) Global number of iterations. """

        self.off_training_epsilon = 0.3
        """ (float) Epsilon for training. """

        self.on_training_epsilon = 0.3
        """ (float) Epsilon for training. """

        self.on_data_training_epsilon = 0.1
        """ (float) Epsilon for training. """

        if self.args.suffix == 'Easy':
            self.stn_N_theta = 6
            self.stn_translation = '-0.1,0.1'
            self.stn_shear = '-0.2,0.2'
            self.stn_scale = '0.9,1.1'
            self.stn_rotation = '%g,%g' % (-math.pi / 6, math.pi / 6)
        elif self.args.suffix == 'Moderate':
            self.stn_N_theta = 6
            self.stn_translation = '-0.2,0.2'
            self.stn_shear = '-0.4,0.4'
            self.stn_scale = '0.85,1.15'
            self.stn_rotation = '%g,%g' % (-2 * math.pi / 6, 2 * math.pi / 6)
        elif self.args.suffix == 'Hard':
            self.stn_N_theta = 6
            self.stn_translation = '-0.2,0.2'
            self.stn_shear = '-0.5,0.5'
            self.stn_scale = '0.75,1.15'
            self.stn_rotation = '%g,%g' % (-3 * math.pi / 6, 3 * math.pi / 6)
        else:
            raise NotImplementedError()

        assert self.stn_N_theta is not None
        assert self.stn_translation is not None
        assert self.stn_shear is not None
        assert self.stn_scale is not None
        assert self.stn_rotation is not None

        self.max_iterations = 40
        """ (int) Global number of iterations. """

        self.off_attack_epsilons = [0.3]
        """ ([flaot]) Epsilons for attacking. """

        self.on_attack_epsilons = [0.3]
        """ ([flaot]) Epsilons for attacking. """

        self.on_data_attack_epsilons = [0.1]
        """ ([float]) Epsilons for attacking. """

        self.off_training_epsilon = 0.3
        """ (float) Epsilon for training. """

        self.on_training_epsilon = 0.3
        """ (float) Epsilon for training. """

        self.on_data_training_epsilon = 0.1
        """ (float) Epsilon for training. """

        assert self.args.training_sizes is not None
        training_sizes = list(map(int, self.args.training_sizes.split(',')))
        self.training_options = [TrainingOptions(training_size, 80) for training_size in training_sizes]
        """ ([TrainingOptions]) Training options. """

        self.off_attack_options = []
        """ ([OffAttackOptions]) Attack options. """

        self.off_training_options = OffAttackMadryLInfFullIterationOptions(self.off_training_epsilon, self.max_iterations)
        """ (OffAttackOptions) Taining options. """

        for epsilon in self.off_attack_epsilons:
            self.off_attack_options += [
                OffAttackMadryLInfOptions(epsilon, self.max_iterations),
            ]

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = super(VerifyLongResNetHypotheses, self).get_parser()
        parser.add_argument('-suffix', default='Hard', help='Fonts dataset.', type=str)
        parser.add_argument('-normal', action='store_true', default=False, help='Normal training')

        return parser

    def network_parameters(self):
        """
        Get network parameters.
        """

        if self.network_architecture is None:
            self.network_architecture = [
                '-network_architecture=resnet',
                # '-network_no_batch_normalization',
                # '-network_dropout',
                # '-network_activation=relu',
                '-network_channels=64',
                '-network_units=2,2,2', # blocks for resnet14
            ]

        return self.network_architecture

    def experiment(self):
        """
        Experiment.
        """

        return 'VerifyLongResNetHypotheses/Fonts/%s/' % self.args.suffix

    def run(self):
        """
        Run.
        """

        models = []
        for m in range(self.args.max_models - self.args.start_model):
            for t in range(len(self.training_options)):
                models = []

                # Order is important!
                if self.args.normal:
                    models.append(self.train_normal(t))
                else:
                    models.append(self.train_off_manifold_adversarial(t, self.off_training_options))

                # for model in models:
                #    for a in range(len(self.off_attack_options)):
                #        self.attack_off_manifold(model, t, a)

                self.training_options[t].model += 1

        return models

    def evaluate(self, models):
        """
        Evaluation.
        """

        utils.makedir(paths.experiment_dir('0_evaluation'))

        for model in models:
            keys = []
            self.statistics[model] = numpy.mean(self.results[model], axis=1)

        self.plots(models, keys)

    def plots(self, models, keys):
        """
        Plots.
        """

        # Standard error
        x = numpy.stack([self.statistics[model][:, 0] for model in models], axis=1).T
        y = numpy.stack([self.statistics[model][:, 1] for model in models], axis=1).T
        print(x)
        print(y)


if __name__ == '__main__':
    program = VerifyLongResNetHypotheses()
    program.main()