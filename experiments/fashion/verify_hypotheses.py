import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from experiments.experiment import *
from experiments.options import *


class VerifyHypotheses(Experiment):
    """
    Explore number of epochs for models.
    """

    def __init__(self, args=None):
        """
        Constructor, also sets options.
        """

        super(VerifyHypotheses, self).__init__(args)

        self.class_latent_space_size = 10
        """ (int) Latent space size. """

        self.data_latent_space_size = 10
        """ (int) Latent space size. """

        decoder_file = 'Manifolds/Fashion/vaegan2_%d_%d_abs_60_64_64_%g_%g_0_manual/decoder'
        encoder_file = 'Manifolds/Fashion/vaegan2_%d_%d_abs_60_64_64_%g_%g_0_manual/encoder'

        self.betas = [
            2.75,
            2.75,
            2.75,
            2.75,
            2.75,
            2.75,
            2.75,
            2.75,
            2.75,
            2.75,
            3,
        ]
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

        self.manifold_directory = 'Manifolds/Fashion/vaegan2'
        """ (str) Manifold directory. """

        self.label_index = 0
        """ (int) Label index. """

        self.train_images_file = paths.fashion_train_images_file()
        """ (str) Train images file. """

        self.test_images_file = paths.fashion_test_images_file()
        """ (str) Test images file. """

        self.train_codes_file = paths.fashion_train_labels_file()
        """ (str) Train codes file. """

        self.test_codes_file = paths.fashion_test_labels_file()
        """ (str) Test codes file. """

        self.max_iterations = 40
        """ (int) Global number of iterations. """

        self.off_training_epsilon = 0.3
        """ (float) Epsilon for training. """

        self.on_training_epsilon = 0.3
        """ (float) Epsilon for training. """

        self.on_data_training_epsilon = 0.1
        """ (float) Epsilon for training. """

        self.stn_N_theta = 6
        self.stn_translation = '-0.15,0.15'
        self.stn_shear = '-0.2,0.2'
        self.stn_scale = '0.85,1.15'
        self.stn_rotation = '%g,%g' % (-math.pi / 10, math.pi / 10)

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
        self.training_options = [TrainingOptions(training_size, 20) for training_size in training_sizes]
        """ ([TrainingOptions]) Training options. """

        self.off_attack_options = []
        """ ([OffAttackOptions]) Attack options. """

        self.off_training_options = OffAttackMadryLInfFullIterationOptions(self.off_training_epsilon, self.max_iterations)
        """ (OffAttackOptions) Taining options. """

        self.on_attack_options = []
        """ ([OnAttackOptions]) Attack options. """

        self.on_training_options = OnAttackMadryLInfFullIterationOptions(self.on_training_epsilon, self.max_iterations)
        """ (OnAttackOptions) Training options. """

        self.learned_on_class_attack_options = []
        """ ([LearnedOnClassAttackOptions]) Attack options. """

        self.learned_on_class_training_options = LearnedOnClassAttackMadryLInfFullIterationOptions(self.on_training_epsilon, self.max_iterations)
        """ (LearnedOnClassAttackOptions) Training options. """

        self.stn_augmentation_options = STNAugmentationLInfOptions(self.on_training_epsilon, 1, self.stn_N_theta, self.stn_translation, self.stn_shear, self.stn_scale, self.stn_rotation)
        """ ([STNAugmentationOptions]) Augmentation options. """

        self.stn_attack_options = []
        """ ([STNAttackOptions]) Attack options. """

        self.stn_training_options = STNAttackMadryLInfFullIterationOptions(self.on_training_epsilon, self.max_iterations, self.stn_N_theta, self.stn_translation, self.stn_shear, self.stn_scale, self.stn_rotation)
        """ (STNAttackOptions) Training options. """

        self.learned_on_data_attack_options = []
        """ ([LearnedOnClassAttackOptions]) Attack options. """

        self.learned_on_data_training_options = LearnedOnDataAttackMadryLInfFullIterationOptions(self.on_data_training_epsilon, self.max_iterations)
        """ (LearnedOnClassAttackOptions) Training options. """

        for epsilon in self.off_attack_epsilons:
            self.off_attack_options += [
                OffAttackMadryLInfOptions(epsilon, self.max_iterations),
            ]

            if self.args.strong:
                self.off_attack_options.append(OffAttackMadryLInfOptions(epsilon, 2*self.max_iterations))

        for epsilon in self.on_attack_epsilons:
            self.on_attack_options += [
                OnAttackMadryLInfOptions(epsilon, self.max_iterations),
            ]

            if self.args.strong:
                self.on_attack_options.append(OnAttackMadryLInfOptions(epsilon, 2*self.max_iterations))

            self.learned_on_class_attack_options += [
                LearnedOnClassAttackMadryLInfOptions(epsilon, self.max_iterations),
            ]

            if self.args.strong:
                self.learned_on_class_attack_options.append(LearnedOnClassAttackMadryLInfOptions(epsilon, 2*self.max_iterations))

            self.stn_attack_options += [
                STNAttackMadryLInfOptions(epsilon, self.max_iterations, self.stn_N_theta, self.stn_translation, self.stn_shear, self.stn_scale, self.stn_rotation),
            ]

            if self.args.strong:
                self.stn_attack_options.append(STNAttackMadryLInfOptions(epsilon, 2*self.max_iterations, self.stn_N_theta, self.stn_translation, self.stn_shear, self.stn_scale, self.stn_rotation))

        for epsilon in self.on_data_attack_epsilons:
            self.learned_on_data_attack_options += [
                LearnedOnDataAttackMadryLInfOptions(epsilon, self.max_iterations),
            ]

            if self.args.strong:
                self.learned_on_data_attack_options.append(LearnedOnDataAttackMadryLInfOptions(epsilon, 2*self.max_iterations))

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = super(VerifyHypotheses, self).get_parser()
        parser.add_argument('-transfer', default=False, action='store_true', help='Transfer attacks.')
        parser.add_argument('-learned_on_data', default=False, action='store_true', help='Learned on data attacks.')
        parser.add_argument('-transfer_model_directory', default='', help='Transfer model directory.', type=str)
        parser.add_argument('-strong', default='', help='Strong attack.', type=str)

        return parser

    def experiment(self):
        """
        Experiment.
        """

        return 'VerifyHypotheses/Fashion/'

    def run(self):
        """
        Run.
        """

        self.compute_class_theta()
        if self.args.learned_on_data:
            self.compute_data_theta()

        models = []
        for m in range(self.args.max_models - self.args.start_model):
            for t in range(len(self.training_options)):
                models = []
                models.append(self.train_normal(t))

                models.append(self.train_off_manifold_adversarial(t, self.off_training_options))

                models.append(self.train_regular_augmented(t, self.stn_augmentation_options))
                models.append(self.train_adversarial_augmented(t, self.stn_training_options))

                models.append(self.train_learned_on_class_manifold_adversarial(t, self.learned_on_class_training_options))

                if self.args.learned_on_data:
                    models.append(self.train_learned_on_data_manifold_adversarial(t, self.learned_on_data_training_options))

                for model in models:
                    for a in range(len(self.off_attack_options)):
                        self.attack_off_manifold(model, t, a)
                for model in models:
                    for a in range(len(self.learned_on_class_attack_options)):
                        self.attack_learned_on_class_manifold(model, t, a)

                if self.args.learned_on_data:
                    for model in models:
                        for a in range(len(self.learned_on_data_attack_options)):
                            self.attack_learned_on_data_manifold(model, t, a)

                self.training_options[t].model += 1

        if self.args.transfer:
            assert self.args.transfer_model_directory

            for t in range(len(self.training_options)):
                self.training_options[t].model = self.args.start_model

            for m in range(self.args.max_models - self.args.start_model):
                for t in range(len(self.training_options)):
                    models = ['normal']
                    models.append(self.model_name('off_manifold_adversarial', self.training_options[t], self.off_training_options))
                    models.append(self.model_name('regular_augmented', self.training_options[t], self.stn_augmentation_options))
                    models.append(self.model_name('adversarial_augmented', self.training_options[t], self.stn_training_options))
                    models.append(self.model_name('learned_on_class_manifold_adversarial', self.training_options[t], self.learned_on_class_training_options))

                    if self.args.transfer:
                        for model in models:
                            for a in range(len(self.off_attack_options)):
                                self.attack_off_manifold_transfer(self.args.transfer_model_directory, model, t, a)
                    if self.args.transfer:
                        for model in models:
                            for a in range(len(self.learned_on_class_attack_options)):
                                self.attack_learned_on_class_manifold_transfer(self.args.transfer_model_directory, model, t, a)

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
            self.statistics[model + '_off'] = numpy.mean(self.results[model + '_off'], axis=1)
            keys.append('off')
            self.statistics[model + '_learned_on_class'] = numpy.mean(self.results[model + '_learned_on_class'], axis=1)
            keys.append('learned_on_class')
            if self.args.transfer:
                ignore = 0
                if model == 'normal':
                    ignore = 1

                self.statistics[model + '_off_transfer'] = numpy.mean(self.results[model + '_off_transfer'][:, ignore:, :, :, :], axis=1)
                keys.append('off_transfer')
                self.statistics[model + '_learned_on_class_transfer'] = numpy.mean(self.results[model + '_learned_on_class_transfer'][:, ignore:, :, :, :], axis=1)
                keys.append('learned_on_class_transfer')

        self.plots(models, keys)

    def plots(self, models, keys):
        """
        Plots.
        """

        labels = ['Normal', 'OffAdvTrain', 'STNAugm', 'STNAdvTrain', 'OnClassAdvTrain']
        norms = [1, 2, float('inf')]
        attacks = keys

        for model in models:
            log('[Experiment] %s' % model)
            for attack in attacks:
                n = 2
                errors = self.results[model][:, :, 1].flatten()
                distances = self.results['%s_%s' % (model, attack)][:, :, 0, n, 2].flatten()
                successes = self.results['%s_%s' % (model, attack)][:, :, 0, 0, 0].flatten()

                distance_slope, _, distance_correlation, distance_p_value, _ = scipy.stats.linregress(errors, distances)
                success_slope, _, success_correlation, success_p_value, _ = scipy.stats.linregress(errors, successes)

                log('[Experiment]     %s: distance=%g (corr=%g, p=%g) success=%g (corr=%g, p=%g)' % (
                    attack, distance_slope, distance_correlation, distance_p_value, success_slope, success_correlation, success_p_value
                ))

        # Standard error
        x = numpy.stack([self.statistics[model][:, 0] for model in models], axis=1).T
        y = numpy.stack([self.statistics[model][:, 1] for model in models], axis=1).T
        plot_file = paths.image_file('0_evaluation/plot_error')
        plot.line(plot_file, x, y, labels)
        latex_file = paths.latex_file('0_evaluation/latex_error')
        latex.line(latex_file, x, y, labels)

        for attack in attacks:
            # Success rate
            x = numpy.stack([self.statistics[model][:, 0] for model in models], axis=1).T
            y = numpy.stack([self.statistics['%s_%s' % (model, attack)][:, 0, 0, 0] for model in models], axis=1).T
            plot_file = paths.image_file('0_evaluation/plot_%s_success_rate' % attack)
            plot.line(plot_file, x, y, labels)
            latex_file = paths.latex_file('0_evaluation/latex_%s_success_rate' % attack)
            latex.line(latex_file, x, y, labels)

            # L_inf Distances Off-Manifold
            n = 2
            x = numpy.stack([self.statistics[model][:, 0] for model in models], axis=1).T
            y = numpy.stack([self.statistics['%s_%s' % (model, attack)][:, 0, n, 2] for model in models], axis=1).T
            plot_file = paths.image_file('0_evaluation/plot_%s_distance_%g' % (attack, norms[n]))
            plot.line(plot_file, x, y, labels)
            latex_file = paths.latex_file('0_evaluation/latex_%s_distance_%g' % (attack, norms[n]))
            latex.line(latex_file, x, y, labels)

            # Error and success rate
            c = numpy.stack([self.results[model][:, :, 0].flatten() for model in models], axis=1).T
            x = numpy.stack([self.results[model][:, :, 1].flatten() for model in models], axis=1).T
            y = numpy.stack([self.results['%s_%s' % (model, attack)][:, :, 0, 0, 0].flatten() for model in models], axis=1).T
            plot_file = paths.image_file('0_evaluation/plot_%s_error_success_rate' % attack)
            plot.scatter2(plot_file, x, y, labels)
            latex_file = paths.latex_file('0_evaluation/latex_%s_error_success_rate' % attack)
            latex.scatter2(latex_file, x, y, labels, c)

            # Error and success rate as line
            c = numpy.stack([self.statistics[model][:, 0] for model in models], axis=1).T
            x = numpy.stack([self.statistics[model][:, 1] for model in models], axis=1).T
            y = numpy.stack([self.statistics['%s_%s' % (model, attack)][:, 0, 0, 0].flatten() for model in models], axis=1).T
            plot_file = paths.image_file('0_evaluation/plot_%s_error_success_rate_line' % attack)
            plot.line(plot_file, x, y, labels)
            latex_file = paths.latex_file('0_evaluation/latex_%s_error_success_rate_line' % attack)
            latex.line(latex_file, x, y, labels, c)

    def visualize(self, models):
        """
        Detect.
        """

        assert len(self.training_options) == 1

        for t in range(len(self.training_options)):
            self.training_options[t].model = 0

        for m in [0, 1, 4]:
            success_file = self.visualize_learned_on_class_manifold(models[m])
            self.visualize_off_manifold(models[m], success_file)

        for m in [0, 1, 4]:
            log('[Experiment] visualized %s' % models[m])

        self.detect_off_manifold(models[0])
        self.detect_learned_on_class_manifold(models[0])


if __name__ == '__main__':
    program = VerifyHypotheses()
    program.main()