import math
MAX_SAMPLES = 1000
MAX_ATTEMPTS = 5


class TrainingOptions:
    def __init__(self, training_samples, epochs=20):
        self.model = 0
        """ (int) Model id. """

        self.epochs = epochs
        """ (int) Number of epochs. """

        self.batch_size = 100
        """ (int) batch size. """

        self.validation_samples = 10000
        """ (int) Samples used for validation. """

        self.test_samples = 10000
        """ (int) Test samples. """

        self.training_samples = training_samples
        """ (int) Training samples. """

        self.weight_decay = 0.0001
        """ (float) Weight decay. """

        self.lr = 0.01
        """ (float) Learning rate. """

        self.lr_decay = 0.95
        """ (float) Learning rate decay. """

        self.suffix = None
        """ (str) Suffix. """


class OffAugmentationOptions:
    def __init__(self, norm, epsilon, max_iterations):
        assert isinstance(max_iterations, int), max_iterations
        assert norm in ['1', '2', 'inf'], norm

        self.norm = norm
        """ (str) Norm. """

        self.epsilon = epsilon
        """ (str) Epsilon. """

        self.max_iterations = max_iterations
        """ (int) Max iterations. """

        self.strong_variant = False
        "" "(bool) Strong variant. "

        self.suffix = None
        """ (str) Suffix. """


class OffAugmentationLInfOptions(OffAugmentationOptions):
    def __init__(self, epsilon, max_iterations):
        super(OffAugmentationLInfOptions, self).__init__('inf', epsilon, max_iterations)


class OffAugmentationL2Options(OffAugmentationOptions):
    def __init__(self, epsilon, max_iterations):
        super(OffAugmentationL2Options, self).__init__('2', epsilon, max_iterations)


# EMNIST FASHION FONTS
class OnAugmentationOptions:
    def __init__(self, norm, epsilon, max_iterations):
        assert isinstance(max_iterations, int), max_iterations
        assert norm in ['1', '2', 'inf'], norm
        assert epsilon > 0, epsilon

        self.norm = norm
        """ (str) Norm. """

        self.epsilon = epsilon
        """ (str) Epsilon. """

        self.max_iterations = max_iterations
        """ (int) Max iterations. """

        self.strong_variant = False
        """ (bool) Strong variant. """

        self.suffix = None
        """ (str) Suffix. """


class OnAugmentationLInfOptions(OnAugmentationOptions):
    def __init__(self, epsilon, max_iterations):
        super(OnAugmentationLInfOptions, self).__init__('inf', epsilon, max_iterations)


class OnAugmentationL2Options(OnAugmentationOptions):
    def __init__(self, epsilon, max_iterations):
        super(OnAugmentationL2Options, self).__init__('2', epsilon, max_iterations)


class LearnedOnClassAugmentationOptions:
    def __init__(self, norm, epsilon, max_iterations):
        assert isinstance(max_iterations, int), max_iterations
        assert norm in ['1', '2', 'inf'], norm
        assert epsilon > 0, epsilon
        self.norm = norm
        """ (str) Norm. """

        self.epsilon = epsilon
        """ (str) Epsilon. """

        self.max_iterations = max_iterations
        """ (int) Max iterations. """

        self.strong_variant = False
        """ (bool) Strong variant. """

        self.bound = 2
        """ (float) Bound. """

        self.suffix = None
        """ (str) Suffix. """


class LearnedOnClassAugmentationLInfOptions(LearnedOnClassAugmentationOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnClassAugmentationLInfOptions, self).__init__('inf', epsilon, max_iterations)


class LearnedOnClassAugmentationL2Options(LearnedOnClassAugmentationOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnClassAugmentationL2Options, self).__init__('2', epsilon, max_iterations)


class LearnedOnDataAugmentationOptions:
    def __init__(self, norm, epsilon, max_iterations):
        assert isinstance(max_iterations, int), max_iterations
        assert norm in ['1', '2', 'inf'], norm
        assert epsilon > 0, epsilon

        self.norm = norm
        """ (str) Norm. """

        self.epsilon = epsilon
        """ (str) Epsilon. """

        self.max_iterations = max_iterations
        """ (int) Max iterations. """

        self.strong_variant = False
        """ (bool) Strong variant. """

        self.bound = 2
        """ (float) Bound. """

        self.suffix = None
        """ (str) Suffix. """


class LearnedOnDataAugmentationLInfOptions(LearnedOnDataAugmentationOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnDataAugmentationLInfOptions, self).__init__('inf', epsilon, max_iterations)


class LearnedOnDataAugmentationL2Options(LearnedOnDataAugmentationOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnDataAugmentationL2Options, self).__init__('2', epsilon, max_iterations)


class STNAugmentationOptions:
    def __init__(self, norm, epsilon, max_iterations, N_theta=6, translation='-0.2,0.2', shear='-0.5,0.5', scale='0.9,1.1',
                 rotation='%g,%g' % (-math.pi / 4, math.pi / 4)):
        assert isinstance(max_iterations, int), max_iterations
        assert norm in ['1', '2', 'inf'], norm

        self.norm = norm
        """ (str) Norm. """

        self.epsilon = epsilon
        """ (str) Epsilon. """

        self.max_iterations = max_iterations
        """ (int) Max iterations. """

        self.strong_variant = False
        """ (bool) Strong variant. """

        self.N_theta = N_theta
        """ (int) Number of transformations. """

        self.translation_x = translation
        """ (str) Translation in x. """

        self.translation_y = translation
        """ (str) Translation in y. """

        self.shear_x = shear
        """ (str) Shear in x. """

        self.shear_y = shear
        """ (str) Shear in y. """

        self.scale = scale
        """ (str) Scale. """

        self.rotation = rotation
        """ (str) Rotation. """

        self.color = 0.5
        """ (float) Color. """

        self.suffix = None
        """ (str) Suffix. """


class STNAugmentationLInfOptions(STNAugmentationOptions):
    def __init__(self, epsilon, max_iterations, N_theta, translation, shear, scale, rotation):
        super(STNAugmentationLInfOptions, self).__init__('inf', epsilon, max_iterations, N_theta, translation, shear, scale, rotation)


class STNAugmentationL2Options(STNAugmentationOptions):
    def __init__(self, epsilon, max_iterations, N_theta, translation, shear, scale, rotation):
        super(STNAugmentationL2Options, self).__init__('2', epsilon, max_iterations, N_theta, translation, shear, scale, rotation)


class OffAttackOptions:
    def __init__(self, attack, objective, epsilon, max_iterations, c_0, training_mode, full=False):
        assert isinstance(max_iterations, int), max_iterations
        assert c_0 >= 0, c_0

        self.attack = attack
        """ (str) Attack name. """

        self.objective = objective
        """ (str) Objective name. """

        self.epsilon = epsilon
        """ (str) Epsilon. """

        self.max_iterations = max_iterations
        """ (int) Max iterations. """

        self.c_0 = c_0
        """ (float) Weight of norm. """

        self.c_1 = 0
        """ (float) Weight of bound. """

        self.c_2 = 1
        """ (float) Weight of objective. """

        self.max_projections = 5
        """ (int) Max projections. """

        self.base_lr = 0.005
        """ (float) Base learning rate. """

        self.max_samples = MAX_SAMPLES
        """ (int) Max samples. """

        self.max_attempts = MAX_ATTEMPTS
        """ (int) Max attempts. """

        self.training_mode = training_mode
        """ (bool) Training mode. """

        self.full = full
        """ (bool) Full variant. """

        self.suffix = None
        """ (str) Suffix. """


class OffAttackMadryLInfOptions(OffAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OffAttackMadryLInfOptions, self).__init__('UntargetedBatchLInfProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, False)


class OffAttackMadryLInfFullIterationOptions(OffAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OffAttackMadryLInfFullIterationOptions, self).__init__('UntargetedBatchLInfProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, True)


class OffAttackMadryLInfFullOptions(OffAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OffAttackMadryLInfFullOptions, self).__init__('UntargetedBatchLInfProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, True, True)


class OffAttackMadryUnconstrainedOptions(OffAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OffAttackMadryUnconstrainedOptions, self).__init__('UntargetedBatchL2ClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, False)


class OffAttackCWLInfOptions(OffAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OffAttackCWLInfOptions, self).__init__('UntargetedBatchLInfReparameterizedGradientDescent', 'UntargetedF6', epsilon, max_iterations, 1, False)


class OffAttackMadryL2Options(OffAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OffAttackMadryL2Options, self).__init__('UntargetedBatchL2ProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, False)


class OffAttackMadryL2FullIterationOptions(OffAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OffAttackMadryL2FullIterationOptions, self).__init__('UntargetedBatchL2ProjectedClippedGradientDescent', 'UntargetedF0', epsilon,  max_iterations, 0, True)


class OffAttackCWL2Options(OffAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OffAttackCWL2Options, self).__init__('UntargetedBatchL2ReparameterizedGradientDescent', 'UntargetedF6', epsilon, max_iterations, 1, False)


# EMNIST FASHION FONTS
class OnAttackOptions:
    def __init__(self, attack, objective, epsilon, max_iterations, c_0, training_mode, full=False):
        assert isinstance(max_iterations, int), max_iterations
        assert c_0 >= 0, c_0
        assert epsilon > 0

        self.attack = attack
        """ (str) Attack name. """

        self.objective = objective
        """ (str) Objective name. """

        self.epsilon = epsilon
        """ (str) Epsilon. """

        self.max_iterations = max_iterations
        """ (int) Max iterations. """

        self.c_0 = c_0
        """ (float) Weight of norm. """

        self.c_1 = 0
        """ (float) Weight of bound. """

        self.c_2 = 1
        """ (float) Weight of objective. """

        self.max_projections = 5
        """ (int) Max projections. """

        self.base_lr = 0.005
        """ (float) Base learning rate. """

        self.max_samples = int(2.5 * MAX_SAMPLES)
        """ (int) Max samples. """

        self.max_attempts = MAX_ATTEMPTS
        """ (int) Max attempts. """

        self.training_mode = training_mode
        """ (bool) Training mode. """

        self.full = full
        """ (bool) Full variant. """

        self.suffix = None
        """ (str) Suffix. """


class OnAttackMadryLInfOptions(OnAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OnAttackMadryLInfOptions, self).__init__('UntargetedBatchLInfProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, False)


class OnAttackMadryLInfFullIterationOptions(OnAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OnAttackMadryLInfFullIterationOptions, self).__init__('UntargetedBatchLInfProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, True)


class OnAttackMadryLInfFullOptions(OnAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OnAttackMadryLInfFullOptions, self).__init__('UntargetedBatchLInfProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, True, True)


class OnAttackMadryUnconstrainedOptions(OnAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OnAttackMadryUnconstrainedOptions, self).__init__('UntargetedBatchL2ClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, False)


class OnAttackCWLInfOptions(OnAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OnAttackCWLInfOptions, self).__init__('UntargetedBatchLInfReparameterizedGradientDescent', 'UntargetedF6', epsilon, max_iterations, 1, False)


class OnAttackMadryL2Options(OnAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OnAttackMadryL2Options, self).__init__('UntargetedBatchL2ProjectedClippedGradientDescent', 'UntargetedF0', epsilon,  max_iterations, 0, False)


class OnAttackMadryL2FullIterationOptions(OnAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OnAttackMadryL2FullIterationOptions, self).__init__('UntargetedBatchL2ProjectedClippedGradientDescent', 'UntargetedF0', epsilon,  max_iterations, 0, True)


class OnAttackCWL2Options(OnAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(OnAttackCWL2Options, self).__init__('UntargetedBatchL2ReparameterizedGradientDescent', 'UntargetedF6', epsilon,  max_iterations, 1, False)


class LearnedOnClassAttackOptions:
    def __init__(self, attack, objective, epsilon, max_iterations, c_0, training_mode, full=False):
        assert isinstance(max_iterations, int), max_iterations
        assert c_0 >= 0, c_0
        assert epsilon > 0

        self.attack = attack
        """ (str) Attack name. """

        self.objective = objective
        """ (str) Objective name. """

        self.epsilon = epsilon
        """ (str) Epsilon. """

        self.max_iterations = max_iterations
        """ (int) Max iterations. """

        self.c_0 = c_0
        """ (float) Weight of norm. """

        self.c_1 = 0
        """ (float) Weight of bound. """

        self.c_2 = 1
        """ (float) Weight of objective. """

        self.max_projections = 5
        """ (int) Max projections. """

        self.base_lr = 0.005
        """ (float) Base learning rate. """

        self.max_samples = int(2.5 * MAX_SAMPLES)
        """ (int) Max samples. """

        self.max_attempts = MAX_ATTEMPTS
        """ (int) Max attempts. """

        self.bound = 2
        """ (float) Bound. """

        self.training_mode = training_mode
        """ (bool)"""

        self.full = full
        """ (bool) Full variant. """

        self.suffix = None
        """ (str) Suffix. """


class LearnedOnClassAttackMadryLInfOptions(LearnedOnClassAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnClassAttackMadryLInfOptions, self).__init__('UntargetedBatchLInfProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, False)


class LearnedOnClassAttackMadryLInfFullIterationOptions(LearnedOnClassAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnClassAttackMadryLInfFullIterationOptions, self).__init__('UntargetedBatchLInfProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, True)


class LearnedOnClassAttackMadryLInfFullOptions(LearnedOnClassAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnClassAttackMadryLInfFullOptions, self).__init__('UntargetedBatchLInfProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, True, True)


class LearnedOnClassAttackMadryUnconstrainedOptions(LearnedOnClassAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnClassAttackMadryUnconstrainedOptions, self).__init__('UntargetedBatchL2ClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, False)


class LearnedOnClassAttackCWLInfOptions(LearnedOnClassAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnClassAttackCWLInfOptions, self).__init__('UntargetedBatchLInfReparameterizedGradientDescent', 'UntargetedF6', epsilon, max_iterations, 1, False)


class LearnedOnClassAttackMadryL2Options(LearnedOnClassAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnClassAttackMadryL2Options, self).__init__('UntargetedBatchL2ProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, False)


class LearnedOnClassAttackMadryL2FullIterationOptions(LearnedOnClassAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnClassAttackMadryL2FullIterationOptions, self).__init__('UntargetedBatchL2ProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, True)


class LearnedOnClassAttackCWL2Options(LearnedOnClassAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnClassAttackCWL2Options, self).__init__('UntargetedBatchL2ReparameterizedGradientDescent', 'UntargetedF6', epsilon, max_iterations, 1, False)


class LearnedOnDataAttackOptions:
    def __init__(self, attack, objective, epsilon, max_iterations, c_0, training_mode, full=False):
        assert isinstance(max_iterations, int), max_iterations
        assert c_0 >= 0, c_0
        assert epsilon > 0

        self.attack = attack
        """ (str) Attack name. """

        self.objective = objective
        """ (str) Objective name. """

        self.epsilon = epsilon
        """ (str) Epsilon. """

        self.max_iterations = max_iterations
        """ (int) Max iterations. """

        self.c_0 = c_0
        """ (float) Weight of norm. """

        self.c_1 = 0
        """ (float) Weight of bound. """

        self.c_2 = 1
        """ (float) Weight of objective. """

        self.max_projections = 5
        """ (int) Max projections. """

        self.base_lr = 0.005
        """ (float) Base learning rate. """

        self.max_samples = int(2.5 * MAX_SAMPLES)
        """ (int) Max samples. """

        self.max_attempts = MAX_ATTEMPTS
        """ (int) Max attempts. """

        self.bound = 2
        """ (float) Bound. """

        self.training_mode = training_mode
        """ (bool) Training mode. """

        self.full = full
        """ (bool) Full version. """

        self.suffix = None
        """ (str) Suffix. """


class LearnedOnDataAttackMadryLInfOptions(LearnedOnDataAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnDataAttackMadryLInfOptions, self).__init__('UntargetedBatchLInfProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, False)


class LearnedOnDataAttackMadryLInfFullIterationOptions(LearnedOnDataAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnDataAttackMadryLInfFullIterationOptions, self).__init__('UntargetedBatchLInfProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, True)


class LearnedOnDataAttackMadryLInfFullOptions(LearnedOnDataAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnDataAttackMadryLInfFullOptions, self).__init__('UntargetedBatchLInfProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, True, True)

class LearnedOnDataAttackCWLInfOptions(LearnedOnDataAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnDataAttackCWLInfOptions, self).__init__('UntargetedBatchLInfReparameterizedGradientDescent', 'UntargetedF6', epsilon, max_iterations, 1, False)


class LearnedOnDataAttackMadryL2Options(LearnedOnDataAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnDataAttackMadryL2Options, self).__init__('UntargetedBatchL2ProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, False)


class LearnedOnDataAttackMadryL2FullIterationOptions(LearnedOnDataAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnDataAttackMadryL2FullIterationOptions, self).__init__('UntargetedBatchL2ProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, True)


class LearnedOnDataAttackMadryUnconstrainedOptions(LearnedOnDataAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnDataAttackMadryUnconstrainedOptions, self).__init__('UntargetedBatchL2ClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, False)


class LearnedOnDataAttackCWL2Options(LearnedOnDataAttackOptions):
    def __init__(self, epsilon, max_iterations):
        super(LearnedOnDataAttackCWL2Options, self).__init__('UntargetedBatchL2ReparameterizedGradientDescent', 'UntargetedF6', epsilon, max_iterations, 1, False)


class STNAttackOptions:
    def __init__(self, attack, objective, epsilon, max_iterations, c_0, training_mode, N_theta=6, translation='-0.2,0.2', shear='-0.5,0.5', scale='0.9,1.1',
                 rotation='%g,%g' % (-math.pi / 4, math.pi / 4), full=False):
        assert isinstance(max_iterations, int), max_iterations
        assert c_0 >= 0, c_0
        assert epsilon > 0

        self.attack = attack
        """ (str) Attack name. """

        self.objective = objective
        """ (str) Objective name. """

        self.epsilon = epsilon
        """ (str) Epsilon. """

        self.max_iterations = max_iterations
        """ (int) Max iterations. """

        self.c_0 = c_0
        """ (float) Weight of norm. """

        self.c_1 = 0
        """ (float) Weight of bound. """

        self.c_2 = 1
        """ (float) Weight of objective. """

        self.max_projections = 5
        """ (int) Max projections. """

        self.base_lr = 0.005
        """ (float) Base learning rate. """

        self.max_samples = int(2.5 * MAX_SAMPLES)
        """ (int) Max samples. """

        self.max_attempts = MAX_ATTEMPTS
        """ (int) Max attempts. """

        self.training_mode = training_mode
        """ (bool) Training mode. """

        self.N_theta = N_theta
        """ (int) Number of transformations. """

        self.translation_x = translation
        """ (str) Translation in x. """

        self.translation_y = translation
        """ (str) Translation in y. """

        self.shear_x = shear
        """ (str) Shear in x. """

        self.shear_y = shear
        """ (str) Shear in y. """

        self.scale = scale
        """ (str) Scale. """

        self.rotation = rotation
        """ (str) Rotation. """

        self.color = 0.5
        """ (float) Color. """

        self.full = full
        """ (bool) Full version. """

        self.suffix = None
        """ (str) Suffix. """


class STNAttackMadryLInfOptions(STNAttackOptions):
    def __init__(self, epsilon, max_iterations, N_theta, translation, shear, scale, rotation):
        super(STNAttackMadryLInfOptions, self).__init__('UntargetedBatchLInfProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, False,
                                                        N_theta, translation, shear, scale, rotation)


class STNAttackMadryLInfFullIterationOptions(STNAttackOptions):
    def __init__(self, epsilon, max_iterations, N_theta, translation, shear, scale, rotation):
        super(STNAttackMadryLInfFullIterationOptions, self).__init__('UntargetedBatchLInfProjectedClippedGradientDescent', 'UntargetedF0', epsilon,
                                                                     max_iterations, 0, True, N_theta, translation, shear, scale, rotation)


class STNAttackMadryLInfFullOptions(STNAttackOptions):
    def __init__(self, epsilon, max_iterations, N_theta, translation, shear, scale, rotation):
        super(STNAttackMadryLInfFullOptions, self).__init__('UntargetedBatchLInfProjectedClippedGradientDescent', 'UntargetedF0', epsilon,
                                                                     max_iterations, 0, True, N_theta, translation, shear, scale, rotation, True)


class STNAttackMadryUnconstrainedOptions(STNAttackOptions):
    def __init__(self, epsilon, max_iterations, N_theta, translation, shear, scale, rotation):
        super(STNAttackMadryUnconstrainedOptions, self).__init__('UntargetedBatchL2ClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, False,
                                                                 N_theta, translation, shear, scale, rotation)


class STNAttackCWLInfOptions(STNAttackOptions):
    def __init__(self, epsilon, max_iterations, N_theta, translation, shear, scale, rotation):
        super(STNAttackCWLInfOptions, self).__init__('UntargetedBatchLInfReparameterizedGradientDescent', 'UntargetedF6', epsilon, max_iterations, 1, False,
                                                     N_theta, translation, shear, scale, rotation)


class STNAttackMadryL2Options(STNAttackOptions):
    def __init__(self, epsilon, max_iterations, N_theta, translation, shear, scale, rotation):
        super(STNAttackMadryL2Options, self).__init__('UntargetedBatchL2ProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, False, N_theta, translation, shear, scale, rotation)


class STNAttackMadryL2FullIterationOptions(STNAttackOptions):
    def __init__(self, epsilon, max_iterations, N_theta, translation, shear, scale, rotation):
        super(STNAttackMadryL2FullIterationOptions, self).__init__('UntargetedBatchL2ProjectedClippedGradientDescent', 'UntargetedF0', epsilon, max_iterations, 0, True, N_theta, translation, shear, scale, rotation)


class STNAttackCWL2Options(STNAttackOptions):
    def __init__(self, epsilon, max_iterations, N_theta, translation, shear, scale, rotation):
        super(STNAttackCWL2Options, self).__init__('UntargetedBatchL2ReparameterizedGradientDescent', 'UntargetedF6', epsilon, max_iterations, 1, False, N_theta, translation, shear, scale, rotation)
