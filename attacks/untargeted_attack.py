import torch
from common import cuda


class UntargetedAttack:
    """
    Generic untargeted attack.
    """

    def __init__(self, model, images, classes=None):
        """
        Constructor.

        :param model: model to attack
        :type model: torch.nn.Module
        :param images: image(s) to attack
        :type images: torch.autograd.Variable
        :param classes: true classes, if None, they will be deduced to avoid label leaking
        :type classes: torch.autograd.Variable
        """

        assert isinstance(model, torch.nn.Module), 'given model is not of type torch.nn.Module'
        assert isinstance(images, torch.autograd.Variable), 'given image has to be torch.autograd.Variable'
        assert isinstance(classes, torch.autograd.Variable) or classes is None, 'given class has to be torch.autograd.Variable'
        assert images.requires_grad == False, 'image does not require grads'
        assert classes is None or classes.requires_grad == False, 'classes does not require grad'

        self.model = model
        """ (torch.nn.Module) Model to attack. """

        self.images = images
        """ (torch.autograd.Variable) Images to attack. """

        self.classes = classes
        """
        (torch.autograd.Variable or None) True classes.
        The true classes, as given in the constructor or set through set_classes.
        If None, the classes will be derived from the logits using argmax to avoid
        label leaking. Note that there are also objectives available that do not
        make use of the true or derived classes.
        """

        self.logits = self.model.forward(self.images)
        self.logits = self.logits.detach() # to avoid double backward
        """ (torch.autograd.Variable) Reference logits. """

        if self.classes is None:
            self.classes = torch.max(self.logits, 1)[1]
        assert self.classes.size(0) == self.images.size(0)

        self.min_bound = torch.zeros_like(self.images)
        """
        (torch.Tensor or None) Minimum bound.
        min_bound and max_bound represent simple bounds on the possible
        values of the image - or image + perturbation, respectively.
        Depending on the implementation, these bounds are enforced through
        optimization or projections. They can be set to None; however, default
        is [0, 1] as commonly used for images.
        """

        self.max_bound = torch.ones_like(self.images)
        """ (torch.Tensor or None) Maximum bound; see min_bound. """

        self.encoder = None
        """
        (torch.nn.Module or None) Encoder to use if an encoder bound/on-manifold bound is specified.
        The bound can be, depending on implementation, enforced through optimization
        or projection; if None, no bound is enforced.
        """

        self.encoder_min_bound = None
        """ (torch.Tensor or None) Minimum bound for encoder; also see encoder, min_bound. """
        self.encoder_max_bound = None
        """ (torch.Tensor or None) Maximum bound for encoder; also see encoder, min_bound. """

        self.auto_encoder = None
        """ (torch.nn.Module) Might hold an auto encoder used to enforce on-manifold constraints by projection. """

        self.training_mode = False
        """ (bool) Training mode. """

        self.history = []
        """ ([dict] History. """

    def set_training_mode(self, training_mode=True):
        """
        Set training mode for attack.

        :param training_mode: training mode
        :type training_mode: bool
        """

        self.training_mode = training_mode

    def set_classes(self, classes):
        """
        Set true classes.

        :param classes: true classes, if None, they will be deduced to avoid label leaking
        :type classes: torch.autograd.Variable or None
        """

        self.classes = classes
        if self.classes is None:
            self.classes = torch.max(self.logits, 1)[1]

    def set_bound(self, min, max):
        """
        Set minimum and maximum bound; if None is specified, bounds are not enforced.

        :param min: minimum bound
        :type min: float, torch.Tensor or None
        :param max: maximum bound
        :return: float, torch.Tensor or None
        """

        assert isinstance(min, torch.Tensor) or isinstance(min, float) or min is None, 'min needs to be float or torch.Tensor'
        assert isinstance(max, torch.Tensor) or isinstance(max, float) or max is None, 'max needs to be float or torch.Tensor'

        if min is None:
            self.min_bound = None
        else:
            if isinstance(min, torch.Tensor):
                self.min_bound = min
            elif isinstance(min, float):
                self.min_bound = torch.ones_like(self.images)*min

            if cuda.is_cuda(self.model):
                self.min_bound = self.min_bound.cuda()

        if max is None:
            self.max_bound = None
        else:
            if isinstance(max, torch.Tensor):
                self.max_bound = max
            elif isinstance(max, float):
                self.max_bound = torch.ones_like(self.images) * max

            if cuda.is_cuda(self.model):
                self.max_bound = self.max_bound.cuda()

    def set_encoder_bound(self, encoder, min, max):
        """
        Set a bound on the encoder of the perturbation; this allows to easily implement "on-manifold constraints".
        If None is specified, bounds are not enforced.

        :param encoder: encoder
        :type encoder: torch.nn.Module
        :param min: minimum bound
        :type min: float, torch.Tensor or None
        :param max: maximum bound
        :return: float, torch.Tensor or None
        """

        assert isinstance(min, torch.Tensor) or isinstance(min, float) or min is None, 'min needs to be float or torch.Tensor'
        assert isinstance(max, torch.Tensor) or isinstance(max, float) or max is None, 'max needs to be float or torch.Tensor'
        assert isinstance(encoder, torch.nn.Module), 'encoder needs to be torch.nn.Module'

        self.encoder = encoder

        if min is None:
            self.encoder_min_bound = None
        else:
            if isinstance(min, torch.Tensor):
                self.encoder_min_bound = min
            elif isinstance(min, float):
                self.encoder_min_bound = torch.ones_like(self.images) * min

            if cuda.is_cuda(self.model):
                self.encoder_min_bound = self.encoder_min_bound.cuda()

        if max is None:
            self.encoder_max_bound = None
        else:
            if isinstance(max, torch.Tensor):
                self.encoder_max_bound = max
            elif isinstance(max, float):
                self.encoder_max_bound = torch.ones_like(self.images) * max

            if cuda.is_cuda(self.model):
                self.encoder_max_bound = self.encoder_max_bound.cuda()

    def set_auto_encoder(self, auto_encoder):
        """
        Set an auto-encoder to use as a projection to implement "on-manifold constraints".

        :param auto_encoder: auto encoder
        :type auto_encoder: torch.nn.Module
        """

        assert isinstance(auto_encoder, torch.nn.Module), 'auto encoder needs to be torch.nn.Module'
        #assert isinstance(auto_encoder, models.AutoEncoder), 'auto encoder needs to be AutoEncoder'
        self.auto_encoder = auto_encoder

    def project_auto_encoder(self, perturbations):
        """
        Project the current image + perturbation onto the manifold and deduce the new perurbation from it.

        :param perturbations: current perturbations
        :type perturbations: torch.Tensor
        :return: projected perturbations
        :rtype: torch.Tensor
        """

        assert self.auto_encoder is not None, 'called project_auto_encoder without setting the auto encoder first'
        assert isinstance(perturbations, torch.Tensor), 'given perturbation needs to be torch.Tensor'

        images = torch.autograd.Variable(self.images + perturbations, False)
        if cuda.is_cuda(self.auto_encoder):
            images = images.cuda()

        reconstruction, _, _ = self.auto_encoder.forward(images)
        return reconstruction.data - self.images.data # Retrieve the perturbation from the projected image!

    def run(self, objective, verbose=False):
        """
        Run attack.
        
        :param untargeted_objective: untargeted objective
        :type untargeted_objective: UntargetedObjective
        :param verbose: output progress
        :type verbose: bool
        """

        raise NotImplementedError()