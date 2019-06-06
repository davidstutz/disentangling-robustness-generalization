import torch
import os


class State:
    """
    Represents a training state that can be saved and loaded.
    """

    def __init__(self, model, optimizer, epoch):
        """
        Constructor.

        :param model: model
        :type model: torch.nn.Module
        :param optimizer: optimizer
        :type optimizer: torch.optim.Optimizer
        :param epoch: epoch
        :type epoch: int
        """

        if isinstance(model, dict):
            self.model = model
        else:
            self.model = model.state_dict()
        """ (dict) Model. """

        if isinstance(optimizer, dict):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer.state_dict()
        """ (dict) Optimizer. """

        self.epoch = epoch
        """ (int) Epcoh. """

    def save(self, filepath):
        """
        Save the state.

        :param filepath: file to save to
        :type filepath: str
        """

        torch.save({
            'model': self.model,
            'optimizer': self.optimizer,
            'epoch': self.epoch,
        }, filepath)

    @staticmethod
    def checkpoint(model, optimizer, epoch, filepath):
        """
        Quick access to State.save.

        :param model: model
        :type model: torch.nn.Module
        :param optimizer: optimizer
        :type optimizer: torch.optim.Optimizer
        :param epoch: epoch
        :type epoch: int
        :param filepath: path to file
        :type filepath: str
        """

        state = State(model, optimizer, epoch)
        state.save(filepath)

    @staticmethod
    def load(filepath):
        """
        Load a state.

        :param filepath: file to load
        :type filepath: str
        :return: state
        :rtype: State
        """

        assert os.path.exists(filepath), 'file %s not found' % filepath

        # https://discuss.pytorch.org/t/gpu-memory-usage-increases-by-90-after-torch-load/9213/3
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        state = State(checkpoint['model'], checkpoint['optimizer'], checkpoint['epoch'])

        del checkpoint
        torch.cuda.empty_cache()

        return state
