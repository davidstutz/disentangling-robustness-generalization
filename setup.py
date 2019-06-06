def check_simple():
    """
    Checks packages that can be installed via PIP.
    """

    try:
        import os
        import re
        import json
        import numpy
        import zipfile
        import importlib
        import umap
        import argparse
        import math
        import functools
        import warnings
        import sklearn
        import xgboost
        import sklearn
        import imageio
        import scipy
    except ImportError as e:
        print('A package could not be loaded, check the source and install it via pip3.')


def check_h5py():
    """
    Check h5py installation.
    """

    try:
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        import h5py
        warnings.resetwarnings()
        print('h5py imported.')
        print('If you still get an error when importing h5py (e.g. saying that numpy has not attribute dtype), try uninstalling h5py, numpy and six, and reinstalling it using pip3.')
    except ImportError as e:
        print('Could not load h5py; it might to be installed manually and might have problems with the installed NumPy version.')


def check_torch():
    """
    Check torch installation.
    """

    try:
        import torch
        import numpy
        print('Torch imported.')

        if torch.cuda.is_available():
            print('CUDA seems to be available and supported.')

        # Might print something like
        # BS/dstutz/work/dev-box/pip9/lib/python2.7/site-packages/torch/cuda/__init__.py:89: UserWarning:
        #     Found GPU0 Tesla V100-PCIE-16GB which requires CUDA_VERSION >= 8000 for
        #     optimal performance and fast startup time, but your PyTorch was compiled
        #     with CUDA_VERSION 8000. Please install the correct PyTorch binary
        #     using instructions from http://pytorch.org
        # or:
        # Segmentation fault (not sure what's the cause).

        target = torch.from_numpy(numpy.array([[0, 0], [0, 1], [0, 1]], dtype=float))
        target = target.cuda()

        import torchvision
        print('Torchvision imported.')

        print('Unless there were warnings or segmentation faults, everything works!')
    except ImportError as e:
        print('Torch could not be imported.')


def check_common():
    import os
    import sys
    sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')

    import common.utils
    import common.cuda
    import common.latex
    import common.numpy
    import common.paths
    import common.ppca
    import common.scheduler
    import common.state
    import common.timer
    if common.utils.display():
        import common.vis
        import common.plot
        import common.table

    from common.log import log, LogLevel
    log('BASE_EXPERIMENTS=%s' % common.paths.BASE_EXPERIMENTS)
    log('BASE_DATA=%s' % common.paths.BASE_DATA)
    if not common.utils.display():
        log('NO DISPLAY', LogLevel.WARNING)


def check_files():
    import os
    from common import paths

    def check_file(filepath):
        if not os.path.exists(filepath):
            print('File %s not found.' % filepath)

    check_file(paths.database_file())
    check_file(paths.images_file())
    check_file(paths.theta_file())
    check_file(paths.codes_file())

    check_file(paths.test_images_file())
    check_file(paths.train_images_file())
    check_file(paths.test_theta_file())
    check_file(paths.train_theta_file())
    check_file(paths.test_codes_file())
    check_file(paths.train_codes_file())

    check_file(paths.emnist_test_images_file())
    check_file(paths.emnist_train_images_file())
    check_file(paths.emnist_test_labels_file())
    check_file(paths.emnist_train_labels_file())

    check_file(paths.fashion_test_images_file())
    check_file(paths.fashion_train_images_file())
    check_file(paths.fashion_test_labels_file())
    check_file(paths.fashion_train_labels_file())

    check_file(paths.celeba_test_images_file())
    check_file(paths.celeba_train_images_file())
    check_file(paths.celeba_test_labels_file())
    check_file(paths.celeba_train_labels_file())


if __name__ == '__main__':
    check_simple()
    check_h5py()
    check_torch()
    check_common()
    check_files()