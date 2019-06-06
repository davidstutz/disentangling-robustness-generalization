import os
from .log import log, LogLevel

# This file holds a bunch of specific paths used for experiments and
# data. The intention is to have all important paths at a central location, while
# allowing to easily prototype new experiments.

# Base directory for data and experiments.
BASE_DATA = '/BS/dstutz2/work/cvpr2019/data/'
BASE_EXPERIMENTS = '/BS/dstutz2/work/cvpr2019/experiments/'

if not os.path.exists(BASE_DATA):
    log('[Error] could not find data directory %s' % BASE_DATA, LogLevel.ERROR)
    log('[Error] set the data directory in common/paths.py', LogLevel.ERROR)
    raise Exception('Data directory %s not found.' % BASE_DATA)

if not os.path.exists(BASE_EXPERIMENTS):
    log('[Error] could not find experiment directory %s' % BASE_DATA, LogLevel.ERROR)
    log('[Error] set the experiment directory in common/paths.py', LogLevel.ERROR)
    raise Exception('Experiment directory %s not found.' % BASE_DATA)

# Common extension types used.
TXT_EXT = '.txt'
HDF5_EXT = '.h5'
STATE_EXT = '.pth.tar'
LOG_EXT = '.log'
PNG_EXT = '.png'
CSV_EXT = '.csv'
PICKLE_EXT = '.pkl'
LATEX_EXT = '.tex'
PDF_EXT = '.pdf'
GZIP_EXT = '.gz'
MAT_EXT = '.mat'

# To manipulate paths globally.
_GLOBAL_KWARGS = dict()


def set_globals(**kwargs):
    """
    Set global kwargs used in all paths.
    """

    global _GLOBAL_KWARGS
    _GLOBAL_KWARGS = kwargs


def get_globals():
    """
    Get global kwargs used in all paths.
    """

    global _GLOBAL_KWARGS
    return _GLOBAL_KWARGS


# For merging kwargs.
def merge_kwargs(**kwargs):
    """
    Merge kwargs with globals.
    """

    return {**get_globals(), **kwargs}


# Naming conventions.
def data_file(name, ext=HDF5_EXT, **kwargs):
    """
    Generate path to data file.

    :param name: name of file
    :type name: str
    :param ext: extension (including period)
    :type ext: str
    :return: filepath
    :rtype: str
    """

    kwargs = merge_kwargs(**kwargs)

    filepath = os.path.join(BASE_DATA, name)
    if 'characters' in kwargs:
        filepath += '_' + str(kwargs.get('characters'))
    if 'fonts' in kwargs:
        filepath += '_' + str(kwargs.get('fonts'))
    if 'transformations' in kwargs:
        filepath += '_' + str(kwargs.get('transformations'))
    if 'size' in kwargs:
        filepath += '_' + str(kwargs.get('size'))
    if 'suffix' in kwargs:
        filepath += '_' + str(kwargs.get('suffix'))

    return filepath + ext


def raw_emnist_train_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('emnist/emnist-digits-train-images-idx3-ubyte', GZIP_EXT)


def raw_emnist_test_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('emnist/emnist-digits-test-images-idx3-ubyte', GZIP_EXT)


def raw_emnist_train_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('emnist/emnist-digits-train-labels-idx1-ubyte', GZIP_EXT)


def raw_emnist_test_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('emnist/emnist-digits-test-labels-idx1-ubyte', GZIP_EXT)


def emnist_train_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('emnist/train_images', HDF5_EXT)


def emnist_test_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('emnist/test_images', HDF5_EXT)


def emnist_train_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('emnist/train_labels', HDF5_EXT)


def emnist_test_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('emnist/test_labels', HDF5_EXT)


def raw_fashion_train_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('fashion/train-images-idx3-ubyte', GZIP_EXT)


def raw_fashion_test_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('fashion/t10k-images-idx3-ubyte', GZIP_EXT)


def raw_fashion_train_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('fashion/train-labels-idx1-ubyte', GZIP_EXT)


def raw_fashion_test_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('fashion/t10k-labels-idx1-ubyte', GZIP_EXT)


def fashion_train_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('fashion/train_images', HDF5_EXT)


def fashion_test_images_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('fashion/test_images', HDF5_EXT)


def fashion_train_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('fashion/train_labels', HDF5_EXT)


def fashion_test_labels_file():
    """
    Train images of MNSIT.

    :return: filepath
    :rtype: str
    """

    return data_file('fashion/test_labels', HDF5_EXT)


def raw_celeba_images_dir():
    """
    Image directory for CelebA.

    :return: filepath
    :rtype: str
    """

    return data_file('CelebA/img_align_celeba', '')


def raw_celeba_labels_file():
    """
    Labels file for CelebA.

    :return: filepath
    :rtype: str
    """

    return data_file('CelebA/Anno/list_attr_celeba', TXT_EXT)


def celeba_train_images_file():
    """
    Image directory for CelebA.

    :return: filepath
    :rtype: str
    """

    return data_file('CelebA/train_images', HDF5_EXT)


def celeba_test_images_file():
    """
    Image directory for CelebA.

    :return: filepath
    :rtype: str
    """

    return data_file('CelebA/test_images', HDF5_EXT)


def celeba_train_labels_file():
    """
    Labels file for CelebA.

    :return: filepath
    :rtype: str
    """

    return data_file('CelebA/train_labels', HDF5_EXT)


def celeba_test_labels_file():
    """
    Labels file for CelebA.

    :return: filepath
    :rtype: str
    """

    return data_file('CelebA/test_labels', HDF5_EXT)


def database_file(**kwargs):
    """
    Generate fonts file path.

    :return: filepath to image data
    :rtype: str
    """

    return data_file('fonts/fonts', HDF5_EXT, **kwargs)


def images_file(**kwargs):
    """
    Generate image data file path.

    :return: filepath to image data
    :rtype: str
    """

    return data_file('fonts/images', HDF5_EXT, **kwargs)


def codes_file(**kwargs):
    """
    Generate codes data file path.

    :return: filepath to codes data
    :rtype: str
    """

    return data_file('fonts/codes', HDF5_EXT, **kwargs)


def theta_file(**kwargs):
    """
    Generate theta data file path.

    :return: filepath to theta data
    :rtype: str
    """

    return data_file('fonts/theta', HDF5_EXT, **kwargs)


def test_images_file(**kwargs):
    """
    Generate test image file path.

    :return: filepath to test images
    :rtype: str
    """

    return data_file('fonts/test_images', HDF5_EXT, **kwargs)


def train_images_file(**kwargs):
    """
    Generate train image file path.

    :return: filepath to train images
    :rtype: str
    """

    return data_file('fonts/train_images', HDF5_EXT, **kwargs)


def test_codes_file(**kwargs):
    """
    Generate test code file path.

    :return: filepath to test codes
    :rtype: str
    """

    return data_file('fonts/test_codes', HDF5_EXT, **kwargs)

def train_codes_file(**kwargs):
    """
    Generate train code file path.

    :return: filepath to train codes
    :rtype: str
    """

    return data_file('fonts/train_codes', HDF5_EXT, **kwargs)


def test_theta_file(**kwargs):
    """
    Generate test theta file path.

    :return: filepath to test theta
    :rtype: str
    """

    return data_file('fonts/test_theta', HDF5_EXT, **kwargs)


def train_theta_file(**kwargs):
    """
    Generate train theta file path.

    :return: filepath to train theta
    :rtype: str
    """

    return data_file('fonts/train_theta', HDF5_EXT, **kwargs)


def experiment_dir(name, **kwargs):
    """
    Generate path to experiment directory.

    :param name: name of directory
    :type name: str
    :return: filepath
    :rtype: str
    """

    kwargs = merge_kwargs(**kwargs)

    base = BASE_EXPERIMENTS
    if 'experiment' in kwargs:
        base = os.path.join(base, kwargs.get('experiment'))
    else:
        base = os.path.join(base, 'Fonts')
    # Process kwargs
    return os.path.join(base, name)


def experiment_file(name, ext=HDF5_EXT, **kwargs):
    """
    Generate path to experiment file.

    :param name: name of file
    :type name: str
    :return: filepath
    :rtype: str
    """

    kwargs = merge_kwargs(**kwargs)

    base = BASE_EXPERIMENTS
    if 'experiment' in kwargs:
        base = os.path.join(base, kwargs.get('experiment'))
    else:
        base = os.path.join(base, 'Fonts')
    # Process kwargs
    return os.path.join(base, name) + ext


def results_file(name, **kwargs):
    """
    Generate path to experiment data/results file.

    :param name: name of file
    :type name: str
    :return: filepath
    :rtype: str
    """

    return experiment_file(name, HDF5_EXT, **kwargs)


def image_file(name, **kwargs):
    """
    Generate path to experiment image file.

    :param name: name of file
    :type name: str
    :return: filepath
    :rtype: str
    """

    return experiment_file(name, PNG_EXT, **kwargs)


def state_file(name, **kwargs):
    """
    Generate path to experiment state file.

    :param name: name of file
    :type name: str
    :return: filepath
    :rtype: str
    """

    return experiment_file(name, STATE_EXT, **kwargs)


def log_file(name, **kwargs):
    """
    Generate path to experiment log file.

    :param name: name of file
    :type name: str
    :return: filepath
    :rtype: str
    """

    return experiment_file(name, LOG_EXT, **kwargs)


def statistic_file(name, **kwargs):
    """
    Generate path to experiment statistic csv file.

    :param name: name of file
    :type name: str
    :return: filepath
    :rtype: str
    """

    return experiment_file(name, CSV_EXT, **kwargs)


def pickle_file(name, **kwargs):
    """
    Generate path to experiment pickle file.

    :param name: name of file
    :type name: str
    :return: filepath
    :rtype: str
    """

    return experiment_file(name, PICKLE_EXT, **kwargs)


def latex_file(name, **kwargs):
    """
    Generate path to experiment pickle file.

    :param name: name of file
    :type name: str
    :return: filepath
    :rtype: str
    """

    return experiment_file(name, LATEX_EXT, **kwargs)