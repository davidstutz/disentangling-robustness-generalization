import os
from . import utils
from .log import log, LogLevel


if not utils.display():
    log('[Error] DISPLAY not found, plot not available!', LogLevel.ERROR)
    raise Exception("DISPLAY not found, plot not available!")


import skimage.transform
from matplotlib import pyplot
import numpy
import math


def image(filepath, image, scale=1, cmap='gray', vmin=0, vmax=1):
    """
    Save an image using pyplot.

    :param filepath: filepath
    :type filepath: str
    :param image: image
    :type image: numpy.ndarray
    :param scale: scale
    :type scale: float
    :param cmap: cmape
    :type cmape: str
    :param vmin: minimum
    :type vmin: float
    :param vmax: minimum
    :type vmax: float
    """

    # pyplot.imshow(image, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    # pyplot.axis('off')
    # pyplot.gca().get_xaxis().set_visible(False)
    # pyplot.gca().get_yaxis().set_visible(False)
    # pyplot.savefig(filepath, bbox_inches='tight', pad_inches=0) # dpi
    # pyplot.close()

    image = numpy.squeeze(image)
    utils.makedir(os.path.dirname(filepath))
    image = skimage.transform.rescale(image, scale=scale, order=0)
    pyplot.imsave(filepath, image, cmap=cmap, vmin=vmin, vmax=vmax)


def perturbation(filepath, perturbation, scale=1, cmap='seismic', vmin=-1, vmax=1):
    """
    Save a perturbation.

    :param filepath: filepath
    :type filepath: str
    :param perturbation: perturbation
    :type perturbation: numpy.ndarray
    :param scale: scale
    :type scale: float
    :param cmap: cmape
    :type cmape: str
    :param vmin: minimum
    :type vmin: float
    :param vmax: minimum
    :type vmax: float
    """

    perturbation = numpy.squeeze(perturbation)
    utils.makedir(os.path.dirname(filepath))
    perturbation = skimage.transform.rescale(perturbation, scale=scale, order=0)
    pyplot.imsave(filepath, perturbation, cmap=cmap, vmin=vmin, vmax=vmax)


def matrix(filepath, matrix, scale=25, cmap='seismic', vmin=-1, vmax=1):
    """
    Save matrix.

    :param filepath: filepath
    :type filepath: str
    :param matrix: matrix
    :type matrix: numpy.ndarray
    :param scale: scale
    :type scale: float
    :param cmap: cmape
    :type cmape: str
    :param vmin: minimum
    :type vmin: float
    :param vmax: minimum
    :type vmax: float
    """

    if len(matrix.shape) < 2:
        matrix = matrix.reshape((1, matrix.shape[0]))

    # pyplot.matshow(matrix, vmin=vmin, vmax=vmax)
    # pyplot.savefig(filepath)
    # pyplot.close()

    utils.makedir(os.path.dirname(filepath))
    matrix = skimage.transform.rescale(matrix, scale=scale, order=0)
    pyplot.imsave(filepath, matrix, cmap=cmap, vmin=vmin, vmax=vmax)


def mosaic(filepath, images, cols=6, scale=1, cmap='gray', vmin=0, vmax=1):
    """
    Create a mosaic of images, specifically display a set of images in
    a variable number of rows with a fixed number of columns.

    :param filepath: file to image
    :type filepath: str
    :param images: images
    :type images: numpy.ndarray
    :param cols: number of columns
    :type cols: int
    :param cmap: color map to use, default is binary
    :type cmap: str
    """

    assert len(images.shape) == 3 or len(images.shape) == 4
    if len(images.shape) == 3:
        images = numpy.expand_dims(images, axis=4)

    number = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    channels = images.shape[3]

    rows = int(math.ceil(number / float(cols)))
    mosaic_image = numpy.zeros((rows * height, cols * width, channels))

    for i in range(rows):
        for j in range(cols):
            k = i * cols + j
            if k < images.shape[0]:
                mosaic_image[i * height: (i + 1) * height, j * width: (j + 1) * width, :] = images[k][:][:][:]

    image(filepath, mosaic_image, scale, cmap, vmin, vmax)