import numpy


class DataProcessor:
    """
    Data processor interface.
    """

    def __call__(self, images):
        """
        Apply processor.

        :param images: image batch
        :type images: numpy.ndarray
        :return: processed images
        :rtype: numpy.ndarray
        """

        raise NotImplementedError()


class DataPreProcessing:
    """
    Apply multiple processing steps.
    """

    def __init__(self, processors):
        """
        Constructor.

        :param processors: List of processors
        :type processors: [DataProcessor]
        """

        assert isinstance(processors, list)

        self.processors = processors
        """ ([DataProcessor]) Data processors. """

    def __call__(self, images):
        """
        Apply processors.

        :param images: image batch
        :type images: numpy.ndarray
        :return: processed images
        :rtype: numpy.ndarray
        """

        for processor in self.processors:
            images = processor(images)

        return images


class GaussianNoise(DataProcessor):
    """
    Add Gaussian noise.
    """

    def __init__(self, std):
        """
        Constructor.

        :param std: std of Gaussian noise
        :type std: float
        """

        self.std = std
        """ (float) Standard deviation of Gaussian noise. """

    def __call__(self, images):
        """
        Apply processor.

        :param images: image batch
        :type images: numpy.ndarray
        :return: processed images
        :rtype: numpy.ndarray
        """

        images = images + numpy.random.randn(*images.shape)*self.std
        return images.astype(numpy.float32)


class Clip(DataProcessor):
    """
    Add Gaussian noise.
    """

    def __init__(self, min, max):
        """
        Constructor.

        :param min: min value
        :type min: float
        :param max: max value
        :type max: float
        """

        self.min = min
        """ (float) Min value. """

        self.max = max
        """ (float) Max value. """

    def __call__(self, images):
        """
        Apply processor.

        :param images: image batch
        :type images: numpy.ndarray
        :return: processed images
        :rtype: numpy.ndarray
        """

        return numpy.clip(images, self.min, self.max)


class Flip(DataProcessor):
    """
    Flip.
    """

    def __call__(self, images):
        """
        Apply processor.

        :param images: image batch
        :type images: numpy.ndarray
        :return: processed images
        :rtype: numpy.ndarray
        """

        flipped = numpy.flip(images, 2)
        flip = numpy.random.rand(images.shape[0])
        flip = flip > 0.5
        images[flip] = flipped[flip]
        return images


class Contrast(DataProcessor):
    """
    Contrast.
    """

    def __init__(self, min_exponent, max_exponent):
        """
        Constructor.

        :param min_exponent: max exponent
        :type min_exponent: float
        :param max_exponent: min exponent
        :type max_exponent: float
        """

        self.min = min_exponent
        """ (float) Minimum exponent. """

        self.max = max_exponent
        """ (float) Maximum exponent. """

    def __call__(self, images):
        """
        Apply processor.

        :param images: image batch
        :type images: numpy.ndarray
        :return: processed images
        :rtype: numpy.ndarray
        """

        exponent = numpy.random.uniform(self.min, self.max, size=(images.shape[0])).astype(numpy.float32)
        return numpy.power(images, exponent.reshape(-1, 1, 1, 1).repeat(images.shape[1], 1).repeat(images.shape[2], 2).repeat(images.shape[3], 3))


class Crop(DataProcessor):
    """
    Add Gaussian noise.
    """

    def __init__(self, height, width, height_padding, width_padding):
        """
        Constructor.

        :param height: height
        :type height: int
        :param width: width
        :type width: int
        :param height_padding: padding of height before cropping
        :type height_padding: int
        :param width_padding: padding of width before cropping
        :type width_padding: int
        """

        self.height = height
        """ (int) Height. """

        self.width = width
        """ (int) Width. """

        self.height_padding = height_padding
        """ (int) Padding of height before cropping. """

        self.width_padding = width_padding
        """ (int) Padding of width before cropping. """

    def __call__(self, images):
        """
        Apply processor.

        :param images: image batch
        :type images: numpy.ndarray
        :return: processed images
        :rtype: numpy.ndarray
        """

        if len(images.shape) == 4:
            padded_images = numpy.zeros((images.shape[0], images.shape[1] + 2*self.height_padding, images.shape[2] + 2*self.width_padding, images.shape[3]))
            padded_images[:, self.height_padding : self.height_padding + images.shape[1], self.width_padding : self.width_padding + images.shape[2], :] = images
        elif len(images.shape) == 3:
            padded_images = numpy.zeros((images.shape[0], images.shape[1] + 2*self.height_padding, images.shape[2] + 2*self.width_padding))
            padded_images[:, self.height_padding : self.height_padding + images.shape[1], self.width_padding : self.width_padding + images.shape[2]] = images
        else:
            assert False

        rand_i = numpy.random.randint(0, 2*self.height_padding + 1)
        rand_j = numpy.random.randint(0, 2*self.width_padding + 1)

        return padded_images[:, rand_i : rand_i + images.shape[1], rand_j : rand_j + images.shape[2]].astype(numpy.float32)


class Normalize(DataProcessor):
    """
    Normalize.
    """

    def __init__(self, mean, std):
        """
        Constructor.

        :param mean: mean
        :type mean: [float]
        :param std: std
        :type std: [float]
        """

        self.mean = mean
        """ ([float]) Mean. """

        self.std = std
        """ ([float]) Standard deviation. """

    def __call__(self, images):
        """
        Apply processor.

        :param images: image batch
        :type images: numpy.ndarray
        :return: processed images
        :rtype: numpy.ndarray
        """

        if len(images.shape) == 4:
            return numpy.stack((
                (images[:, :, :, 0] - self.mean[0]) / self.std[0],
                (images[:, :, :, 1] - self.mean[1]) / self.std[1],
                (images[:, :, :, 2] - self.mean[2]) / self.std[2]
            ), axis=-1)
        elif len(images.shape) == 3:
            return (images - self.mean)/self.std
        else:
            assert False