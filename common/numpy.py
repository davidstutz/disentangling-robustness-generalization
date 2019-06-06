import numpy
import scipy.stats


def one_hot(array, N):
    """
    Convert an array of numbers to an array of one-hot vectors.

    :param array: classes to convert
    :type array: numpy.ndarray
    :param N: number of classes
    :type N: int
    :return: one-hot vectors
    :rtype: numpy.ndarray
    """

    array = array.astype(int)
    assert numpy.max(array) < N
    assert numpy.min(array) >= 0

    one_hot = numpy.zeros((array.shape[0], N))
    one_hot[numpy.arange(array.shape[0]), array] = 1
    return one_hot


def contract_dims(array, axis=0):
    """
    Intended as the oppositve of numpy.expand_dims, especially for merging to axes.

    :param array: input array
    :type array: numpy.ndarray
    :param axis: the axis the remove/contract
    :type axis: int
    :return: array
    :rtype: numpy.ndarray
    """

    assert isinstance(array, numpy.ndarray), 'array needs to be numpy.ndarray'
    assert axis < len(array.shape), 'given axis does exceed rank'
    assert axis != len(array.shape) - 1, 'last dimension cannot be contracted'

    i = 0
    shape = []

    while i < len(array.shape):
        if i == axis:
            shape.append(-1)
            i += 1 # Skip the next dimension as we want to contract it
        else:
            shape.append(array.shape[i])
        i += 1

    return array.reshape(tuple(shape))


def concatenate(array1, array2, axis=0):
    """
    Basically a wrapper for numpy.concatenate, with the exception
    that the array itself is returned if its None or evaluates to False.

    :param array1: input array or None
    :type array1: mixed
    :param array2: input array
    :type array2: numpy.ndarray
    :param axis: axis to concatenate
    :type axis: int
    :return: concatenated array
    :rtype: numpy.ndarray
    """

    assert isinstance(array2, numpy.ndarray)
    if array1 is not None:
        assert isinstance(array1, numpy.ndarray)
        return numpy.concatenate((array1, array2), axis=axis)
    else:
        return array2


def uniform_ball(batch_size, dim, epsilon=1, ord=2, alternative_mode=True):
    """
    Sample vectors uniformly in the n-ball.

    See Harman et al., On decompositional algorithms for uniform sampling from n-spheres and n-balls.

    :param batch_size: how many vectors to sample
    :type batch_size: int
    :param dim: dimensionality of vectors
    :type dim: int
    :param epsilon: epsilon-ball
    :type epsilon: float
    :param ord: norm to use
    :type ord: int
    :param alternative_mode: whether to sample from uniform distance instead of sampling uniformly with respect to volume
    :type alternative_mode: bool
    :return: batch_size x dim tensor
    :rtype: numpy.ndarray
    """

    random = numpy.random.randn(batch_size, dim)
    random /= numpy.repeat(numpy.linalg.norm(random, ord=ord, axis=1).reshape(-1, 1), axis=1, repeats=dim)
    random *= epsilon
    if alternative_mode:
        uniform = numpy.random.uniform(0, 1, (batch_size, 1)) # exponent is only difference!
    else:
        uniform = numpy.random.uniform(0, 1, (batch_size, 1)) ** (1. / dim)
    random *= numpy.repeat(uniform, axis=1, repeats=dim)

    return random


def truncated_normal(size, lower=-2, upper=2):
    """
    Sample from truncated normal.

    See https://stackoverflow.com/questions/18441779/how-to-specify-upper-and-lower-limits-when-using-numpy-random-normal.

    :param size: size of vector
    :type size: [int]
    :param lower: lower bound
    :type lower: float
    :param upper: upper bound
    :type upper: float
    :return: batch_size x dim tensor
    :rtype: numpy.ndarray
    """

    return scipy.stats.truncnorm.rvs(lower, upper, size=size)


def project_orthogonal(basis, vectors, rank=None):
    """
    Project the given vectors on the basis using an orthogonal projection.

    :param basis: basis vectors to project on
    :type basis: numpy.ndarray
    :param vectors: vectors to project
    :type vectors: numpy.ndarray
    :return: projection
    :rtype: numpy.ndarray
    """

    # The columns of Q are an orthonormal basis of the columns of basis
    Q, R = numpy.linalg.qr(basis)
    if rank is not None and rank > 0:
        Q = Q[:, :rank]

    # As Q is orthogonal, the projection is
    beta = Q.T.dot(vectors)
    projection = Q.dot(beta)

    return projection


def project_lstsq(basis, vectors):
    """
    Project using least squares.

    :param basis: basis vectors to project on
    :type basis: numpy.ndarray
    :param vectors: vectors to project
    :type vectors: numpy.ndarray
    :return: projection
    :rtype: numpy.ndarray
    """

    x, _, _, _ = numpy.linalg.lstsq(basis, vectors)
    projection = basis.dot(x)

    return projection


def angles(vectors_a, vectors_b):
    """
    Compute angle between two sets of vectors.

    See https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf.

    :param vectors_a:
    :param vectors_b:
    :return:
    """

    if len(vectors_b.shape) == 1:
        vectors_b = vectors_b.reshape(-1, 1)

    # Normalize vector
    norms_a = numpy.linalg.norm(vectors_a, ord=2, axis=0)
    norms_b = numpy.linalg.norm(vectors_b, ord=2, axis=0)

    norms_a = numpy.repeat(norms_a.reshape(1, -1), vectors_a.shape[0], axis=0)
    norms_b = numpy.repeat(norms_b.reshape(1, -1), vectors_b.shape[0], axis=0)

    vectors_a /= norms_a
    vectors_b /= norms_b

    term_1 = numpy.multiply(vectors_a, norms_b) - numpy.multiply(vectors_b, norms_a)
    term_1 = numpy.linalg.norm(term_1, ord=2, axis=0)

    term_2 = numpy.multiply(vectors_a, norms_b) + numpy.multiply(vectors_b, norms_a)
    term_2 = numpy.linalg.norm(term_2, ord=2, axis=0)
    angles = 2*numpy.arctan2(term_1, term_2)

    return angles