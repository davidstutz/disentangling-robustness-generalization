import numpy
import scipy
import scipy.stats


class PPCA:
    """
    Simple Probabilisitc PCA implementation in the form of sklearn's PCA.
    """

    def __init__(self, n_components=2, k_approximate_fraction=0.5):
        """
        Constructor.
        """

        self.n_components = n_components
        """ (int) Number of latent components. """

        self.k_approximate_fraction = k_approximate_fraction
        """ (float) Fraction of singular values to approximate. """

        self.mean = None
        """ (numpy.ndarray) Mean vector. """

        # In "Machine Learning and Pattern Recognition", V corresponds to the matrix W
        self.V = None
        """ (numpy.ndarray) Transformation matrix. """

        self.var = None
        """ (float) Variance. """

    def fit(self, data):
        """
        Fit PPCA.

        :param data: data, rows are individual data items
        :type data: numpy.ndarary
        """

        assert len(data.shape) == 2

        rows = data.shape[0]
        cols = data.shape[1]
        self.mean = numpy.mean(data.T, axis=1)

        # center
        means = numpy.repeat(numpy.expand_dims(self.mean, axis=0), rows, axis=0)
        data = data - means

        # singular values
        U, s, Vt = scipy.sparse.linalg.svds(data, k=self.n_components)
        approximate_k = max(1, min(int(self.k_approximate_fraction*min(rows, cols)), min(rows, cols) - 1))
        _, s_all, _ = scipy.sparse.linalg.svds(data, k=approximate_k)

        # singular values to eigenvalues
        e = s ** 2 / (data.shape[0] - 1)
        e_all = s_all ** 2 / (data.shape[0] - 1)

        # compute variance
        self.var = 1.0 / (data.shape[0] - self.n_components) * (numpy.sum(e_all) - numpy.sum(e))

        # compute V
        L_m = numpy.diag(e - numpy.ones((self.n_components)) * self.var) ** 0.5
        self.V = Vt.T.dot(L_m)

    def transform(self, data):
        """
        Transform given data.

        :param data: data, rows are individual data items
        :type data: numpy.ndarray
        :return: transformed data
        :rtype: numpy.ndarray
        """

        assert len(data.shape) == 2

        rows = data.shape[0]
        cols = data.shape[1]

        M = self.V.T.dot(self.V) + numpy.eye(self.V.shape[1]) * self.var
        M_inv = numpy.linalg.inv(M)

        means = numpy.repeat(numpy.expand_dims(self.mean, axis=0), rows, axis=0)
        codes = M_inv.dot(self.V.T.dot(data.T - means.T))

        return codes.T

    def inverse_transform(self, data):
        """
        Inverse transform for given data.

        :param data: data, rows are data items
        :type data: numpy.ndarray
        :return: inversely transformed data
        :rtype: numpy.ndarray
        """

        rows = data.shape[0]
        cols = data.shape[1]

        means = numpy.repeat(numpy.expand_dims(self.mean, axis=0), rows, axis=0)
        preds = numpy.dot(self.V, data.T) + means.T

        return preds.T

    def fit_transform(self, data):
        """
        Fit and transform.

        :param data: data, rows are individual data items
        :type data: numpy.ndarray
        :return: transformed data
        :rtype: numpy.ndarray
        """

        self.fit(data)
        return self.transform(data)

    def logmarginal(self, data):
        """
        Compute the marginal of the data.

        :param data: data, rows are data items
        :type data: numpy.ndarray
        :return: marginals
        :rtype: numpy.ndarray
        """

        rows = data.shape[0]
        cols = data.shape[1]

        C = self.V.dot(self.V.T) + numpy.eye(cols)*self.var
        return scipy.stats.multivariate_normal.logpdf(data, mean=self.mean, cov=C)

    def posterior(self):
        raise NotImplementedError()

    def likelihood(self, data, codes):
        """
        Compute the likelihood of the data.

        :param data: codes, transformed data
        :type data: numpy.ndarray
        :param data: true data (not reconstructed), rows are data items
        :type data: numpy.ndarray
        :return: marginals
        :rtype: numpy.ndarray
        """

        rows = data.shape[0]
        cols = data.shape[1]

        means = numpy.repeat(numpy.expand_dims(self.mean, axis=0), rows, axis=0)
        preds = numpy.dot(self.V, codes.T) + means.T

        mean = preds
        cov = numpy.eye(cols)*self.var

        return scipy.stats.multivariate_normal.logpdf(data, mean=mean, cov=cov)
