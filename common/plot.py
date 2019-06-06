import os
from . import utils
from .log import log, LogLevel


if not utils.display():
    log('[Error] DISPLAY not found, plot not available!', LogLevel.ERROR)
    raise Exception("DISPLAY not found, plot not available!")


import math
import matplotlib
from matplotlib import pyplot
import numpy
import xgboost
import sklearn.svm
import sklearn.multioutput
import sklearn.decomposition
import sklearn.manifold
import scipy.stats
import umap as umap_

# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
# https://matplotlib.org/users/usetex.html
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
# https://matplotlib.org/users/customizing.html
matplotlib.rcParams['lines.linewidth'] = 1
# matplotlib.rcParams["figure.figsize"] = [12,10]

# from http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
color_brewer = numpy.array([
    [31, 120, 180],
    [251, 154, 153],
    [178, 223, 138],
    [51, 160, 44],
    [227, 26, 28],
    [253, 191, 111],
    [255, 127, 0],
    [202, 178, 214],
    [106, 61, 154],
    [245, 245, 145],
    [31, 120, 180],
    [251, 154, 153],
    [178, 223, 138],
    [51, 160, 44],
    [227, 26, 28],
    [253, 191, 111],
    [255, 127, 0],
    [202, 178, 214],
    [106, 61, 154],
    [255, 255, 153],
    [31, 120, 180],
    [251, 154, 153],
    [178, 223, 138],
    [51, 160, 44],
    [227, 26, 28],
    [253, 191, 111],
    [255, 127, 0],
    [202, 178, 214],
    [106, 61, 154],
    [255, 255, 153],
], dtype=float)
color_brewer /= 255.

marker_brewer = [
    '.', '.', '.', '.', '.', '.', '.', '.', '.', '.',
    '+', '+', '+', '+', '+', '+', '+', '+', '+', '+',
    '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
]


def label_save_and_close(filepath, show_legend=False, **kwargs):
    """
    Label axes, title etc.

    :param filepath: path to image file
    :type filepath: str
    """

    title = kwargs.get('title', None)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    xscale = kwargs.get('xscale', None)
    yscale = kwargs.get('yscale', None)

    if title is not None:
        pyplot.title(title)
    if xlabel is not None:
        pyplot.xlabel(xlabel)
    if ylabel is not None:
        pyplot.ylabel(ylabel)
    if xscale is not None:
        if xscale == 'symlog':
            pyplot.xscale(xscale, linthreshx=10 ** -10)
        else:
            pyplot.xscale(xscale)
    if yscale is not None:
        if yscale == 'symlog':
            pyplot.yscale(yscale, linthreshx=10 ** -10)
        else:
            pyplot.yscale(yscale)

    xmax = kwargs.get('xmax', None)
    xmin = kwargs.get('xmin', None)
    ymax = kwargs.get('ymax', None)
    ymin = kwargs.get('ymin', None)

    if xmax is not None:
        pyplot.gca().set_xbound(upper=xmax)
    if xmin is not None:
        pyplot.gca().set_xbound(lower=xmin)
    if ymax is not None:
        pyplot.gca().set_ybound(upper=ymax)
    if ymin is not None:
        pyplot.gca().set_ybound(lower=ymin)

    # This is fixed stuff.
    pyplot.grid(b=True, which='major', color=(0.5, 0.5, 0.5), linestyle='-')
    pyplot.grid(b=True, which='minor', color=(0.75, 0.75, 0.75), linestyle='--')

    utils.makedir(os.path.dirname(filepath))
    if show_legend:
        legend = pyplot.gca().legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        pyplot.savefig(filepath, bbox_extra_artists=(legend,), bbox_inches='tight')
    else:
        pyplot.savefig(filepath, bbox_inches='tight')

    # https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
    pyplot.close('all')


def image(filepath, image, cmap='binary', vmin=None, vmax=None, **kwargs):
    """
    Create an image.

    :param filepath: file to image
    :type filepath: str
    :param image: image
    :type image: numpy.ndarray
    :param cmap: color map to use, default is binary
    :type cmap: str
    """

    pyplot.imshow(image, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    label_save_and_close(filepath, False, **kwargs)


def mosaic(filepath, images, cols=6, cmap='binary', vmin=None, vmax=None, **kwargs):
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

    assert len(images.shape) == 3, 'set of images expected, rank 3 required'

    number = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]

    rows = int(math.ceil(number / float(cols)))
    image = numpy.zeros((rows * height, cols * width))

    for i in range(rows):
        for j in range(cols):
            k = i * cols + j
            if k < images.shape[0]:
                image[i * height: (i + 1) * height, j * width: (j + 1) * width] = images[k][:][:]

    pyplot.imshow(image, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    label_save_and_close(filepath, False, **kwargs)


def histogram(filepath, data, bins=50, labels=None, min_max=None, **kwargs):
    """
    Histogram plot.

    :param filepath: file to image
    :type filepath: str
    :param data: vector of data to plot
    :type data: numpy.ndarray
    :param bins: number of bins
    :type bins: int
    """

    assert len(data.shape) <= 2
    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    if labels is not None:
        assert len(labels) == data.shape[0]
    else:
        labels = list(map(str, range(data.shape[0])))

    for i in range(data.shape[0]):
        pyplot.hist(data[i], bins, range=min_max, edgecolor='black', linewidth=0.5, label=labels[i])
    label_save_and_close(filepath, False, **kwargs)


def line(filepath, x, y, labels=None, **kwargs):
    """
    Line plot.

    :param filepath: file to image
    :type filepath: str
    :param data: vector of data to plot
    :type data: numpy.ndarray
    :param labels: optional labels
    :type labels: [str]
    """

    if isinstance(x, numpy.ndarray):
        assert len(x.shape) == 1 or len(x.shape) == 2, ' only one- or two-dimensional data can be line-plotted'
        assert len(y.shape) == 1 or len(y.shape) == 2, ' only one- or two-dimensional data can be line-plotted'

        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if len(y.shape) == 1:
            y = y.reshape((1, -1))

        num_labels = x.shape[0]
    elif isinstance(x, list):
        assert isinstance(y, list)
        assert len(x) == len(y)
        for i in range(len(x)):
            assert x[i].shape[0] == y[i].shape[0]
        num_labels = len(x)

    has_labels = (labels is not None)
    if not has_labels:
        labels = [None] * num_labels
    assert len(labels) <= len(color_brewer), 'currently a maxmimum of %d different labels are supported' % len(color_brewer)

    for i in range(num_labels):
        pyplot.plot(x[i], y[i], color=tuple(color_brewer[i]), label=labels[i], marker=marker_brewer[i])
    pyplot.legend()
    label_save_and_close(filepath, has_labels, **kwargs)


def bar(filepath, x, y, **kwargs):
    """
    Bar plot.

    :param filepath: file to image
    :type filepath: strlist(x)
    :param data: vector of data to plot
    :type data: numpy.ndarray
    :param labels: optional labels
    :type labels: [str]
    """

    assert x.shape[0] == y.shape[0]

    pyplot.bar(x, y)
    label_save_and_close(filepath, False, **kwargs)


def matrix(filepath, data, x_labels=None, y_labels=None, **kwargs):
    """
    Matrix plot.

    :param filepath: file to image
    :type filepath: str
    :param data: data
    :type data: numpy.ndarray
    :param x_labels: labels on x axis
    :type x_labels: [str]
    :param y_labels: labels on y axis
    :type y_labels: [str]
    """

    cax = pyplot.matshow(data, interpolation='nearest', cmap='RdYlGn')
    pyplot.colorbar(cax)
    if x_labels:
        pyplot.gca().set_xticklabels([''] + x_labels)
    if y_labels:
        pyplot.gca().set_yticklabels([''] + y_labels)
    # https://stackoverflow.com/questions/34781096/matplotlib-matshow-with-many-string-labels
    pyplot.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    pyplot.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    label_save_and_close(filepath, False, **kwargs)


def surface(filepath, x, y, z, **kwargs):
    """
    Plot 3D surface.

    :param filepath: file to image
    :type filepath: str
    :param x: x values as grid
    :type x: numpy.ndarray
    :param y: y values as grid
    :type y: numpy.ndarray
    :param z: z values as grid
    :type z: numpy.ndarray
    """

    from mpl_toolkits.mplot3d import Axes3D
    #from matplotlib.ticker import LinearLocator, FormatStrFormatter

    fig = pyplot.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap='coolwarm', linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.view_init(60, 35)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    zlabel = kwargs.get('zlabel', None)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if zlabel is not None:
        ax.set_zlabel(zlabel)

    label_save_and_close(filepath, False, **kwargs)


def contour(filepath, x, y, z, **kwargs):
    """
    Plot 3D surface.

    :param filepath: file to image
    :type filepath: str
    :param x: x values as grid
    :type x: numpy.ndarray
    :param y: y values as grid
    :type y: numpy.ndarray
    :param z: z values as grid
    :type z: numpy.ndarray
    """

    # Plot the surface.
    pyplot.contour(x, y, z)

    label_save_and_close(filepath, False, **kwargs)


def scatter(filepath, x, y, c=None, labels=None, **kwargs):
    """
    Scatter plot or 2D data.

    :param filepath: file to image
    :type filepath: str
    :param x: x data
    :type x: numpy.ndarray
    :param y: y data
    :type y: numpy.ndarray
    :param c: labels as N x 1
    :type c: numpy.ndarray
    :param labels: label names
    :type labels: [str]
    """

    assert len(x.shape) == len(y.shape), 'only one dimensional data arrays supported'
    assert x.shape[0] == y.shape[0], 'only two-dimensional data can be scatter-plotted'
    assert c is None or x.shape[0] == c.shape[0], 'data and labels need to have same number of rows'
    if c is not None:
        assert labels is not None, 'if classes are given, labels need also to be given'

    if c is not None:
        if len(c.shape) > 1:
            c = numpy.squeeze(c)
    elif c is None:
        c = numpy.zeros((x.shape[0]))
        labels = [0]
    c = c.astype(int)  # Important for indexing

    unique_labels = numpy.unique(c)
    assert unique_labels.shape[0] <= len(color_brewer), 'currently a maxmimum of 12 different labels are supported'
    # assert unique_labels.shape[0] == len(labels), 'labels do not match given classes'
    assert numpy.min(unique_labels) >= 0 and numpy.max(unique_labels) < len(labels), 'classes contain elements not in labels'

    pyplot.figure(figsize=(8, 6))
    for i in range(unique_labels.shape[0]):
        pyplot.scatter(x[c == unique_labels[i]], y[c == unique_labels[i]],
                       c=numpy.repeat(numpy.expand_dims(color_brewer[i], 0), x[c == unique_labels[i]].shape[0], axis=0), marker=marker_brewer[i], s=45,
                       edgecolor='black', linewidth=0.5, label=labels[unique_labels[i]])
    label_save_and_close(filepath, (c != None), **kwargs)


def scatter2(filepath, x, y, labels=None, **kwargs):
    """
    Scatter plot or 2D data.

    :param filepath: file to image
    :type filepath: str
    :param x: x data
    :type x: numpy.ndarray
    :param y: y data
    :type y: numpy.ndarray
    :param labels: label names
    :type labels: [str]
    """

    if isinstance(x, numpy.ndarray):
        assert len(x.shape) == 1 or len(x.shape) == 2, ' only one- or two-dimensional data can be line-plotted'
        assert len(y.shape) == 1 or len(y.shape) == 2, ' only one- or two-dimensional data can be line-plotted'

        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if len(y.shape) == 1:
            y = y.reshape((1, -1))

        num_labels = x.shape[0]
    elif isinstance(x, list):
        assert isinstance(y, list)
        assert len(x) == len(y)
        for i in range(len(x)):
            assert x[i].shape[0] == y[i].shape[0]
        num_labels = len(x)

    has_labels = (labels is not None)
    if not has_labels:
        labels = [None] * num_labels
    assert len(labels) <= len(color_brewer), 'currently a maxmimum of %d different labels are supported' % len(color_brewer)

    pyplot.figure(figsize=(8, 6))
    for i in range(x.shape[0]):
        pyplot.scatter(x[i], y[i], c=color_brewer[i], marker=marker_brewer[i], s=45,
                       edgecolor='black', linewidth=0.5, label=labels[i])
    label_save_and_close(filepath, True, **kwargs)


def scatter2_fit(filepath, x, y, labels=None, **kwargs):
    """
    Scatter plot with line fit.

    :param filepath: file to image
    :type filepath: str
    :param x: x data
    :type x: numpy.ndarray
    :param y: y data
    :type y: numpy.ndarray
    :param labels: label names
    :type labels: [str]
        """

    if isinstance(x, numpy.ndarray):
        assert len(x.shape) == 1 or len(x.shape) == 2, ' only one- or two-dimensional data can be line-plotted'
        assert len(y.shape) == 1 or len(y.shape) == 2, ' only one- or two-dimensional data can be line-plotted'

        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if len(y.shape) == 1:
            y = y.reshape((1, -1))

        num_labels = x.shape[0]
    elif isinstance(x, list):
        assert isinstance(y, list)
        assert len(x) == len(y)
        for i in range(len(x)):
            assert x[i].shape[0] == y[i].shape[0]
        num_labels = len(x)

    has_labels = (labels is not None)
    if not has_labels:
        labels = [None] * num_labels
    assert len(labels) <= len(color_brewer), 'currently a maxmimum of %d different labels are supported' % len(color_brewer)

    pyplot.figure(figsize=(8, 6))
    for i in range(x.shape[0]):
        pyplot.scatter(x[i], y[i], c=color_brewer[i], marker=marker_brewer[i], s=45,
                       edgecolor='black', linewidth=0.5, label=labels[i])

    x = x.reshape(-1)
    y = y.reshape(-1)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    line = slope * x + intercept
    pyplot.plot(x, line)

    label_save_and_close(filepath, True, **kwargs)


def manifold(filepath, data, c, labels, method='tsne', pre_pca=20, **kwargs):
    """
    Manifold visualization.

    :param filepath: file to image
    :type filepath: str
    :param data: data as N x D
    :type data: numpy.ndarray
    :param c: labels as N x 1
    :type c: numpy.ndarray
    :param labels: label names
    :type labels: [str]
    :param method: string indicating manifold learning method to use
    :type method: str
    :param pre_pca: perform PCA down to pre_pca dimensions first
    :type pre_pca: int
    """

    method = method.lower()
    methods = [
        'pca', 'tsne', 'mds', 'umap', 'lle', 'ltsa', 'hlle', 'mlle'
    ]

    if labels is not None:
        assert data.shape[0] == labels.shape[0], 'data and colors/labels need to have same number of rows'
    assert method in methods, 'method not supported'

    if pre_pca is not None and pre_pca < data.shape[1] and method != 'pca':
        pca = sklearn.decomposition.IncrementalPCA(n_components=pre_pca)
        pca.fit(data)
        data = pca.transform(data)

    if method == 'pca':
        model = sklearn.decomposition.PCA(n_components=2)
    elif method == 'tsne':
        model = sklearn.manifold.TSNE(n_components=2)
    elif method == 'umap':
        model = umap_.UMAP()
    elif method == 'mds':
        model = sklearn.manifold.MDS(n_components=2)
    elif method == 'lle':
        model = sklearn.manifold.LocallyLinearEmbedding(10, 2, eigen_solver='auto', method='standard')
    elif method == 'ltsa':
        model = sklearn.manifold.LocallyLinearEmbedding(10, 2, eigen_solver='auto', method='ltsa')
    elif method == 'hlle':
        model = sklearn.manifold.LocallyLinearEmbedding(10, 2, eigen_solver='auto', method='hessian')
    elif method == 'mlle':
        model = sklearn.manifold.LocallyLinearEmbedding(10, 2, eigen_solver='auto', method='modified')
    else:
        raise NotImplementedError()

    data_2d = model.fit_transform(data)
    scatter(filepath, data_2d[:, 0], data_2d[:, 1], c, labels, **kwargs)


class ManifoldVisualization:
    """
    Manifold visualization using some standard dimensionality reduction methods.
    """

    def __init__(self, method='umap', pre_pca=20, fit_method='svr'):
        """
        Constructor.

        :param method: string indicating manifold learning method to use
        :type method: str
        :param pre_pca: perform PCA down to pre_pca dimensions first
        :type pre_pca: int
        :param fit_method: method to fit the remaining data
        :type fit_method: str
        """

        method = method.lower()
        methods = [
            'pca', 'tsne', 'mds', 'umap', 'lle', 'ltsa', 'hlle', 'mlle'
        ]

        assert method in methods, 'method not supported'

        self.method = method
        """ (str) Manifold method. """

        self.pre_pca = pre_pca
        """ (int) PCA dimensionality reduction beforehand. """

        fit_method = fit_method.lower()
        fit_methods = [
            'svr', 'xgb'
        ]

        assert fit_method in fit_methods, 'fit_method not supported'

        self.fit_method = fit_method
        """ (str) Method to generalize to testing data. """

        self.pca = None
        """ (sklearn.decomposition.IncrementalPCA) Pre-PCA model. """

        self.model = None
        """ (None) Fitted model. """

    def fit(self, data):
        """
        Fit the dimensionality reduction on the given data.

        :param data: data as N x D
        :type data: numpy.ndarray
        """

        # PCA is the only method that can be used for new test images!
        if self.method == 'pca':
            self.pca = sklearn.decomposition.PCA(n_components=2)
            self.pca.fit(data)
        else:
            if self.pre_pca is not None and self.pre_pca < data.shape[1]:
                self.pca = sklearn.decomposition.IncrementalPCA(n_components=self.pre_pca)
                self.pca.fit(data)
                data = self.pca.transform(data)

            if self.method == 'tsne':
                model = sklearn.manifold.TSNE(n_components=2)
            elif self.method == 'umap':
                model = umap_.UMAP()
            elif self.method == 'mds':
                model = sklearn.manifold.MDS(n_components=2)
            elif self.method == 'lle':
                model = sklearn.manifold.LocallyLinearEmbedding(10, 2, eigen_solver='auto', method='standard')
            elif self.method == 'ltsa':
                model = sklearn.manifold.LocallyLinearEmbedding(10, 2, eigen_solver='auto', method='ltsa')
            elif self.method == 'hlle':
                model = sklearn.manifold.LocallyLinearEmbedding(10, 2, eigen_solver='auto', method='hessian')
            elif self.method == 'mlle':
                model = sklearn.manifold.LocallyLinearEmbedding(10, 2, eigen_solver='auto', method='modified')
            else:
                raise NotImplementedError()

            # Strategy as in https://lvdmaaten.github.io/publications/papers/AISTATS_2009.pdf
            data_2d = model.fit_transform(data)

            if self.fit_method == 'xgb':
                fit_model = xgboost.XGBRegressor(n_estimators=2000, max_depth=20, learning_rate=0.01, verbose=True)
                fit_model = sklearn.multioutput.MultiOutputRegressor(fit_model)
            elif self.fit_method == 'svr':
                fit_model = sklearn.svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
                fit_model = sklearn.multioutput.MultiOutputRegressor(fit_model)
            else:
                raise NotImplementedError()

            fit_model.fit(data, data_2d)
            self.model = fit_model

    def visualize(self, filepath, data, c, labels=None, **kwargs):
        """
        Visualize given data.

        :param filepath: file to image
        :type filepath: str
        :param data: data as N x D
        :type data: numpy.ndarray
        :param c: labels as N x 1
        :type c: numpy.ndarray
        :param labels: label names
        :type labels: [str]
        """

        if self.method == 'pca':
            assert self.pca is not None, 'call fit first'
            data_2d = self.pca.transform(data)
        else:
            assert self.model is not None, 'call fit first'
            if self.pca is not None:
                data = self.pca.transform(data)
            data_2d = self.model.predict(data)

        scatter(filepath, data_2d[:, 0], data_2d[:, 1], c, labels, **kwargs)