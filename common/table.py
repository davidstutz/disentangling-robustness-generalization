import os
from . import utils
from .log import log, LogLevel


if not utils.display():
    log('[Error] DISPLAY not found, plot not available!', LogLevel.ERROR)
    raise Exception("DISPLAY not found, plot not available!")


from . import latex as ltx
import numpy
import terminaltables
from matplotlib import pyplot as plt
import matplotlib
from common import paths

#matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
matplotlib.rc('text', usetex=True)
# https://matplotlib.org/users/customizing.html
matplotlib.rcParams['lines.linewidth'] = 1


def ascii(x, y, labels):
    """
    ASCII table.

    :param x: x data
    :type x: numpy.ndarray
    :param y: y data, corresponding to multiple "lines"
    :type y: numpy.ndarray
    :param labels: labels for different lines
    :type labels: [str]
    """

    if isinstance(x, numpy.ndarray):
        assert len(x.shape) == 1, ' only one-dimensional data can be line-plotted'
        assert len(y.shape) == 1 or len(y.shape) == 2, ' only one- or two-dimensional data can be line-plotted'

        x = x
        if len(y.shape) == 1:
            y = y.reshape((1, -1))

        num_labels = y.shape[0]
    elif isinstance(x, list):
        raise NotImplementedError()

    if labels is None:
        labels = [None]*num_labels

    table_data = []
    table_data.append([''] + ['%.3g' % value for value in x.tolist()])
    for i in range(num_labels):
        table_data.append([labels[i]] + ['%.3g' % value for value in y[i].tolist()])
    table = terminaltables.AsciiTable(table_data)
    return table.table


def pyplot(filepath, x, y, labels, **kwargs):
    """
    Pyplot table.

    :param filepath: path to file
    :type filepath: str
    :param x: x data
    :type x: numpy.ndarray
    :param y: y data, corresponding to multiple "lines"
    :type y: numpy.ndarray
    :param labels: labels for different lines
    :type labels: [str]
    """

    if isinstance(x, numpy.ndarray):
        assert len(x.shape) == 1, ' only one-dimensional data can be line-plotted'
        assert len(y.shape) == 1 or len(y.shape) == 2, ' only one- or two-dimensional data can be line-plotted'

        x = x
        if len(y.shape) == 1:
            y = y.reshape((1, -1))

        num_labels = y.shape[0]
    elif isinstance(x, list):
        raise NotImplementedError()

    if labels is None:
        labels = [None]*num_labels

    columns = ['%.3g' % value for value in x.tolist()]
    rows = labels

    table_data = []
    for i in range(num_labels):
        table_data.append(['%.3g' % value for value in y[i].tolist()])

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    #ax.axis('tight')

    ax.table(cellText=table_data, rowLabels=rows, colLabels=columns, loc='center')

    # Adjust layout to make room for the table:
    title = kwargs.get('title', None)

    if title is not None:
        fig.title(title)

    fig.tight_layout()
    utils.makedir(os.path.dirname(filepath))
    plt.savefig(filepath)

    # https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
    plt.close('all')


def latex(filepath, x, y, labels, **kwargs):
    """
    LaTeX table.

    :param filepath: path to file
    :type filepath: str
    :param x: x data
    :type x: numpy.ndarray
    :param y: y data, corresponding to multiple "lines"
    :type y: numpy.ndarray
    :param labels: labels for different lines
    :type labels: [str]
    """

    ltx.table(filepath, x, y, labels, **kwargs)
