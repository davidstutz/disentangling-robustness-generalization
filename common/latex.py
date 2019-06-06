import os
import numpy
from . import paths
import scipy.stats


def table(filepath, x, y, labels, **kwargs):
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

    if isinstance(x, numpy.ndarray):
        assert len(x.shape) == 1, ' only one-dimensional data can be line-plotted'
        assert len(y.shape) == 1 or len(y.shape) == 2, ' only one- or two-dimensional data can be line-plotted'

        x = x
        if len(y.shape) == 1:
            y = y.reshape((1, -1))

        num_labels = y.shape[0]
    elif isinstance(x, list):
        raise NotImplementedError()

    latex =  '\documentclass{standalone}\n'
    latex += '\\begin{document}\n'

    latex += '\\begin{tabular}{r || ' + ' | '.join(['c' for value in x.tolist()]) + '}\n'
    latex += '&' + ' & '.join(['%.3g' % value for value in x.tolist()]) + '\\\\\\hline\\hline\n'
    for i in range(num_labels):
        latex += labels[i] + '&' + ' & '.join(['%.3g' % value for value in y[i].tolist()]) + '\\\\\\hline\n'
    latex += '\\end{tabular}\n'
    latex += '\\end{document}\n'

    with open(filepath, 'w') as f:
        f.write(latex)

    os.system('pdflatex --output-directory=%s %s' % (os.path.dirname(filepath), filepath))
    os.system('convert -density 150 %s -quality 90 %s' % (os.path.splitext(filepath)[0] + paths.PDF_EXT, os.path.splitext(filepath)[0] + paths.PNG_EXT))


def line(filepath, x, y, labels=None, c=None, **kwargs):
    """
    LaTeX line plot.

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

    if c is None:
        c = x

    if labels is None:
        labels = [None]*num_labels

    latex = '\\documentclass{standalone}\n'

    default_options = {
        'height': '5cm',
        'width': '5cm',
        'legend style': '{at={(1.05,1)},anchor=north west}',
        'x label style': '{at={(axis description cs:0.5,0)},anchor=north}',
        'y label style': '{at={(axis description cs:0.25,0.5)},anchor=south}',
    }
    options = {**default_options, **kwargs}

    packages = ['pgfplots', 'tikz']
    for package in packages:
        latex += '\\usepackage{%s}\n' % package
    latex += '\\begin{document}\n'
    latex += '\\begin{tikzpicture}\n'
    latex += '\\begin{axis}[%s]\n' % ','.join(['%s=%s' % (key, str(value)) for key, value in options.items()])

    for i in range(num_labels):
        latex += '\\addplot[%s] coordinates {\n%s\n};\n' % (labels[i], '\n'.join('(%s, %s) %% %s' % (x[i][j], y[i][j], c[i][j]) for j in range(x[i].shape[0])))
        latex += '\\addlegendentry{%s};\n' % labels[i]

    latex += '\\end{axis}\n'
    latex += '\\end{tikzpicture}\n'
    latex += '\\end{document}\n'

    with open(filepath, 'w') as f:
        f.write(latex)

    #os.system('pdflatex --output-directory=%s %s' % (os.path.dirname(filepath), filepath))
    #os.system('convert -density 150 %s -quality 90 %s' % (os.path.splitext(filepath)[0] + paths.PDF_EXT, os.path.splitext(filepath)[0] + paths.PNG_EXT))


def histogram(filepath, data, bins=50, labels=None, normalized=False, min_max=None, **kwargs):
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

    latex = '\\documentclass{standalone}\n'

    default_options = {
        'height': '5cm',
        'width': '5cm',
        'legend style': '{at={(1.05,1)},anchor=north west}',
        'x label style': '{at={(axis description cs:0.5,0)},anchor=north}',
        'y label style': '{at={(axis description cs:0.25,0.5)},anchor=south}',
        'ymin': '0',
        'area style': True
    }
    options = {**default_options, **kwargs}

    packages = ['pgfplots', 'tikz']
    for package in packages:
        latex += '\\usepackage{%s}\n' % package
    latex += '\\begin{document}\n'
    latex += '\\begin{tikzpicture}\n'
    latex += '\\begin{axis}[%s]\n' % ','.join(['%s=%s' % (key, str(value)) for key, value in options.items()])

    for i in range(data.shape[0]):
        min_max_i = min_max
        if min_max is None:
            min_max_i = (numpy.min(data[i]), numpy.max(data[i]))

        y, x = numpy.histogram(data[i], bins, range=min_max_i)
        if normalized:
            y = y/numpy.sum(y)

        latex += '\\addplot[ybar interval,mark=no] coordinates {\n%s\n};\n' % '\n'.join('(%s, %s)' % (x[j], y[j]) for j in range(y.shape[0]))
        latex += '\\addlegendentry{%s};\n' % labels[i]

    latex += '\\end{axis}\n'
    latex += '\\end{tikzpicture}\n'
    latex += '\\end{document}\n'

    with open(filepath, 'w') as f:
        f.write(latex)

    #os.system('pdflatex --output-directory=%s %s' % (os.path.dirname(filepath), filepath))
    #os.system('convert -density 150 %s -quality 90 %s' % (os.path.splitext(filepath)[0] + paths.PDF_EXT, os.path.splitext(filepath)[0] + paths.PNG_EXT))


def scatter(filepath, x, y, labels=None, c=None, **kwargs):
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

    latex = '\\documentclass{standalone}\n'

    default_options = {
        'height': '5cm',
        'width': '5cm',
        'legend style': '{at={(1.05,1)},anchor=north west}',
        'x label style': '{at={(axis description cs:0.5,0)},anchor=north}',
        'y label style': '{at={(axis description cs:0.25,0.5)},anchor=south}',
    }
    options = {**default_options, **kwargs}

    packages = ['pgfplots', 'tikz']
    for package in packages:
        latex += '\\usepackage{%s}\n' % package
    latex += '\\begin{document}\n'
    latex += '\\begin{tikzpicture}\n'
    latex += '\\begin{axis}[%s]\n' % ','.join(['%s=%s' % (key, str(value)) for key, value in options.items()])

    for i in range(unique_labels.shape[0]):
        x_ = x[c == unique_labels[i]]
        y_ = y[c == unique_labels[i]]
        latex += '\\addplot[only marks,%s,mark=triangle*,mark size=1pt] coordinates {\n%s\n};\n' % (labels[i], '\n'.join('(%s, %s)' % (x_[j], y_[j]) for j in range(x_.shape[0])))
        latex += '\\addlegendentry{%s};\n' % labels[i]

    latex += '\\end{axis}\n'
    latex += '\\end{tikzpicture}\n'
    latex += '\\end{document}\n'

    with open(filepath, 'w') as f:
        f.write(latex)

    #os.system('pdflatex --output-directory=%s %s' % (os.path.dirname(filepath), filepath))
    #os.system('convert -density 150 %s -quality 90 %s' % (os.path.splitext(filepath)[0] + paths.PDF_EXT, os.path.splitext(filepath)[0] + paths.PNG_EXT))


def bar(filepath, x, y, **kwargs):
    """
    LaTeX line plot.

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

    latex = '\\documentclass{standalone}\n'

    default_options = {
        'height': '5cm',
        'width': '5cm',
        'legend style': '{at={(1.05,1)},anchor=north west}',
        'x label style': '{at={(axis description cs:0.5,0)},anchor=north}',
        'y label style': '{at={(axis description cs:0.25,0.5)},anchor=south}',
    }
    options = {**default_options, **kwargs}

    packages = ['pgfplots', 'tikz']
    for package in packages:
        latex += '\\usepackage{%s}\n' % package
    latex += '\\begin{document}\n'
    latex += '\\begin{tikzpicture}\n'
    latex += '\\begin{axis}[%s]\n' % ','.join(['%s=%s' % (key, str(value)) for key, value in options.items()])

    for i in range(num_labels):
        latex += '\\addplot+[ybar] coordinates {\n%s\n};\n' % ('\n'.join('(%s, %s)' % (x[i][j], y[i][j]) for j in range(x[i].shape[0])))

    latex += '\\end{axis}\n'
    latex += '\\end{tikzpicture}\n'
    latex += '\\end{document}\n'

    with open(filepath, 'w') as f:
        f.write(latex)

    #os.system('pdflatex --output-directory=%s %s' % (os.path.dirname(filepath), filepath))
    #os.system('convert -density 150 %s -quality 90 %s' % (os.path.splitext(filepath)[0] + paths.PDF_EXT, os.path.splitext(filepath)[0] + paths.PNG_EXT))


def scatter2(filepath, x, y, labels=None, c=None, **kwargs):
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

    if c is None:
        c = x

    has_labels = (labels is not None)
    if not has_labels:
        labels = [None] * num_labels

    latex = '\\documentclass{standalone}\n'

    default_options = {
        'height': '5cm',
        'width': '5cm',
        'legend style': '{at={(1.05,1)},anchor=north west}',
        'x label style': '{at={(axis description cs:0.5,0)},anchor=north}',
        'y label style': '{at={(axis description cs:0.25,0.5)},anchor=south}',
    }
    options = {**default_options, **kwargs}

    packages = ['pgfplots', 'tikz']
    for package in packages:
        latex += '\\usepackage{%s}\n' % package
    latex += '\\begin{document}\n'
    latex += '\\begin{tikzpicture}\n'
    latex += '\\begin{axis}[%s]\n' % ','.join(['%s=%s' % (key, str(value)) for key, value in options.items()])

    for i in range(num_labels):
        latex += '\\addplot[only marks,%s,mark=triangle*,mark size=1pt] coordinates {\n%s\n};\n' % (labels[i], '\n'.join('(%s, %s) %% %s' % (x[i][j], y[i][j], c[i][j]) for j in range(x[i].shape[0])))
        latex += '\\addlegendentry{%s};\n' % labels[i]

    latex += '\\end{axis}\n'
    latex += '\\end{tikzpicture}\n'
    latex += '\\end{document}\n'

    with open(filepath, 'w') as f:
        f.write(latex)

    #os.system('pdflatex --output-directory=%s %s' % (os.path.dirname(filepath), filepath))
    #os.system('convert -density 150 %s -quality 90 %s' % (os.path.splitext(filepath)[0] + paths.PDF_EXT, os.path.splitext(filepath)[0] + paths.PNG_EXT))


def scatter2_fit(filepath, x, y, labels=None, c=None, **kwargs):
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

    if c is None:
        c = x

    has_labels = (labels is not None)
    if not has_labels:
        labels = [None] * num_labels

    latex = '\\documentclass{standalone}\n'

    default_options = {
        'height': '5cm',
        'width': '5cm',
        'legend style': '{at={(1.05,1)},anchor=north west}',
        'x label style': '{at={(axis description cs:0.5,0)},anchor=north}',
        'y label style': '{at={(axis description cs:0.25,0.5)},anchor=south}',
    }
    options = {**default_options, **kwargs}

    packages = ['pgfplots', 'tikz']
    for package in packages:
        latex += '\\usepackage{%s}\n' % package
    latex += '\\begin{document}\n'
    latex += '\\begin{tikzpicture}\n'
    latex += '\\begin{axis}[%s]\n' % ','.join(['%s=%s' % (key, str(value)) for key, value in options.items()])

    for i in range(num_labels):
        latex += '\\addplot[only marks,%s,mark=triangle*,mark size=1pt] coordinates {\n%s\n};\n' % (labels[i], '\n'.join('(%s, %s) %% %s' % (x[i][j], y[i][j], c[i][j]) for j in range(x[i].shape[0])))
        latex += '\\addlegendentry{%s};\n' % labels[i]

    x = x.reshape(-1)
    y = y.reshape(-1)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    line = slope * x + intercept
    latex += '\\addplot coordinates {\n%s\n};\n' % '\n'.join('(%s, %s)' % (x[j], line[j]) for j in range(x.shape[0]))
    latex += '\\addlegendentry{%s};\n' % labels[i]

    latex += '\\end{axis}\n'
    latex += '\\end{tikzpicture}\n'
    latex += '\\end{document}\n'

    with open(filepath, 'w') as f:
        f.write(latex)

    #os.system('pdflatex --output-directory=%s %s' % (os.path.dirname(filepath), filepath))
    #os.system('convert -density 150 %s -quality 90 %s' % (os.path.splitext(filepath)[0] + paths.PDF_EXT, os.path.splitext(filepath)[0] + paths.PNG_EXT))
