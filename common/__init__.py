import sys
from .log import log, LogLevel

# Make sure that all the stuff is called using Python 3:
if sys.version_info[0] < 3:
    log('[Error] python3 required!', LogLevel.ERROR)
    raise Exception("Python 3 or a more recent version has to be used!")

# https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variablehttps://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import matplotlib
matplotlib.use('Agg')