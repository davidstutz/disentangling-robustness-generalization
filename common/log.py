import io
import os
import sys
import datetime
from enum import Enum
from .timer import Timer


class LogLevel(Enum):
    """
    Defines log level.
    """

    INFO = 1
    WARNING = 2
    ERROR = 3


class Log:
    """
    Simple singleton log implementation with different drivers.
    """

    instance = None
    """ (Log) Log instance. """

    def __init__(self):
        """
        Constructor.
        """

        self.files = dict()
        """ ([file]) Files to write log to (default is sys.stderr). """

        self.verbose = LogLevel.INFO
        """ (LogLevel) Verbosity level. """

        self.silent = True if 'ROBUSTNESS_SILENT' in os.environ else False
        """ (bool) Whether to be silent in the console. """

    class LogMessage:
        """
        Wrap a simple message.
        """

        def __init__(self, message, level=LogLevel.INFO, end="\n"):
            """
            Constructor.

            :param message: actual message
            :type message: str
            :param end: line ending
            :type end: str
            """

            self.message = message
            """ (str) Message. """

            self.level = level
            """ (LogLevel) Level. """

            self.end = end
            """ (str) End of line. """

            self.timer = Timer()

            #              0           1 = INFO    2 = WARNING 3 = ERROR
            self.colors = ['\033[94m', '\033[94m', '\033[93m', '\033[91m\033[1m']
            """ ([str]) Level colors. """

        def timestamp(self):
            """
            Print timestamp.

            :return: date and time
            :rtype: str
            """

            dt = datetime.datetime.now()
            return '[%s] ' % dt.strftime('%d%m%y%H%M')

        def __enter__(self):
            """
            Enter.
            """

            files = Log.get_instance()._files()
            for key in files.keys():
                files[key].write(self.timestamp() + str(self.message))
                files[key].flush()

            if not Log.get_instance().silent:
                sys.stderr.write(self.colors[self.level.value])
                sys.stderr.write(self.timestamp() + str(self.message))
                sys.stderr.write('\033[0m')

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Close files.
            """

            files = Log.get_instance()._files()
            for key in files.keys():
                files[key].write(str(' [%g]' % self.timer.elapsed()))
                files[key].write(str(self.end))
                files[key].flush()

            if not Log.get_instance().silent:
                sys.stderr.write(self.colors[self.level.value])
                sys.stderr.write(str(' [%g]' % self.timer.elapsed()))
                sys.stderr.write('\033[0m')
                sys.stderr.write(str(self.end))

        def dispatch(self):
            """
            Simply write log messahe.
            """

            files = Log.get_instance()._files()
            for key in files.keys():
                files[key].write(self.timestamp() + str(self.message))
                files[key].write(str(self.end))
                files[key].flush()

            if not Log.get_instance().silent:
                sys.stderr.write(self.colors[self.level.value])
                sys.stderr.write(self.timestamp() + str(self.message))
                sys.stderr.write('\033[0m')
                sys.stderr.write(str(self.end))

    def __del__(self):
        """
        Close files.
        """

        keys = list(self.files.keys()) # Force a list instead of an iterator!
        for key in keys:
            if isinstance(self.files[key], io.TextIOWrapper):
                self.files[key].close()
                del self.files[key]

    def attach(self, file):
        """
        Attach a file to write to.
        :param file: log file
        :type file: file
        """

        self.files[file.name] = file

    def detach(self, key):
        """
        Detach a key.

        :param key: log file name
        :type key: str
        """

        assert isinstance(key, str)
        if key in self.files.keys():
            if isinstance(self.files[key], io.TextIOWrapper):
                self.files[key].close()
                del self.files[key]

    def _files(self):
        """
        Get files.

        :return: files
        :rtype: [File]
        """

        return self.files

    @staticmethod
    def get_instance():
        """
        Get current log instance, simple singleton.
        :return: log
        :rtype: Log
        """

        if Log.instance is None:
            Log.instance = Log()

        return Log.instance

    def verbose(self, level=LogLevel.INFO):
        """
        Sets the log verbostiy.

        :param level: minimum level to report
        :return: LogLevel
        """

        self.verbose = level

    def log(self, message, level=LogLevel.INFO, end="\n"):
        """
        Log a message.

        :param message: message or variable to log
        :type message: mixed
        :param level: level, i.e. color
        :type level: LogColor
        :param end: whether to use carriage return
        :type end: str
        """

        if level.value >= self.verbose.value:
            return Log.LogMessage(message, level, end)


def log(message, level=LogLevel.INFO, end="\n"):
    """
    Quick access to logger instance.

    :param message: message or variable to log
    :type message: mixed
    :param level: level, i.e. color
    :type level: LogColor
    :param end: whether to use carriage return
    :type end: str
    """

    Log.get_instance().log(message, level, end).dispatch()


def logw(message, level=LogLevel.INFO, end="\n"):
    """
    Quick access to logger instance for with statements.

    :param message: message or variable to log
    :type message: mixed
    :param level: level, i.e. color
    :type level: LogColor
    :param end: whether to use carriage return
    :type end: str
    """

    return Log.get_instance().log(message, level, end)