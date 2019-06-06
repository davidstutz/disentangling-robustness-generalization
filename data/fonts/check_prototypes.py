import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')
from common import utils
from common.log import log, LogLevel
from common import paths
from common.fonts import get_fonts
from common import plot
import PIL, PIL.ImageFont, PIL.Image, PIL.ImageDraw, PIL.ImageChops, PIL.ImageOps
import string
import numpy
import argparse


class CheckPrototypes:
    """
    Creates a set of prototype fonts, i.e. renders the relevant characters of each font.
    Adapted from https://github.com/erikbern/deep-fonts.
    """

    def __init__(self, args=None):
        """
        Initialize.

        :param args: optional arguments if not to use sys.argv
        :type args: [str]
        """

        self.args = None
        """ Arguments of program. """

        parser = self.get_parser()
        if args is not None:
            self.args = parser.parse_args(args)
        else:
            self.args = parser.parse_args()

        log('-- ' + self.__class__.__name__)
        for key in vars(self.args):
            log('[Data] %s=%s' % (key, str(getattr(self.args, key))))

    def read_font(self, fn):
        """
        Read a font file and generate all letters in self.chars as images.

        :param fn: path to font file as TTF
        :rtype: str
        :return: images
        :rtype: numpy.ndarray
        """

        log('[Data] processing %s' % fn)

        # a bit smaller than image size to take transformations into account
        points = self.args.size - self.args.size/4
        font = PIL.ImageFont.truetype(fn, int(points))

        # some fonts do not like this
        # https://github.com/googlefonts/fontbakery/issues/703
        try:
            # https://stackoverflow.com/questions/43060479/how-to-get-the-font-pixel-height-using-pil-imagefont
            ascent, descent = font.getmetrics()
            (width, baseline), (offset_x, offset_y) = font.font.getsize('A')
        except IOError:
            return None

        data = []
        for char in self.args.characters:
            img = PIL.Image.new('L', (self.args.size, self.args.size), 255)
            draw = PIL.ImageDraw.Draw(img)
            textsize = draw.textsize(char, font=font)
            draw.text(((-offset_x + self.args.size - textsize[0])//2, (-offset_y + self.args.size - textsize[1])//2), char, font=font)

            matrix = numpy.array(img).astype(numpy.float32)
            matrix = 255 - matrix
            matrix /= 255.
            data.append(matrix)

        return numpy.array(data)

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Create HDF5 file of rendered font letters and digits.')
        parser.add_argument('-database_file', default=paths.database_file(), help='HDF5 file created.', type=str)
        parser.add_argument('-size', default=24, help='Width and height of images.', type=int)
        parser.add_argument('-characters', default='ABCDEFGHIJ', help='Characters for generate prototypes for.')

        return parser

    def main(self):
        """
        Main method.
        """

        ttfs = get_fonts()
        for i in range(900, len(ttfs)):
            fn = paths.data_file('fonts/fonts-master/%s' % ttfs[i], '')
            data = self.read_font(fn)
            if data is not None:
                plot.mosaic(str(i) + '.png', data, len(self.args.characters), title=os.path.basename(fn).replace('_', '\_'))


if __name__ == '__main__':
    program = CheckPrototypes()
    program.main()


