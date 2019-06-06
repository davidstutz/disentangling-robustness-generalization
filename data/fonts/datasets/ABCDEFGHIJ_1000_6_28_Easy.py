import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../../')
from common import utils
import math
from data.fonts.generate_prototypes import *
from data.fonts.generate_codes import *
from data.fonts.generate_dataset import *
from data.fonts.split_dataset import *
from data.fonts.check_dataset import *


class ABCDEFGHIJ_1000_6_28_Easy:
    """
    Dataset definition.
    """

    def main(self):
        """
        Main.
        """

        fonts = 1000
        characters = 'ABCDEFGHIJ'
        transformations = 6
        size = 28
        suffix = 'Easy'
        N_train = 960000

        min_scale = 0.9
        max_scale = 1.1
        min_rotation = -math.pi/6
        max_rotation = math.pi/6
        min_translation = -0.1
        max_translation = 0.1
        min_shear = -0.2
        max_shear = 0.2
        multiplier = 112
        batch_size = 100

        GeneratePrototypes([
            '-database_file=%s' % paths.database_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-font_number=%d' % fonts,
            '-size=%d' % size,
            '-characters=%s' % characters,
        ]).main()
        GenerateCodes([
            '-database_file=%s' % paths.database_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-codes_file=%s' % paths.codes_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-theta_file=%s' % paths.theta_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-number_transformations=%d' % transformations,
            '-min_scale=%g' % min_scale,
            '-max_scale=%g' % max_scale,
            '-min_rotation=%g' % min_rotation,
            '-max_rotation=%g' % max_rotation,
            '-min_translation=%g' % min_translation,
            '-max_translation=%g' % max_translation,
            '-min_shear=%g' % min_shear,
            '-max_shear=%g' % max_shear,
            '-multiplier=%d' % multiplier
        ]).main()
        GenerateDataset([
            '-database_file=%s' % paths.database_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-codes_file=%s' % paths.codes_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-theta_file=%s' % paths.theta_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-batch_size=%d' % batch_size,
            '-images_file=%s' % paths.images_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix)
        ]).main()
        SplitDataset([
            '-codes_file=%s' % paths.codes_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-theta_file=%s' % paths.theta_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-images_file=%s' % paths.images_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-train_codes_file=%s' % paths.train_codes_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-test_codes_file=%s' % paths.test_codes_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-train_theta_file=%s' % paths.train_theta_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-test_theta_file=%s' % paths.test_theta_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-train_images_file=%s' % paths.train_images_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-test_images_file=%s' % paths.test_images_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-N_train=%d' % N_train
        ]).main()
        if utils.display():
            from data.fonts.inspect_dataset import InspectDataset
            InspectDataset([
                '-images_file=%s' % paths.images_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
                '-output_directory=%s' % paths.data_file('fonts/', '', characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix)
            ]).main()
        CheckDataset([
            '-database_file=%s' % paths.database_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-codes_file=%s' % paths.codes_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-theta_file=%s' % paths.theta_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-images_file=%s' % paths.images_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-train_codes_file=%s' % paths.train_codes_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-test_codes_file=%s' % paths.test_codes_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-train_theta_file=%s' % paths.train_theta_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-test_theta_file=%s' % paths.test_theta_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-train_images_file=%s' % paths.train_images_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix),
            '-test_images_file=%s' % paths.test_images_file(characters=characters, fonts=fonts, transformations=transformations, size=size, suffix=suffix)
        ]).main()


if __name__ == '__main__':
    program = ABCDEFGHIJ_1000_6_28_Easy()
    program.main()