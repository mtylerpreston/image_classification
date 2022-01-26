import numpy as np
import os
import argparse


class Generator():
    def __init__(self, data_dir, val_size=.2, test_size=.2):
        self.data_dir = data_dir
        self.class_names = [i for i in os.listdir(data_dir)
                            if i not in ['train', 'validation',
                                         'test', '.DS_Store']]
        self.val_size = val_size
        self.test_size = test_size
        self.train_size = 1 - (self.val_size + self.test_size)

    def _set_up(self):
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.val_dir = os.path.join(self.data_dir, 'validation')
        self.test_dir = os.path.join(self.data_dir, 'test')

        if not os.path.isdir(self.train_dir):
            os.mkdir(self.train_dir)
        if not os.path.isdir(self.val_dir):
            os.mkdir(self.val_dir)
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        for name in self.class_names:
            train_subdir = os.path.join(self.train_dir, name)
            val_subdir = os.path.join(self.val_dir, name)
            test_subdir = os.path.join(self.test_dir, name)
            if not os.path.isdir(train_subdir):
                os.mkdir(train_subdir)
            if not os.path.isdir(val_subdir):
                os.mkdir(val_subdir)
            if not os.path.isdir(test_subdir):
                os.mkdir(test_subdir)

    def _generate(self):
        for name in self.class_names:
            subdir = os.path.join(self.data_dir, name)

            for filename in os.listdir(subdir):
                randn = np.random.rand()
                if randn <= self.val_size:
                    split = 'validation'
                elif (randn <= (self.test_size + self.val_size) and
                      randn > self.val_size):
                    split = 'test'
                elif randn >= (self.test_size + self.val_size):
                    split = 'train'

                current_path = os.path.join(self.data_dir, name, filename)
                dest_path = os.path.join(self.data_dir, split, name, filename)
                os.rename(current_path, dest_path)

    def generate(self):
        self._set_up()
        self._generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_size', metavar='Validation split size',
                        type=float, nargs='?', default='.2')
    parser.add_argument('--test_size', metavar='Test split size', type=float,
                        nargs='?', default='.2')
    parser.add_argument('--data_dir', metavar='data_directory', type=str,
                        nargs='?', default='data')

    args = parser.parse_args()

    generator = Generator(data_dir=args.data_dir,
                          val_size=args.val_size,
                          test_size=args.test_size)
    generator.generate()
