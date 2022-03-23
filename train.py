import time
import os
import argparse
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks


class ImageClassifier():
    def __init__(self, epochs=10, batch_size=32, target_size=128,
                 base_learning_rate=10**-3, architecture='EfficientNet',
                 dataset_dir='data', save_model=False, save_checkpoints=True,
                 es_patience=5, lr_patience=2, visualize=True, hflip=True,
                 vflip=True, zoom_range=.2, shear_range=.2, rotation_range=15
                 ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.target_size = (target_size, target_size)
        self.img_shape = self.target_size + (3,)
        self.base_learning_rate = base_learning_rate
        self.architecture = architecture
        self.dataset_dir = dataset_dir
        self.save_model = save_model
        self.save_checkpoints = save_checkpoints
        self.es_patience = es_patience
        self.lr_patience = lr_patience
        self.visualize = visualize
        self.hflip = hflip
        self.vflip = vflip
        self.zoom_range = zoom_range
        self.shear_range = shear_range
        self.rotation_range = rotation_range
        self.train_id = str(int(time.time()))

    def _prep_dataset(self):
        train_datagen = ImageDataGenerator(shear_range=self.shear_range,
                                           zoom_range=self.zoom_range,
                                           rotation_range=self.rotation_range,
                                           horizontal_flip=self.hflip,
                                           vertical_flip=self.vflip
                                           )
        val_datagen = ImageDataGenerator()
        self.train_generator = train_datagen.flow_from_directory(
                                os.path.join(self.dataset_dir, 'train'),
                                target_size=self.target_size,
                                batch_size=self.batch_size,
                                class_mode='binary')
        self.val_generator = val_datagen.flow_from_directory(
                                os.path.join(self.dataset_dir, 'validation'),
                                target_size=self.target_size,
                                batch_size=self.batch_size,
                                class_mode='binary')

    def _get_base_model(self):
        if 'efficient' in self.architecture.lower():
            from tensorflow.keras.applications.efficientnet \
                import EfficientNetB3
            self.base_model = EfficientNetB3(input_shape=self.img_shape,
                                             include_top=False,
                                             weights='imagenet')
            self.preprocess_input = \
                tf.keras.applications.efficientnet.preprocess_input
            self.official_architecture = 'EfficientNetB3'
            self.base_model.trainable = True
        elif 'inception' in self.architecture.lower():
            from tensorflow.keras.applications.inception_v3 \
                import InceptionV3
            self.base_model = InceptionV3(
                input_shape=self.img_shape,
                include_top=False,
                weights='imagenet'
                )
            self.preprocess_input = \
                tf.keras.applications.inception_v3.preprocess_input
            self.official_architecture = 'InceptionV3'
            self.base_model.trainable = True
        elif 'custom' in self.architecture.lower():
            self.official_architecture = 'CustomCNN'
            self.base_model.trainable = True
        else:
            raise NameError(f'''Could not find supported architecture matching
            '{self.architecture}' ''')

    def _connect_head(self):
        inputs = tf.keras.Input(shape=self.img_shape)
        x = self.preprocess_input(inputs)
        x = self.base_model(x, training=self.base_model.trainable)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        self.model = tf.keras.Model(inputs, outputs)

    def _compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.base_learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy', tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall()]
                        )

    def _train(self):
        my_callbacks = [callbacks.EarlyStopping(patience=self.es_patience,
                        restore_best_weights=True),
                        callbacks.ReduceLROnPlateau(
                            patience=self.lr_patience,
                            factor=.2,
                            min_lr=0.00001)]
        if self.save_checkpoints:
            my_callbacks.append(callbacks.ModelCheckpoint(
                os.path.join('checkpoints/',
                             f'{self.official_architecture}_{self.train_id}'),
                save_best_only=True))
        start = time.time()
        train_steps = int(len(self.train_generator.filepaths)/self.batch_size)
        val_steps = int(len(self.val_generator.filepaths)/self.batch_size)
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=train_steps,
            epochs=self.epochs,
            validation_data=self.val_generator,
            validation_steps=val_steps,
            callbacks=my_callbacks
                                      )
        end = time.time()
        print('Training Complete!')
        print(f'Duration of train: {(end-start)/60:.1f}minutes\n')

    def _save_model(self):
        if not os.path.isdir('./saved_models/'):
            os.mkdir('saved_models')
        save_path = os.path.join('saved_models',
                                 f'{self.used_architecture}_{self.train_id}')
        self.model.save(save_path)

    def _visualize(self):
        acc = self.history.history.get('accuracy')
        val_acc = self.history.history.get('val_accuracy')
        loss = self.history.history.get('loss')
        val_loss = self.history.history.get('val_loss')

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Accuracy Comparison')

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation loss')
        plt.legend(loc='lower right')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Loss Comparison')

    def train(self):
        self._get_base_model()
        self._prep_dataset()
        self._connect_head()
        self._compile()
        self._train()
        if self.save_model:
            self._save_model()
        if self.visualize:
            self._visualize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a convolutional neural network using Tensorflow')
    parser.add_argument('--arch', metavar='architecture', type=str, nargs='?',
                        default='efficientnet',
                        help='')
    parser.add_argument('--ep', metavar='epochs', type=int, nargs='?',
                        default='10',
                        help='')
    parser.add_argument('--bs', metavar='batch_size', type=int, nargs='?',
                        default=32,
                        help='')
    parser.add_argument('--lr', metavar='learning_rate', type=float, nargs='?',
                        default=0.001,
                        help='')
    parser.add_argument('--ts', metavar='target_size', type=int, nargs='?',
                        default=128,
                        help='')
    parser.add_argument('--dir', metavar='directory', type=str, nargs='?',
                        default='data/cats_v_dogs',
                        help='')
    parser.add_argument('--hflip', metavar='horizontal_flip', type=bool,
                        nargs='?', default=True,
                        help='')
    parser.add_argument('--vflip', metavar='vertical_flip', type=bool,
                        nargs='?', default=False,
                        help='')
    parser.add_argument('--zoom', metavar='zoom_range', type=float, nargs='?', 
                        default=.2,
                        help='')
    parser.add_argument('--shear', metavar='shear_range', type=float, 
                        nargs='?', default=.2,
                        help='')
    parser.add_argument('--rotation', metavar='rotation_range', type=int,
                        nargs='?', default=30,
                        help='')
    parser.add_argument('--es_patience', metavar='early_stopping_patience',
                        type=int, nargs='?', default=5,
                        help='')
    parser.add_argument('--lr_patience', metavar='learning_rate_patience',
                        type=int, nargs='?', default=2,
                        help='')
    parser.add_argument('--visualize', metavar='visualize', type=bool,
                        nargs='?', default=True,
                        help='')
    parser.add_argument('--save', metavar='save_model', type=bool, nargs='?',
                        default=False,
                        help='')
    parser.add_argument('--ckpt', metavar='save_checkpoints', type=bool,
                        nargs='?', default=True,
                        help='')

    args = parser.parse_args()

    classifier = ImageClassifier(epochs=args.ep,
                                 batch_size=args.bs,
                                 target_size=args.ts,
                                 base_learning_rate=args.lr,
                                 architecture=args.arch,
                                 dataset_dir=args.dir,
                                 hflip=args.hflip,
                                 vflip=args.vflip,
                                 zoom_range=args.zoom,
                                 shear_range=args.shear,
                                 rotation_range=args.rotation,
                                 es_patience=args.es_patience,
                                 lr_patience=args.lr_patience,
                                 visualize=args.visualize,
                                 save_model=args.save,
                                 save_checkpoints=args.ckpt)
    classifier.train()
