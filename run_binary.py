import numpy as np
from skimage.io import imshow, imread
import pandas as pd

import keras.callbacks
import keras.utils

from model import get_model
from common import (
    bound_gpu_usage,
    get_rotated_subregion,
    augment_images
)


NAME = '01_demo'
LR = 0.000005
BATCH_SIZE = 16
EPOCHS = 128
NUM_CLASSES = 2
IMAGE_SHAPE = (256, 256)
AUGMENT = True

STEPS = 192
STEPS_VAL = 64


def image_generator(df):
    while True:
        image_index = np.random.randint(0, len(df))
        big_image = imread(df['path'][image_index])
        mask = imread(df['mask'][image_index], as_grey=True)

        angles = np.random.uniform(0, np.pi * 2, (BATCH_SIZE,))
        angles = np.random.uniform(0, np.pi * 2, (BATCH_SIZE,))
        shifts_x = np.random.uniform(0, mask.shape[0], (BATCH_SIZE,))
        shifts_y = np.random.uniform(0, mask.shape[1], (BATCH_SIZE,))

        images = np.array([get_rotated_subregion(
                big_image, IMAGE_SHAPE, angle, shift
            ) for angle, shift in zip(angles, zip(shifts_x, shifts_y))
        ])
        if AUGMENT:
            images = augment_images(images)
        Ys = np.array([get_rotated_subregion(mask, IMAGE_SHAPE,
            angle, shift) for angle, shift in zip(angles,
                zip(shifts_x, shifts_y))])
        if NUM_CLASSES > 2:
            labels = keras.utils.to_categorical(
                Ys.flatten(),
                NUM_CLASSES
            ).reshape(BATCH_SIZE, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUM_CLASSES)
        else:
            labels = Ys.reshape(BATCH_SIZE, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
        yield images, labels


def fit():
    model = get_model(NUM_CLASSES, LR)

    train_df = pd.read_csv('demo/train.tsv', sep='\t')
    val_df = pd.read_csv('demo/val.tsv', sep='\t')

    hist = model.fit_generator(
        image_generator(train_df),
        steps_per_epoch=STEPS,
        validation_data=image_generator(val_df),
        validation_steps=STEPS_VAL,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.TensorBoard(
                'logs/{}'.format(NAME),
                write_images=False,
                batch_size=BATCH_SIZE
            ),
            keras.callbacks.ModelCheckpoint(
                'models/{}.h5'.format(NAME), verbose=False,
                save_best_only=True, monitor='val_loss'
            )
        ]
    )


if __name__ == '__main__':
    bound_gpu_usage()
    fit()
