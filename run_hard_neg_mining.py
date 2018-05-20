import numpy as np
from skimage.io import imshow, imread
import pandas as pd

import keras.callbacks
import keras.utils

from model import get_model
from common import (
    bound_gpu_usage,
    get_rotated_subregion,
    get_cropped_image,
    augment_images
)
from predict_fcn import predict_with_model


NAME = '20_hard_negative_mining'
LR = 0.00002
BATCH_SIZE = 16
EPOCHS = 128
NUM_CLASSES = 2
IMAGE_SHAPE = (256, 256)
AUGMENT = False

STEPS = 192
STEPS_VAL = 64


model = None

train_df = pd.read_csv('datasets/train.tsv', sep='\t')
val_df = pd.read_csv('datasets/val.tsv', sep='\t')

train_true_masks = {
    path: np.zeros(get_cropped_image(path).shape[:2], dtype=bool)
    for path in train_df['path']
}


def write_mask(image):
    i = 0
    pattern = 'tmp/{}.png'
    while os.path.exists(pattern.format(i)):
        i += 1
    path = pattern.format(i)
    imsave(path, image)


def recalc_true_mask():
    THRESHOLD = 0.5
    new_dict = {}
    global train_true_masks
    wrote = False
    for path in train_true_masks.keys():
        mask_path = train_df[train_df['path'] == path]['mask'].iloc[0]
        image = get_cropped_image(path)
        gt = get_cropped_image(mask_path, as_grey=True) > 0
        result = predict_with_model(model, image)[..., 0]
        result = result > THRESHOLD
        new_dict[path] = (result ^ gt).astype(bool)
        if not wrote:
            write_mask(new_dict[path])
            wrote = True
    train_true_masks = new_dict


def image_generator(df):
    iteration = 0
    while True:
        iteration += 1

        image_index = np.random.randint(0, len(df))
        path = df['path'][image_index]
        big_image = get_cropped_image(path)
        mask = get_cropped_image(df['mask'][image_index], as_grey=True)

        assert(len(mask.shape) == 2)
        if path in train_true_masks:
            false_prediction_positions = np.array(list(np.ndindex(mask.shape)))[
                ~train_true_masks[path].flatten()
            ]
            indexes = np.random.choice(len(false_prediction_positions), BATCH_SIZE)
            positions = false_prediction_positions[indexes]
            shifts_x, shifts_y = zip(*positions)
        else:
            shifts_x = np.random.uniform(0, mask.shape[0], (BATCH_SIZE,))
            shifts_y = np.random.uniform(0, mask.shape[1], (BATCH_SIZE,))

        angles = np.random.uniform(0, np.pi * 2, (BATCH_SIZE,))
        angles = np.random.uniform(0, np.pi * 2, (BATCH_SIZE,))

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
    global model
    model = get_model(NUM_CLASSES, LR)

    for epoch in range(EPOCHS):
        model.fit_generator(
            image_generator(train_df),
            steps_per_epoch=STEPS,
            validation_data=image_generator(val_df),
            validation_steps=STEPS_VAL,
            epochs=1,
            initial_epoch=epoch,
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
        recalc_true_mask()


if __name__ == '__main__':
    bound_gpu_usage()
    fit()
