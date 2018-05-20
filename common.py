import numpy as np
import cv2
from skimage.io import imread
from imgaug import augmenters as iaa
import keras.backend as K


def bound_gpu_usage():
    config = K.tf.ConfigProto(
        gpu_options=K.tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
    K.tensorflow_backend.set_session(K.tf.Session(config=config))


def get_cropped_image(path, as_grey=False):
    image = imread(path, as_grey=as_grey)
    image = image[:64 * (image.shape[0] // 64), :64 * (image.shape[1] // 64)]
    return image


def get_rotated_subregion(image, shape, angle, shift=(0, 0), scale=1):
    radius = max(shape) / 2 ** 0.5
    shift = (min(max(shift[1], radius), image.shape[1] - radius),
             min(max(shift[0], radius), image.shape[0] - radius))
    cosa = np.cos(angle)
    sina = np.sin(angle)

    matShiftB = np.array([[1., 0., -shift[0]], [0., 1., -shift[1]], [0., 0., 1.]])
    matRot = np.array([[cosa, sina, 0.], [-sina, cosa, 0.], [0., 0., 1.]])
    matShiftF = np.array([[1., 0., shape[0] / 2.], [0., 1., shape[1] / 2.], [0., 0., 1.]])
    matScale = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., 1.]])
    matTotal = matShiftF.dot(matRot.dot(matScale.dot(matShiftB)))

    return cv2.warpAffine(image, matTotal[:2, :], shape)


aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Multiply((0.9, 1.1), per_channel=0.6),
        iaa.Sometimes(0.4,
            iaa.GaussianBlur(sigma=(0, 0.5))
        )
    ])


augment_images = aug.augment_images
