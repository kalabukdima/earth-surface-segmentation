import sys
import numpy as np
from skimage.io import imread

THRESHOLD = 0.5

def get_bin_image(filename):
    image = imread(filename, as_grey=True)
    assert 0 <= np.min(image) <= 1
    image = image > THRESHOLD
    return image


def crop_center(image, shape):
    assert(len(shape) == 2)
    for i in range(2):
        assert(image.shape[i] >= shape[i])

    offset_i = (image.shape[0] - shape[0]) // 2
    offset_j = (image.shape[1] - shape[1]) // 2
    return image[offset_i:offset_i + shape[0],
                 offset_j:offset_j + shape[1]]


def iou(filename1, filename2):
    image1, image2 = map(get_bin_image, (filename1, filename2))
    print(image1.shape)
    print(image2.shape)
    if image2.shape[0] > image1.shape[0]:
        image1, image2 = image2, image1
    image1 = crop_center(image1, image2.shape)
    print(np.count_nonzero(image1 & image2) * 1.\
        / np.count_nonzero(image1 | image2))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Pass 2 filenames.')
        exit(0)
    print(iou(sys.argv[1], sys.argv[2]))
