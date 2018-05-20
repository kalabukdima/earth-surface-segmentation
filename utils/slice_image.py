import numpy as np
import pandas as pd
import skimage.io
import sys
import os
import warnings


def generate_monoclass_chunks(mask, size, step, coverage=0.9):
    """Gets subimages which are mostly covered with a single label.

    Args:
    size (int) :
        Consider subimages size*size pixels.
    step (int) :
        Consider subimages, which left and top edge coords are multiples of
        step.
    coverage (float):
        Assign image to class if relative amount of this class in image is at
        least coverage.

    Returns:
        list of ((x, y), label), where (x, y) are coords of upper left corner
        of found subimage.
    """
    for x in range(0, mask.shape[0] - size, step):
        for y in range(0, mask.shape[1] - size, step):
            submask = mask[x:x + size, y:y + size]
            bincount = np.bincount(submask.flatten()).astype(float) *\
                (1. / size / size)
            for (label, freq) in enumerate(bincount):
                if freq >= coverage:
                    yield ((x, y), label)


def process_big_image(path, path_to_mask, path_to_output_dir, sizes=(64, 128),
                      step=32):
    """Divides image into small images and saves them as separate files.
    Args:
        path_to_output_dir (str) : where to store resulting files.

    Returns:
        pandas.DataFrame with info about every small image.
    """
    image = skimage.io.imread(path)
    mask = skimage.io.imread(path_to_mask)
    data = []
    for size in sizes:
        dir_path = os.path.join(path_to_output_dir, str(size))
        sys.stderr.write('Saving to {}.\n'.format(dir_path))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for ((x, y), label) in generate_monoclass_chunks(mask, size, step):
            subimage = image[x:x + size, y:y + size]
            name = '{}_{}.jpg'.format(str(x // step).zfill(3),
                                      str(y // step).zfill(3))
            path = os.path.join(dir_path, name)
            skimage.io.imsave(path, subimage)
            data.append({'path': path, 'label': label, 'x': x, 'y': y,
                         'size': size})
            sys.stderr.write('{} {}\r'.format(x, y))
    return pd.DataFrame(data, columns=['path', 'label', 'size', 'x', 'y'])


def main():
    if len(sys.argv) != 4:
        print('Usage: slice_image.py path_to_image path_to_mask \
path_to_output_dir')
        return
    _, image_path, mask_path, output_dir = sys.argv
    warnings.simplefilter('ignore', UserWarning)
    df = process_big_image(image_path, mask_path, output_dir, (64,))
    df.to_csv(sys.stdout, index=False)


if __name__ == '__main__':
    main()
