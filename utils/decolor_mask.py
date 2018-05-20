import numpy as np
from skimage.io import imread, imsave
import sys
import os


def main():
    if len(sys.argv) < 3:
        print(
            'Usage: {me} input_file output_file'
            .format(me=os.path.basename(__file__))
        )
        return
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    mask = imread(input_filename)
    gray_mask = np.zeros(mask.shape[0] * mask.shape[1], dtype=mask.dtype)
    gray_mask[((mask[:, :, 1] >= 128) &\
        (mask[:, :, 0] < 128)).flatten()] = 1
    gray_mask[((mask[:, :, 0] >= 128) &\
        (mask[:, :, 1] < 128)).flatten()] = 2
    gray_mask = gray_mask.reshape(mask.shape[:2])
    assert(gray_mask.shape[0] == mask.shape[0] and
           gray_mask.shape[1] == mask.shape[1])
    imsave(output_filename, gray_mask)


if __name__ == '__main__':
    main()
