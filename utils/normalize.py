import numpy as np
import scipy.signal
from skimage.io import imread, imsave
import sys

def get_circle_filter(r):
    z = np.zeros((2 * r - 1, 2 * r - 1))
    ci, cj = r - 1, r - 1
    I, J = np.meshgrid(np.arange(z.shape[0]), np.arange(z.shape[1]))
    dist_sq = (I - ci) ** 2 + (J - cj) ** 2
    z[np.where(dist_sq <= r ** 2)] = 1
    return z * 1. / np.sum(z)

def convolve_with_padding(image, filt):
    p = (filt.shape[0] + 1) // 2
    return scipy.signal.fftconvolve(
            np.pad(image, p, mode='edge').astype(np.float32),
            filt, mode='same'
            )[p:-p, p:-p]

def preprocess_normalize(image, r=64):
    circle = get_circle_filter(r)
    filt = -circle
    filt[(filt.shape[0] + 1) // 2, (filt.shape[1] + 1) // 2] += 1
    w = filt.shape[0]
    p = w / 2
    result = np.zeros(image.shape, dtype=np.float32)
    for channel in range(3):
        result[:, :, channel] = convolve_with_padding(image[:, :, channel],
                filt)
        variance = convolve_with_padding(image[:, :, channel] *
                image[:, :, channel], get_circle_filter(r))
        result[:, :, channel] /= np.sqrt(variance)
    result -= np.min(result)
    result /= np.max(result)
    return result

def main():
    assert(len(sys.argv) > 2)
    image = imread(sys.argv[1])
    result = preprocess_normalize(image)
    imsave(sys.argv[2], result)

if __name__ == '__main__':
    main()
