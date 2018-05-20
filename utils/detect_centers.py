import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from skimage.io import imread, imsave, imshow
import skimage.measure
import tqdm


def find_centers(arg):
    """ Returns mask with Trues in object centers and average area of objects. """
    assert(arg.dtype == bool)
    labels = skimage.measure.label(arg)
    result = np.zeros_like(arg)
    areas = []
    for i in tqdm.trange(np.max(labels) + 1):
        mask = labels == i
        areas.append(np.count_nonzero(mask))
        grid = np.indices(mask.shape)
        x = np.mean(grid[0][mask])
        y = np.mean(grid[1][mask])
        result[int(x), int(y)] = 1
    return result, np.mean(areas)


def gaussian_peaks(peaks, sigma):
    assert(peaks.dtype == bool)
    grid = np.indices(peaks.shape)
    xs = grid[0][peaks]
    ys = grid[1][peaks]
    result = np.zeros(peaks.shape, dtype=float)
    for x, y in tqdm.tqdm(list(zip(xs, ys))):
        x2d, y2d = np.indices(peaks.shape)
        x2d -= x
        y2d -= y
        kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
        kernel_2d = kernel_2d / (2 * np.pi * sigma ** 2)
        result += kernel_2d
    return result


def gaussian_peaks_fast(peaks, sigma):
    """ Used memory is O(n * m * npeaks). """
    assert(peaks.dtype == bool)
    grid = np.indices(peaks.shape)
    meansx = grid[0][peaks]
    meansy = grid[1][peaks]
    npeaks = len(meansx)
    x2d, y2d = np.indices(peaks.shape)
    x2d = np.tile(x2d, (npeaks, 1, 1)) - meansx[:, np.newaxis, np.newaxis]
    y2d = np.tile(y2d, (npeaks, 1, 1)) - meansy[:, np.newaxis, np.newaxis]
    kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    return np.sum(kernel_2d, axis=0)


def gaussian_peaks_faster(peaks, sigma):
    assert(peaks.dtype == bool)
    blob_size = int(6 * sigma + 1)
    x2d, y2d = np.indices((blob_size, blob_size))
    x2d -= blob_size // 2
    y2d -= blob_size // 2
    blob = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    blob = blob / np.max(blob)

    result = scipy.signal.fftconvolve(peaks, blob, mode='same')
    return result


def convert_to_gaussian_blobs(input_filename, output_filename):
    image = imread(input_filename, as_grey=True) > 0.5
    peaks, area = find_centers(image)
    peaks = gaussian_peaks_faster(peaks, np.sqrt(area) / 6)
    peaks[peaks > 1.5] = 1.5
    peaks = (peaks * 255 / np.max(peaks)).astype(np.uint8)
    imsave(output_filename, peaks)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(
            'Usage: {me} input_file output_file'
            .format(me=os.path.basename(__file__))
        )
        return
    convert_to_gaussian_blobs(sys.argv[1], sys.argv[2])
