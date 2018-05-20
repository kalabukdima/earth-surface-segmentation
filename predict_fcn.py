import numpy as np
from skimage.io import imshow, imread, imsave
import keras.models

from common import bound_gpu_usage, get_cropped_image


NAME = '21_detect'
PATH = 'regions/binary/g01_g.tif'
OUTPUT = 'p20_02_gg.png'


def predict_with_model(model, image):
    assert(image.shape[0] % 64 == 0)
    assert(image.shape[1] % 64 == 0)
    return model.predict(np.array([image]))[0].reshape(
            image.shape[0], image.shape[1], -1)


def main():
    image = get_cropped_image(PATH)
    bound_gpu_usage()

    result = predict_with_model(
        keras.models.load_model('models/{}.h5'.format(NAME)),
        image
    )
    print('SHAPE:', result.shape)

    pic = np.zeros([result.shape[0], result.shape[1], 3])
    if result.shape[-1] == 1:
        pic[:, :, 0] = result[:, :, 0]
        pic[:, :, 1] = result[:, :, 0]
        pic[:, :, 2] = result[:, :, 0]
    else:
        pic[:, :, 0] = result[:, :, 1] + result[:, :, 3]
        pic[:, :, 1] = result[:, :, 0] + result[:, :, 1]
        pic[:, :, 2] = result[:, :, 2]

    imsave('results/{}'.format(OUTPUT), pic)

if __name__ == '__main__':
    main()
