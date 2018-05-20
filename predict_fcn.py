import numpy as np
from skimage.io import imshow, imread, imsave
import keras.models
import argparse

from common import bound_gpu_usage, get_cropped_image


def predict_with_model(model, image):
    assert(image.shape[0] % 64 == 0)
    assert(image.shape[1] % 64 == 0)
    return model.predict(np.array([image]))[0].reshape(
            image.shape[0], image.shape[1], -1)


def main(model, in_filename, out_filename):
    image = get_cropped_image(in_filename)
    bound_gpu_usage()

    result = predict_with_model(
        keras.models.load_model('models/{}.h5'.format(model)),
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

    imsave(out_filename, pic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, help='model name')
    parser.add_argument('--input', '-i', type=str, help='path to input image')
    parser.add_argument('--output', '-o', type=str, help='path to store result')
    args = parser.parse_args()
    main(args.model, args.input, args.output)
