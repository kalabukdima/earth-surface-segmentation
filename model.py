import keras.layers
import keras.models
import keras.optimizers
import keras.backend as K


CONVS = 2
LAYERS = 6
FILTERS = 8


def get_model(NUM_CLASSES, LR):
    dataInput = keras.layers.Input(shape=(None, None, 3))
    x = dataInput
    # -------- Encoder --------
    lstMaxPools = []
    for cc in range(LAYERS):
        for ii in range(CONVS):
            x = keras.layers.Conv2D(filters=FILTERS * (2**cc),
                kernel_size=(3, 3),
                padding='same',
                activation='relu')(x)
        lstMaxPools.append(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    # -------- Decoder --------
    for cc in range(LAYERS):
        for ii in range(CONVS):
            x = keras.layers.Conv2D(filters=FILTERS * (2 ** (LAYERS - 1 - cc)),
                kernel_size=(3, 3),
                padding='same',
                activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
        if cc + 2 < LAYERS:
            x = keras.layers.concatenate([x, lstMaxPools[-1 - cc]], axis=-1)
    #
    for ii in range(CONVS):
        x = keras.layers.Conv2D(FILTERS, kernel_size=(3, 3), padding='same', activation='relu')(x)
    # 1x1 Convolution: emulation of Dense layer
    if NUM_CLASSES == 2:
        x = keras.layers.Conv2D(filters=1, kernel_size=(1,1), padding='valid', activation='sigmoid')(x)
        x = keras.layers.Lambda(lambda z: K.squeeze(z, -1))(x)
    else:
        x = keras.layers.Conv2D(filters=NUM_CLASSES, kernel_size=(1, 1),
                                padding='valid')(x)
        x = keras.layers.Activation('softmax')(x)
    model = keras.models.Model(dataInput, x)

    model.compile(keras.optimizers.adam(lr=LR), 'binary_crossentropy',
                  ['binary_accuracy'])
    return model
