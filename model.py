import numpy as np
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras import backend
from keras.initializers import he_normal, he_uniform

import common

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + backend.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + backend.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+backend.epsilon()))


def unet(input_size, pretrained_weights=None, seed=common.SEED, unif_init=True):
    kinit = he_uniform(seed=seed) if unif_init else he_normal(seed=seed)
    print("[MODEL] Using kernel initializer {} with seed {}".format(
        "he_uniform" if unif_init else "he_normal", seed))

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kinit)(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kinit)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kinit)(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kinit)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kinit)(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kinit)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kinit)(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kinit)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=kinit)(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=kinit)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=kinit)(UpSampling2D(size=(2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kinit)(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kinit)(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=kinit)(UpSampling2D(size=(2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kinit)(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kinit)(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=kinit)(UpSampling2D(size=(2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kinit)(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kinit)(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=kinit)(UpSampling2D(size=(2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kinit)(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kinit)(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=kinit)(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()

    return model
