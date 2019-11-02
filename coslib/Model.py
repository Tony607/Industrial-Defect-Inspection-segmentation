from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from scipy.ndimage.measurements import label
import time


# for command line arguments
import argparse

from coslib import load_images

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np


def load_data(args):
    '''
    '''
    defect_imgs = load_images(args)

    X_train, X_valid = train_test_split()

    return X_train, X_valid


def build_model():
    '''
    '''
    with tf.name_scope('encoder'):
        x = Convolution2D(128, (1, 1), activation='elu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Convolution2D(96, (3, 3), activation='elu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Convolution2D(80, (3, 3), activation='elu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Convolution2D(64, (1, 1), activation='elu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Convolution2D(48, (3, 3), activation='elu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Convolution2D(36, (3, 3), activation='elu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
    with tf.name_scope('decoder'):
        x = Convolution2D(36, (3, 3), activation='elu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(48, (3, 3), activation='elu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(64, (1, 1), activation='elu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(80, (3, 3), activation='elu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(96, (3, 3), activation='elu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(128, (1, 1), activation='elu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Convolution2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='nadam', loss='mean_squared_error')

    return model


def build_small_unet(img_rows, img_cols):
    """build u-net model

    Args:
        img_rows (int):
        img_cols (int):

    Return:

    Notes:
    """
    inputs = Input((img_rows, img_cols, 1))
    inputs_norm = Lambda(lambda x: x / 127.5 - 1.)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4)], mode = 'concat', concat_axis = 3)
    conv6=Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(up6)
    conv6=Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(conv6)

    up7=merge([UpSampling2D(size=(2, 2)(conv6)), conv3], mode = 'concat', concat_axis = 3)
    conv7=Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(up7)
    conv7=Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(conv7)

    up8=merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode = 'concat', concat_axis = 3)
    conv8=Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(up8)
    conv8=Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(conv8)

    up9=merge([UpSampling2D(size=(2, 2))(conv8), conv1], model = 'concat', concat_axis = 3)
    conv9=Conv2D(8, (3, 3), activation = 'relu', padding = 'same')(up9)
    conv9=Conv2D(8, (3, 3), activation = 'relu', padding = 'same')(conv9)

    conv10=Conv2D(1, (1, 1), activation = 'sigmoid')(conv9)

    model=Model(inputs = inputs, outputs = conv10)

    return model





def train_model(model, args, X_train, X_valid):

    # set up the checkpoint for the model
    checkpoint=ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor = 'val_loss',
                                 verbose = 0,
                                 save_best_only = args.save_best_only,
                                 mode = 'auto')

    # compoile the model
    model.compile(loss = 'mean_squared_error', optimizer = Adam(lr=args.learning_rate)
    # fitting the model
    model.fit(np.reshape(X_train (len(X_train), X_train.shape[0]), X_train.shpae[1]),
              batch_size=args.samples_per_epoch,
              )


    return model

def main():
    """
    Load training data and training the model
    """

    ap = argparse.ArgumentParser('Cosmetics Inspection with AutoEncoder')
    ap.add_argument()
