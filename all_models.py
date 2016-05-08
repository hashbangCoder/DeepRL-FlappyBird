## Define all NN models
from six.moves import cPickle
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D,MaxPooling1D
from keras.layers.normalization import BatchNormalization

from keras.callbacks import ModelCheckpoint


## Model with NO dropout NO batchnorm - https://github.com/yenchenlin1994/DeepLearningFlappyBird
def model_default(input_shape):
    model = Sequential()
    model.add(Convolution2D(32,8,8,subsample=(4,4), border_mode='same',init='he_uniform',input_shape=input_shape))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64,4,4, subsample=(2,2),border_mode='same' , init='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64,3,3, subsample=(1,1),border_mode='same' , init='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Dense(2, init='he_uniform'))
   
    return model
    
    
# Model WITH BATCHNORM NO MAXPOOL NO Dropout
def model_bnorm(input_shape):
    model = Sequential()
    model.add(Convolution2D(32,8,8, border_mode='same' , init='he_uniform',input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Convolution2D(64,4,4, border_mode='same' , init='he_uniform'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64,3,3, border_mode='same' , init='he_uniform'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64,3, 3, border_mode='same' , init='he_uniform'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(256, init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, init='he_uniform'))
    
    return model
    
## Model with NO dropout and NO stride NO maxpool NO batchnorm
#def model_no_MaxPool(input_shape):
#    model = Sequential()
#    model.add(Convolution2D(32,8,8, border_mode='same' , init='he_uniform',input_shape=input_shape))
#    model.add(Activation('relu'))

#    model.add(Convolution2D(64,4,4, border_mode='same' , init='he_uniform'))
#    model.add(Activation('relu'))

#    model.add(Convolution2D(64,3,3, border_mode='same' , init='he_uniform'))
#    model.add(Activation('relu'))

#    model.add(Convolution2D(64,3, 3, border_mode='same' , init='he_uniform'))
#    model.add(Activation('relu'))

#    model.add(Flatten())
#    model.add(Dense(256, init='he_uniform'))
#    model.add(Activation('relu'))
#    model.add(Dense(2, init='he_uniform'))
#    model.add(Activation('softmax'))
#    return model
