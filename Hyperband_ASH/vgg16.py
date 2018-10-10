"function (and parameter space) definitions for hyperband"
"regression with Keras (multilayer perceptron)"

from common_defs import *
import keras
# a dict with x_train, y_train, x_test, y_test

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization as BatchNorm
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import *
from keras.optimizers import Adam, Nadam, RMSprop

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10

from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
#import tensorflow as tf

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

# print hidden layers config in readable way
def print_layers( params ):
    for i in range( 1, params['n_layers'] + 1 ):
        print ("layer {} | size: {:>3} | activation: {:<7} | extras: {}".format( i,
            params['layer_{}_size'.format( i )], 
            params['layer_{}_activation'.format( i )],
            params['layer_{}_extras'.format( i )]['name'] ), end = ' ')
        if params['layer_{}_extras'.format( i )]['name'] == 'dropout':
            print ("- rate: {:.1%}".format( params['layer_{}_extras'.format( i )]['rate'] ), end = ' ')
        print()


def print_params( params ):
    pprint({ k: v for k, v in params.items() if not k.startswith( 'layer_' )})
    print_layers( params )
    print()


def vgg16(params=None):
    print ('Hyperspace testing: ', params)
    """Create model according to the hyperparameter space given."""


    model = Sequential()

    model.add(Conv2D(params['units1'], kernel_size=(3, 3), activation=params['layer_1_activation'], input_shape=(32, 32, 3)))
    model.add(Conv2D(params['units1'], kernel_size=(3, 3), activation=params['layer_1_activation']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout1']))

    model.add(Conv2D(params['units2'], kernel_size=(3, 3), activation=params['layer_2_activation']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(params['units2'], kernel_size=(3, 3), activation=params['layer_2_activation']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout2']))
        
    model.add(Flatten())
    model.add(Dense(params['units3'], activation='relu'))
    model.add(Dropout(params['dropout3']))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss = 'mae', optimizer = 'adam', metrics = ["mae"])
    model.summary()
    return model
    
    
