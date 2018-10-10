from multiprocessing import Queue, Process

import numpy as np
import os
"function (and parameter space) definitions for hyperband"
"regression with Keras (multilayer perceptron)"
import sys
from common_defs import *
from time import time, ctime

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization as BatchNorm
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import *
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

import keras
import pdb
from keras.layers.core import K  # import keras.backend as K
from keras.optimizers import Adam, Nadam, RMSprop
from hyperopt import STATUS_OK, STATUS_FAIL
import tensorflow as tf
#import uuid
import traceback
import vgg16

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

from keras.datasets import cifar10
from keras.utils import to_categorical
fn ='vgglog.csv'


NB_CLASSES_FINE = 10
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
    
def predict(xconfig):
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(xconfig[3])
    n_iterations =xconfig[1]
    xfile =xconfig[-1]
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    early_stopping = EarlyStopping( monitor = 'val_loss', patience = 5, verbose = 0 )


    x_train = x_train.astype('float32') / 255.0 - 0.5
    x_test = x_test.astype('float32') / 255.0 - 0.5
    y_train = to_categorical(y_train, NB_CLASSES_FINE)
    y_test = to_categorical(y_test, NB_CLASSES_FINE)
    gpu = xconfig [3]
    with tf.Session(graph=tf.Graph()) as sess:
        K.set_session(sess)
        with tf.device(gpu):
            #model = xnet
            model = vgg16(xfile)
            '''
            model = Sequential()

            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
        
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.25))
            model.add(Dense(10, activation='softmax'))

            model.compile(loss = 'mae', optimizer = 'adam', metrics = ["mae"])
            '''
    #model._make_predict_function()
            # Train net:
            model.fit(
                x_train,
                y_train,
                batch_size=128,  #int(xfile['batch_size']),
                epochs=n_iterations,
                shuffle=True,
                verbose=1,
                callbacks=[early_stopping],
                validation_data=(x_test, y_test)
            )
  

            #pdb.set_trace()
            #p = model.predict( x_train, batch_size = 128)#int(xfile['batch_size']) )
            scores = model.evaluate(x_train, y_train)

            print('Loss: %.3f' % scores[0])
            print('Accuracy: %.3f' % scores[1])
        
            #mse = MSE( y_train, p )
            #rmse = sqrt( mse )
            #mae = MAE( y_train, p )
            #print ("\n# training | RMSE: {:.4f}, MAE: {:.4f}".format( rmse, mae ))
            mae = scores[0]
            rmse = scores[1]
            mae = scores[0]
            return { 'loss': mae, 'rmse': rmse, 'mae': mae, 'early_stop': model.stop_training }

        
def vgg16_worker(xfile):
    #xnet = vgg16.vgg16(xfile[-1])
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(xfile[3])
    
    #print ('xnet is {}'.format(xnet))
    print('vggnet init done', xfile[3])
    start_time = time()
    label = predict( xfile)
    
    end_time = time()
    print('worker', xfile[3], ' xfile ', xfile, " predicted as label", label)

    dur_in_sec = int( round( time() - start_time ))
    xfile[2] = label['loss']
    #xfile[3] = self._gpuid
    xfile[5].append([label['loss'], start_time, end_time, dur_in_sec])
    '''
    with open(fn, 'a+') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(xfile)
    '''    
    #total.append([self._gpuid, xfile, label])
    print('vggnet done ', xfile[3])
        #print (total)
    return xfile


