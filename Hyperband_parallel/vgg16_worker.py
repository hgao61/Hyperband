from multiprocessing import Queue, Process
import cv2
import numpy as np
import os
"function (and parameter space) definitions for hyperband"
"regression with Keras (multilayer perceptron)"
import sys
from common_defs import *
from time import time, ctime
import csv
# a dict with x_train, y_train, x_test, y_test
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization as BatchNorm
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import *

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
#load models
import keras

from keras.layers.core import K  # import keras.backend as K
from keras.optimizers import Adam, Nadam, RMSprop
import tensorflow as tf
from hyperopt import STATUS_OK, STATUS_FAIL

import uuid
import traceback
import os
import vgg16
from keras.datasets import cifar10
from keras.utils import to_categorical
fn ='vgglog.csv'
NB_CHANNELS = 3
IMAGE_BORDER_LENGTH = 32
# NB_CLASSES = 10
NB_CLASSES_FINE = 10


        
class Vgg16Worker(Process):
    def __init__(self, gpuid, queue):
        Process.__init__(self, name='ModelProcessor')
        self._gpuid = gpuid
        self._queue = queue

    def run(self):

        #set enviornment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)
        
        #total=[]
        while True:
            xfile = self._queue.get()
            if xfile == None:
                self._queue.put(None)
                break
            xnet = vgg16.Vgg16(xfile[-1])
            print ('xnet is {}'.format(xnet))
            print('vggnet init done', self._gpuid)
            start_time = time()
            label = self.predict(xnet, xfile)
            end_time = time()
            print('worker', self._gpuid, ' xfile ', xfile, " predicted as label", label)
            #total = [self._gpuid,label, xfile]
            dur_in_sec = int( round( time() - start_time ))
            xfile[2] = label
            xfile[3] = self._gpuid
            xfile[4] = [start_time, end_time, dur_in_sec]

            with open(fn, 'a+') as log_file:
                csv_writer = csv.writer(log_file)
                csv_writer.writerow(xfile)
            #total.append([self._gpuid, xfile, label])
        print('vggnet done ', self._gpuid)
        #print (total)

    def predict(self, xnet, xconfig):
        n_iterations =xconfig[1]
        xfile =xconfig[-1]
        '''
        if xfile['scaler']:
            scaler = eval( "{}()".format( xfile['scaler'] ))
            x_train_ = scaler.fit_transform( data['x_train'].astype( float ))
            x_test_ = scaler.transform( data['x_test'].astype( float ))
        else:
            x_train_ = data['x_train']
            x_test_ = data['x_test']
        '''
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        #print('x_train shape is {}'.format(x_train.shape))
        #print('x_train shape[1] is {}'.format(x_train.shape[1]))
        validation_data = ( x_test, y_test )
        early_stopping = EarlyStopping( monitor = 'val_loss', patience = 5, verbose = 0 )


        x_train = x_train.astype('float32') / 255.0 - 0.5
        x_test = x_test.astype('float32') / 255.0 - 0.5
        y_train = to_categorical(y_train, NB_CLASSES_FINE)
        y_test = to_categorical(y_test, NB_CLASSES_FINE)
        model = xnet
            # Train net:
        history = model.fit(
            x_train,
            y_train,
            batch_size=int(xfile['batch_size']),
            epochs=n_iterations,
            shuffle=True,
            verbose=1,
            callbacks=[early_stopping],
            validation_data=(x_test, y_test)
        ).history
  
        '''
        
        history = model.fit( x_train/255.0, to_categorical(y_train),
            epochs = int( round( n_iterations )),
            batch_size = xfile['batch_size'], 
            #shuffle = xfile['shuffle'], 
            validation_data = validation_data)#, 
            #callbacks = [ early_stopping ])    
        '''  
        #
  
        p = model.predict( x_train, batch_size = xfile['batch_size'] )
        scores = model.evaluate(x_train, y_train)

        print('Loss: %.3f' % scores[0])
        print('Accuracy: %.3f' % scores[1])
        
        mse = MSE( y_train, p )
        rmse = sqrt( mse )
        mae = MAE( y_train, p )
        print ("\n# training | RMSE: {:.4f}, MAE: {:.4f}".format( rmse, mae ))
    
        return { 'loss': mae, 'rmse': rmse, 'mae': mae, 'early_stop': model.stop_training }
