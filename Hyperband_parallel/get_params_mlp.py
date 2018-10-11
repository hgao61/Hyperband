from multiprocessing import Process, Queue
import os
import argparse

import numpy as np
import csv
import logging
from random import random
from math import log, ceil, floor
from time import time, ctime
from pprint import pprint
from time import sleep
from itertools import product
from ast import literal_eval
import _pickle as cPickle
import pickle
#from parallel_run import parallel_run
import operator
try:
    from hyperopt import hp
    from hyperopt.pyll.stochastic import sample
except ImportError:
    print ("In order to achieve operational capability, this programme requires hyperopt to be installed (pip install hyperopt), unless you make get_params() use something else.")

max_layers = 5
max_layer_size = 100
fn ='vgglog.csv'

space = {
    'scaler': hp.choice( 's', 
        ( None, 'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'MaxAbsScaler' )),
    'n_layers': hp.quniform( 'ls', 1, max_layers, 1 ),
    #'layer_size': hp.quniform( 'ls', 5, 100, 1 ),
    #'activation': hp.choice( 'a', ( 'relu', 'sigmoid', 'tanh' )),    
    'init': hp.choice( 'i', ( 'uniform', 'normal', 'glorot_uniform', 
        'glorot_normal', 'he_uniform', 'he_normal' )),
    'batch_size': hp.choice( 'bs', ( 16, 32, 64, 128, 256 )),
    'shuffle': hp.choice( 'sh', ( False, True )),
    'loss': hp.choice( 'l', ( 'mean_absolute_error', 'mean_squared_error' )),
    'optimizer': hp.choice( 'o', ( 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax' ))        
}

# for each hidden layer, we choose size, activation and extras individually
for i in range( 1, max_layers + 1 ):
    space[ 'layer_{}_size'.format( i )] = hp.quniform( 'ls{}'.format( i ), 
        2, max_layer_size, 1 )
    space[ 'layer_{}_activation'.format( i )] = hp.choice( 'a{}'.format( i ), 
        ( 'relu', 'sigmoid', 'tanh' ))
    space[ 'layer_{}_extras'.format( i )] = hp.choice( 'e{}'.format( i ), ( 
        { 'name': 'dropout', 'rate': hp.uniform( 'd{}'.format( i ), 0.1, 0.5 )}, 
        { 'name': 'batchnorm' },
        { 'name': None } ))    
# handle floats which should be integers
# works with flat params
def handle_integers( params ):

    new_params = {}
    for k, v in list(params.items()):
        if type( v ) == float and int( v ) == v:
            new_params[k] = int( v )
        else:
            new_params[k] = v
    
    return new_params
        
def get_params_mlp():

    params = sample( space )
    return handle_integers( params )
