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

max_layers = 13

max_layer_size = 100
fn ='vgglog.csv'
    
space  = { 'choice': hp.choice('layers_number',
                             [{'layers': 'two'},
                             {'layers': 'three',
                             'units3': hp.choice('units3', [32, 64, 128]),
                             'dropout3': hp.choice('dropout3', np.linspace(0.1, 0.3, 3, dtype=float))
                             }]),

            'units1': hp.choice('units1', [32, 64, 128, 256,512]),
            'units2': hp.choice('units2', [32, 64, 128, 256]),
            'units3': hp.choice('units3', [32, 64, 128]), 

            'dropout1': hp.choice('dropout1', np.linspace(0.25, 0.75, 3, dtype=float)),
            'dropout2': hp.choice('dropout2', np.linspace(0.25, 0.75, 3, dtype=float)),
            'dropout3': hp.choice('dropout3', np.linspace(0.1, 0.3, 3, dtype=float)),

            'batch_size' : hp.choice('batch_size', [16, 32, 64, 128, 256, 512]),
            
        }      
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
        
def get_params_vgg16():

    params = sample( space )
    return handle_integers( params )
