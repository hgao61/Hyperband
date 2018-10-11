from multiprocessing import Process, Queue
import os
import argparse
from vgg16_worker import Vgg16Worker
import numpy as np

import logging
from random import random
from math import log, ceil
from time import time, ctime

from time import sleep
from itertools import product
#from parallel_run import parallel_run

T=[{'regressor': 'keras_mlp', 'batch_size': 128, 'init': 'glorot_normal', 'layer_1_activation': 'sigmoid', 'layer_1_extras': {'name': 'dropout', 'rate': 0.4514185384729481}, 'layer_1_size': 73, 'layer_2_activation': 'tanh', 'layer_2_extras': {'name': None}, 'layer_2_size': 17, 'layer_3_activation': 'relu', 'layer_3_extras': {'name': None}, 'layer_3_size': 42, 'layer_4_activation': 'tanh', 'layer_4_extras': {'name': 'batchnorm'}, 'layer_4_size': 40, 'layer_5_activation': 'tanh', 'layer_5_extras': {'name': None}, 'layer_5_size': 16, 'loss': 'mean_squared_error', 'n_layers': 3, 'optimizer': 'rmsprop', 'scaler': 'MaxAbsScaler', 'shuffle': False},
{'regressor': 'keras_mlp', 'batch_size': 128, 'init': 'uniform', 'layer_1_activation': 'relu', 'layer_1_extras': {'name': None}, 'layer_1_size': 97, 'layer_2_activation': 'sigmoid', 'layer_2_extras': {'name': None}, 'layer_2_size': 81, 'layer_3_activation': 'relu', 'layer_3_extras': {'name': 'batchnorm'}, 'layer_3_size': 20, 'layer_4_activation': 'relu', 'layer_4_extras': {'name': 'batchnorm'}, 'layer_4_size': 34, 'layer_5_activation': 'sigmoid', 'layer_5_extras': {'name': 'dropout', 'rate': 0.17307905834175275}, 'layer_5_size': 3, 'loss': 'mean_absolute_error', 'n_layers': 2, 'optimizer': 'adagrad', 'scaler': 'MaxAbsScaler', 'shuffle': False},
   {'regressor': 'keras_mlp', 'batch_size': 32, 'init': 'glorot_uniform', 'layer_1_activation': 'relu', 'layer_1_extras': {'name': 'dropout', 'rate': 0.39559835470634763}, 'layer_1_size': 69, 'layer_2_activation': 'sigmoid', 'layer_2_extras': {'name': None}, 'layer_2_size': 32, 'layer_3_activation': 'tanh', 'layer_3_extras': {'name': 'batchnorm'}, 'layer_3_size': 40, 'layer_4_activation': 'sigmoid', 'layer_4_extras': {'name': None}, 'layer_4_size': 82, 'layer_5_activation': 'sigmoid', 'layer_5_extras': {'name': None}, 'layer_5_size': 13, 'loss': 'mean_absolute_error', 'n_layers': 4, 'optimizer': 'rmsprop', 'scaler': 'MinMaxScaler', 'shuffle': False},
   {'regressor': 'keras_mlp', 'batch_size': 64, 'init': 'normal', 'layer_1_activation': 'sigmoid', 'layer_1_extras': {'name': 'dropout', 'rate': 0.44707847958746927}, 'layer_1_size': 72, 'layer_2_activation': 'relu', 'layer_2_extras': {'name': 'dropout', 'rate': 0.4997939651537242}, 'layer_2_size': 72, 'layer_3_activation': 'relu', 'layer_3_extras': {'name': None}, 'layer_3_size': 27, 'layer_4_activation': 'sigmoid', 'layer_4_extras': {'name': 'batchnorm'}, 'layer_4_size': 64, 'layer_5_activation': 'tanh', 'layer_5_extras': {'name': 'batchnorm'}, 'layer_5_size': 53, 'loss': 'mean_squared_error', 'n_layers': 4, 'optimizer': 'adamax', 'scaler': 'MaxAbsScaler', 'shuffle': True},
   {'regressor': 'keras_mlp', 'batch_size': 64, 'init': 'glorot_normal', 'layer_1_activation': 'sigmoid', 'layer_1_extras': {'name': 'dropout', 'rate': 0.38712516870482905}, 'layer_1_size': 3, 'layer_2_activation': 'tanh', 'layer_2_extras': {'name': None}, 'layer_2_size': 92, 'layer_3_activation': 'tanh', 'layer_3_extras': {'name': 'batchnorm'}, 'layer_3_size': 4, 'layer_4_activation': 'sigmoid', 'layer_4_extras': {'name': 'dropout', 'rate': 0.26068044657963807}, 'layer_4_size': 75, 'layer_5_activation': 'tanh', 'layer_5_extras': {'name': 'dropout', 'rate': 0.267961152064177}, 'layer_5_size': 76, 'loss': 'mean_squared_error', 'n_layers': 3, 'optimizer': 'adagrad', 'scaler': 'StandardScaler', 'shuffle': False}]

class Scheduler:
    def __init__(self, gpuids):
        self._queue = Queue()
        self._gpuids = gpuids

        self.__init_workers()
        self.max_iter = 81      # maximum iterations per configuration
        self.eta = 3            # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log( x ) / log( self.eta )
        self.s_max = int( self.logeta( self.max_iter ))
        self.B = ( self.s_max + 1 ) * self.max_iter
        #self.available_workers = self.get_available_gpus()
        self.results = []    # list of dicts
        self.best_counter = -1

    def __init_workers(self):
        self._workers = list()
        for gpuid in self._gpuids:
            self._workers.append(Vgg16Worker(gpuid, self._queue))


    def start(self, xfilelst):

        # put all of files into queue
        for xfile in xfilelst:
            self._queue.put(xfile)

        #add a None into queue to indicate the end of task
        self._queue.put(None)

        #start the workers
        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        for worker in self._workers:
            worker.join()
        print("all of workers have been done")



def run(T, gpuids):
    #scan all files under img_path
    #xlist = list([{'regressor': 'keras_mlp', 'batch_size': 128, 'init': 'glorot_normal', 'layer_1_activation': 'sigmoid', 'layer_1_extras': {'name': 'dropout', 'rate': 0.4514185384729481}, 'layer_1_size': 73, 'layer_2_activation': 'tanh', 'layer_2_extras': {'name': None}, 'layer_2_size': 17, 'layer_3_activation': 'relu', 'layer_3_extras': {'name': None}, 'layer_3_size': 42, 'layer_4_activation': 'tanh', 'layer_4_extras': {'name': 'batchnorm'}, 'layer_4_size': 40, 'layer_5_activation': 'tanh', 'layer_5_extras': {'name': None}, 'layer_5_size': 16, 'loss': 'mean_squared_error', 'n_layers': 3, 'optimizer': 'rmsprop', 'scaler': 'MaxAbsScaler', 'shuffle': False}])
    #for xfile in os.listdir(img_path):
    #    xlist.append(os.path.join(img_path, xfile))

    #init scheduler
    x = Scheduler(gpuids)

    #start processing and wait for complete
    x.start(T)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", help="path to your images to be proceed")
    parser.add_argument("--gpuids",  type=str, help="gpu ids to run" )

    args = parser.parse_args()

    gpuids = [int(x) for x in args.gpuids.strip().split(',')]

    print(args.imgpath)
    print(gpuids)

    run(T, gpuids)

