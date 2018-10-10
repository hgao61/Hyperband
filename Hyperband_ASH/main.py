from multiprocessing import Process, Queue
import os
import argparse
from vgg16_worker import Vgg16Worker
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
        
def get_params():

    params = sample( space )
    return handle_integers( params )



class Scheduler:
    def __init__(self, gpuids):
        self._queue = Queue()
        self._gpuids = gpuids
        self.__init_workers()

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
        #print(worker.get())
        


def run(T, gpuids):
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
    #definition of parameters
    
    max_iter = 256     # maximum iterations per configuration
    eta = 4            # defines configuration downsampling rate (default = 3)

    logeta = lambda x: log( x ) / log( eta )
    s_max = int( logeta( max_iter ))
    B = ( s_max + 1 ) * max_iter
    #self.available_workers = self.get_available_gpus()
    results = []    # list of dicts
    counter = 0
    best_loss = np.inf
    best_counter = -1
    dry_run = False
    skip_last =0
    print(args.imgpath)
    print(gpuids)
    # can be called multiple times
    
    start_time = time()
    for s in reversed( range( s_max + 1 )):
        #for s in range(1):#only test for one bracket    
        # initial number of configurations
        n = int( ceil( B / max_iter / ( s + 1 ) * eta ** s ))    
            
        # initial number of iterations per config
        r = int(max_iter * eta ** ( -s ))
          
        # n random configurations
        T = [ get_params() for i in range( n )] 
        print ("s={}".format(s))
        print ("T is of size {}".format(len(T)))
        print ("T={}".format(T))
        
        for i in range(( s + 1 ) - int( skip_last )):    # changed from s + 1
            # Run each of the n configs for <iterations> 
            # and keep best (n_configs / eta) configurations
   
            n_configs = int(floor(n * eta ** ( -i )))
            n_iterations = int(r * eta ** ( i ))
            new_T = []
            ##prepare the queue new_T for the worker from the generated T
            for i in range(len(T)):
                new_T.append([i,int(n_iterations),0,100, [], T[i]])    #(0)ith_config,(1)_num_epoch,(2)loss,(3)#th_gpu,(4)[start_time, end_time, duration],(5)config
            print ("newly formed T structure is:{} ".format(new_T))
            print ("\n*** {} configurations x {:.1f} iterations each".format( 
                n_configs, n_iterations ))
                
            val_losses = []
            early_stops = []
            index = []
            #for t in T:
            #process the queue with the workers    
            counter += 1
            print ("\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format( 
                counter, ctime(), best_loss, best_counter ))
            
            #batch_run_start_time = time()
            #The start running of the GPU batch processing
            run(new_T, gpuids)
                
            #print ("result is equal to {}".format(result))
            print ("calculation finished")
         
            #The end of the GPU batch processing
            #seconds = int( round( time() - batch_run_start_time ))#time the running time for the batch of T configs
            #print ("\n{} seconds. take to run the batch of {} configs with {}epoch".format( seconds, len(new_T), n_iterations))
            
                
            #read result from txt file to array, delete the txt file
            #put the results into raw_result
            with open(fn) as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    counter = int(row[0])
                    n_iterations = row[1]
                    loss = row[2]
                    GPU = row[3]
                    runtime = row[4][1:-1].split(',')[-1]
                    config = row[5]
                    print ("#{} epoch={} loss={} #th_GPU={} time={} configuration={}".format(counter, n_iterations, loss, GPU, time, config))
                    result = literal_eval(loss)  #convert the loss from string to dict
                    assert( type( result ) == dict )
                    assert( 'loss' in result )
                    loss = result['loss']   
                    val_losses.append( loss )
                    
                    early_stop = result.get( 'early_stop', False )
                    early_stops.append( early_stop )
                    
                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if loss < best_loss:
                        best_loss = loss
                        best_counter = counter
                
                    result['counter'] = counter
                    result['seconds'] = runtime[-1]
                    result['params'] = config
                    result['iterations'] = n_iterations
                    index.append(int(counter))
                    results.append( result )
            os.remove(fn)
            print ('get a list [results] of length {}'.format(len(results)))
            print ('get a list [loss] of length {}'.format(len(val_losses)))
            print ('get a list [val_loss] of length {}'.format(len(early_stops)))
            # select a number of best configurations for the next loop
            # filter out early stops, if any
            L=sorted(zip(val_losses, index), key = operator.itemgetter(0))
            new_val_losses, indices= zip(*L)
            #indices = np.argsort( val_losses )
            print ('length of indices is {}'.format(indices))
            print ('length of indices is {}'.format(len(indices)))
            print ('length of T is {}'.format(len(T)))
            T = [ T[j] for j in indices if not early_stops[j]]
            T = T[ 0:int( n_configs / eta )]
                
print ("{} total, best:\n".format( len( results )))

for r in sorted( results, key = lambda x: x['loss'] )[:5]:
    print ("loss: {:.2%}  | #{}th iterations | run {} ".format( 
        r['loss'], r['iterations'], r['counter'] ))
    pprint( r['params'] )
    print()

print ("saving...")
seconds = int( round( time() - start_time ))
print ("\n{} seconds.".format( seconds ))
    
with open( 'outputlog', 'wb' ) as f:
    pickle.dump( results, f )
	
    #with open('results_save.csv', mode='w') as colors_file:
    #    csv_writer = csv.writer(colors_file)
    #    for item in results:
    #        csv_writer.write(item, delimter =";")
with open('logwrite.txt','w') as logwriter:
    logwriter.write('\n'.join([str(i) for i in results]) +'\n')
