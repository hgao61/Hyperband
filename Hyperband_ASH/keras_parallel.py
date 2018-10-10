from concurrent.futures import ThreadPoolExecutor

import random
import pdb
from multiprocessing import Process, Queue
import os
import argparse
from vgg16_worker import vgg16_worker
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
#import _pickle as cPickle
import pickle
from operator import itemgetter
#from parallel_run import parallel_run
import operator
import tensorflow as tf
from tensorflow.python.client import device_lib
try:
    from hyperopt import hp
    from hyperopt.pyll.stochastic import sample
except ImportError:
    print ("In order to achieve operational capability, this programme requires hyperopt to be installed (pip install hyperopt), unless you make get_params() use something else.")
    
#space of hyper-settings for the algorithm    
MAX_WORKERS = 3
#MAX_RUNG = 5
knobSize = 3
    
max_layers = 13
max_layer_size = 100
fn ='vgglog.csv'
n_iterations = 1    
 


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
        
def get_params(): #generate the configurations for the hyperparameter space
    params = sample( space )
    return handle_integers( params )


def get_job(rungPool, congPool, MAX_RUNG):
    for i in reversed(range(MAX_RUNG)): #4,3,2,1,0
        if len(rungPool[i]) < knobSize:
            continue
        #else: #promote the rung to higher rank
        elif (len(rungPool[i]) >= knobSize) and (i != MAX_RUNG -1): #promote the rung to higher rank
            print('rungPool[i][0] is {}'.format(rungPool[i][0]))
            to_do = rungPool[i].pop(0)
            to_do[4] +=1
            #rungPool[i][0][0] += 1
            #pring
            return to_do, rungPool, congPool
    return congPool.pop() , rungPool, congPool
        
def update (rungPool, result):
    #print('in update, the input is {}'.format(result))
    rungNumber = result[4] #get the rung number
    rungPool[rungNumber].append(result) #Add trained result to the corresponding rung
    print('current rungPool is {}'.format(rungPool))
    
    if len(rungPool[rungNumber])>1:

        rungPool[rungNumber].sort(key=itemgetter(2))
        #rungPool[rungNumber].reverse()      #reverse the rung so that smallest loss is at right most.
        print('performed sorting on rungPool {}'.format(rungPool))
    return rungPool

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
    
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
            
def return_future_result(message):
    print ('the return_future_result input is ()'.format(message))
    #pdb.set_trace()
    label = vgg16_worker(message)
    print ('returned label is {}'.format(label))
    return label
    
def best_results(rung):
    best_loss = 100
    best_counter = -100
    num_iterations = -10
    for i in range(len(rung)):
        for j in range(len(rung[i])):
            loss = rung[i][j][2]
            if loss < best_loss:
                best_loss = loss
                best_counter = rung[i][j][0]
                num_iterations = rung[i][j][4]
                
                #result['counter'] = counter
                #result['seconds'] = runtime[-1]
                #result['params'] = config
                #result['iterations'] = n_iterations
                #index.append(int(counter))
                #results.append( result )
    best_results = [best_loss, best_counter, num_iterations]
    return best_results

def parallel_comp(cong_pool, rung_pool, gpus, maxRung):
    pool = ThreadPoolExecutor(max_workers = 3)
    input1 = cong_pool.pop()
    input2 = cong_pool.pop()
    input3 = cong_pool.pop()
    input1[3] = '/device:GPU:0'
    input2[3] = '/device:GPU:1'
    input3[3] = '/device:GPU:2'
    future1 = pool.submit(return_future_result, (input1))
    future2 = pool.submit(return_future_result, (input2))
    future3 = pool.submit(return_future_result, (input3))

    while cong_pool:
      
        if future1.done():
            #print('the future1.result() is {} '.format(future1.result()))
            rung_pool = update(rung_pool, future1.result())
    
            #print('rung_pool is {}'.format(rung_pool))
            message1, rung_pool, cong_pool = get_job(rung_pool, cong_pool,maxRung)
            print ('message1 is {},\nrung_pool1 is {},\ncong_pool1 is {}'.format(message1, rung_pool, len(cong_pool)))
            message1[3] = '/device:GPU:0'
            future1 = pool.submit(return_future_result, (message1))
       
        elif future2.done():
            #print('the future2.result() is {} '.format(future2.result()))
            rung_pool = update(rung_pool, future2.result())
            #print('rung_pool is {}'.format(rung_pool))
            message2, rung_pool, cong_pool = get_job(rung_pool, cong_pool ,maxRung)
            print ('message2 is {},\nrung_pool is {},\ncong_pool is {}'.format(message2, rung_pool, len(cong_pool)))
            message2[3] = '/device:GPU:1'
            future2 = pool.submit(return_future_result, (message2))
        elif future3.done():
            #print('the future3.result() is {} '.format(future3.result()))
            rung_pool = update(rung_pool, future3.result())
            #print('rung_pool is {}'.format(rung_pool))
            message3, rung_pool, cong_pool = get_job(rung_pool, cong_pool,maxRung)
            print ('message3 is {},\nrung_pool is {},\ncong_pool is {}'.format(message3, rung_pool, len(cong_pool)))
            message3[3] = '/device:GPU:2'
            future3 = pool.submit(return_future_result, (message3))
        sleep(0.001)
        #print("checking")
    return rung_pool


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("--imgpath", help="path to your images to be proceed")
    parser.add_argument("--gpuids",  type=str, help="gpu ids to run" )

    args = parser.parse_args()

    #definition of parameters
    
    max_iter = 81     # maximum iterations per configuration
    eta = 3           # defines configuration downsampling rate (default = 3)
    gpuids = get_available_gpus()    
    print('available gpus are {}'.format(gpuids))

    logeta = lambda x: log( x ) / log( eta )
    s_max = int( logeta( max_iter ))
    B = ( s_max + 1 ) * max_iter
    
    #self.available_workers = self.get_available_gpus()
    results = []    # list of dicts
    final_results = []
    counter = 0
    best_loss = np.inf
    best_counter = -1
    dry_run = False
    skip_last =0
    print(gpuids)
    # can be called multiple times
    
    start_time = time()
    for s in reversed( range( s_max + 1 )):
    
        #for s in range(1):#only test for one bracket    
        # initial number of configurations
        n = int( ceil( B / max_iter / ( s + 1 ) * eta ** s ))    
            
        # initial number of iterations per config
        r = int(max_iter * eta ** ( -s ))
        #maxRung =  int( log(n, eta) - (s_max - s) + 1)
        maxRung =  5
        if maxRung ==0:
            maxRung += 1
        print ("maxRung is {}".format(maxRung))
        # Run each of the n configs for <iterations> 
        # and keep best (n_configs / eta) configurations
        #n_configs = int(floor(n * eta ** ( -i )))
        #n_iterations = int(r * eta ** ( i ))
                    # n random configurations
        n_configs = n
        n_iterations = r
        T = [ get_params() for i in range( n + len(gpuids) )] 
        print ("bracket s = {}".format(s))
        print ("T is of size {}".format(len(T)))
        print ("T = {}".format(T))
        cong_pool =[]

        for i in range(n):
            cong_pool.append([i+counter,int(n_iterations),0,100,0,[], T[i]])    
            #(0)ith_config,(1)num_epoch,(2)loss,(3)#th_gpu,
             #(4)rung number (5)[start_time, end_time, duration],(6)config
        counter += n_configs
        print ('cong_pool is created: {}'.format(cong_pool))
        
        rung_pool = []
        for i in range(maxRung):#MAX_RUNG):
            rung_pool.append([])
        print ('rung_pool is created :{}'.format(rung_pool))

        print ("\n*** {} configurations x {:.1f} iterations each".format( 
            n_configs, n_iterations ))
            
        val_losses = []
        early_stops = []
        index = []
        
        print ("\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format( 
            counter, ctime(), best_loss, best_counter ))
         
        rungPool = parallel_comp(cong_pool, rung_pool, gpuids,maxRung)    
        #print ("result is equal to {}".format(result))
        print ("calculation finished")
        print("the final rung pool for this round is {}".format(rungPool)) 
        #print ("\n{} seconds. take to run the batch of {} configs with {}epoch".format( seconds, len(new_T), n_iterations))        best_results = [best_loss, best_counter, num_iterations]

        bestResults = best_results(rungPool)  
        print ("at bracket {}, with n_configs = {} and n_iterations = {}, best loss is: {}, at config#: {} at rung#: {}".format(s, n_configs, n_iterations, bestResults[0], bestResults[1], bestResults[2]))  
        results.append([s, n_configs, n_iterations])    
        results.append(bestResults)
        
        #print ("{} total, best:\n".format( len( results )))

        #for r in sorted( results, key = lambda x: x['loss'] )[:5]:
        #    print ("loss: {:.2%}  | #{}th iterations | run {} ".format( 
        #        r['loss'], r['iterations'], r['counter'] ))
        #pprint( r['params'] )
        #print()
        final_results.append(results)
    print ("saving...")
    seconds = int( round( time() - start_time ))
    print ("\n{} seconds.".format( seconds ))
    print('the final rung_pool is  {}\n'.format(final_results))

    
    with open( 'outputlog', 'wb' ) as f:
        pickle.dump( results, f )
	
        #with open('results_save.csv', mode='w') as colors_file:
        #    csv_writer = csv.writer(colors_file)
        #    for item in results:
        #        csv_writer.write(item, delimter =";")
    with open('logwrite.txt','w') as logwriter:
        logwriter.write('\n'.join([str(i) for i in results]) +'\n')
        
