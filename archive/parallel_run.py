"function (and parameter space) definitions for hyperband"
"regression with Keras (multilayer perceptron)"

from common_defs import *

# a dict with x_train, y_train, x_test, y_test
from load_data_for_regression import data

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization as BatchNorm
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import *

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler


    

def get_job():
    while (rung_pool[0]):
        
        for i in reversed( range( max_rung  )):
            #check if theta(k) exists on the furthest trained promotable cong theta
            #1. checking from highest rung group to lowest group
            #2. it needs to be on the top 1/eta fraction of loss, e.g. if eta=3, then it's top 1/3
        
            if len(rung_pool[i] >= eta):
                #pop the config with min loss to temp_save: selected
                selected = rung_pool[i].pop(0)
                #promote the rung by 1
                selected[0] +=1
                return selected

        return rung_pool[0].pop(0)
          
            
def parallel_run( n_iter, params, loops, max_rung, eta, available_workers):
    rung_pool=[[] for i in range(max_rung)]
    #Create rung_pool with T as Rung[0]
    #Each element in Rung[0] is in the format of [rung#,loss,just_promoted(bool), {configuration}]
    for i in range(len(T)):
        rung_pool[0].append([0,0,0,params[i]])
    
    while available_worker:
        newconfig = get_job()
        config_to_train = newconfig[3]
        n_iter = newconfig[0]
        #create model from the config_to_train
        result = try_params( n_iter, config_to_train, available_worker.pop(0)[-1] )
        
        #



