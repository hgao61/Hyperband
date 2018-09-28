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

def Vgg16(params=None):
    
    #print ("iterations:", n_iterations)
    print_params( params )
    
    y_train = data['y_train']
    y_test = data['y_test']
        
    if params['scaler']:
        scaler = eval( "{}()".format( params['scaler'] ))
        x_train_ = scaler.fit_transform( data['x_train'].astype( float ))
        x_test_ = scaler.transform( data['x_test'].astype( float ))
    else:
        x_train_ = data['x_train']
        x_test_ = data['x_test']
        
    input_dim = x_train_.shape[1]

    model = Sequential()
    model.add( Dense( params['layer_1_size'], init = params['init'], 
        activation = params['layer_1_activation'], input_dim = input_dim ))
    
    for i in range( int( params['n_layers'] ) - 1 ):
        
        extras = 'layer_{}_extras'.format( i + 1 )
        
        if params[extras]['name'] == 'dropout':
            model.add( Dropout( params[extras]['rate'] ))
        elif params[extras]['name'] == 'batchnorm':
            model.add( BatchNorm())
            
        model.add( Dense( params['layer_{}_size'.format( i + 2 )], init = params['init'], 
            activation = params['layer_{}_activation'.format( i + 2 )]))
           
    model.add( Dense( 1, init = params['init'], activation = 'linear' ))

    model.compile( optimizer = params['optimizer'], loss = params['loss'] )
    
    return model
