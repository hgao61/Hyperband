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
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

STARTING_L2_REG = 0.0007
OPTIMIZER_STR_TO_CLASS = {
    'Adam': Adam,
    'Nadam': Nadam,
    'RMSprop': RMSprop
}
NB_CHANNELS = 3
IMAGE_BORDER_LENGTH = 32
# NB_CLASSES = 10
NB_CLASSES_FINE = 10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0 - 0.5
x_test = x_test.astype('float32') / 255.0 - 0.5
y_train = to_categorical(y_train, NB_CLASSES_FINE)
y_test = to_categorical(y_test, NB_CLASSES_FINE)

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
    '''
    if params['choice']['layers'] == 'three':
        #model.add(Dense(output_dim=params['choice']['units3'], init = "glorot_normal")) 
        #model.add(PReLU())
        #model.add(Dropout(params['choice']['dropout3']))  
        
        model.add(Conv2D(params['choice']['units3'], kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(params['choice']['units3'], kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(params['choice']['dropout3']))
    '''    
    model.add(Flatten())
    model.add(Dense(params['units3'], activation='relu'))
    model.add(Dropout(params['dropout3']))
    model.add(Dense(10, activation='softmax'))
    '''
    model = Sequential()

    model.add(Dense(output_dim=params['units1'], input_shape=(32,32,3),init = 'glorot_normal')) 
    model.add(PReLU())
    model.add(Dropout(params['dropout1']))

    model.add(Dense(output_dim=params['units2'], init = "glorot_normal")) 
    model.add(PReLU())
    model.add(Dropout(params['dropout2']))

    if params['choice']['layers'] == 'three':
        model.add(Dense(output_dim=params['choice']['units3'], init = "glorot_normal")) 
        model.add(PReLU())
        model.add(Dropout(params['choice']['dropout3']))    

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('linear'))
    '''
    model.compile(loss = 'mae', optimizer = 'adam', metrics = ["mae"])
    model.summary()
    return model
    '''
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
    '''
    #return model
