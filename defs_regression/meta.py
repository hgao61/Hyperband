# meta regressor
from common_defs import *

regressors = ('keras_mlp', )
#regressors = ( 'gb', 'rf', 'xt', 'sgd', 'polylearn_fm', 'polylearn_pn', 'keras_mlp' )

# import all the functions
for r in regressors:
	exec( "from defs_regression.{} import get_params as get_params_{}".format( r, r ))
	exec( "from defs_regression.{} import try_params as try_params_{}".format( r, r ))
    exec( "from defs_regression.{} import get_job as get_job_{}".format( r, r ))
    exec( "from defs_regression.{} import parallel_run as parallel_run_{}".format( r, r ))
space = { 'regressor': hp.choice( 'r', regressors ) }

def get_job():
    return 
def parallel_run(n_iterations, T,loops, max_rung, gpu_list):
    return
    
def get_params():
	params = sample( space )
	r = params['regressor']
	r_params = eval( "get_params_{}()".format( r ))
	params.update( r_params )
	return params

def try_params( n_iterations, params, gpu_num ):
	
	params_ = dict( params )
	r = params_.pop( 'regressor' )
	print (r)
	
	return eval( "try_params_{}( n_iterations, params_, gpu_num )".format( r ))
			 
	
