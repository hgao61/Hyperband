** Implementation of the parallelized Hyperband algorithm

Although machine learning models have recently achieved dramatic successes in a variety of practical applications, these models
are highly sensitive to internal parameters, e.g. *hyperparameters*

The traditional methods such like random search have the limitation of no early-stopping
and no damatically allocation of computation resources. 

The paper *Massively parallel hyperparameter runing* presented a new way of parallelized hyperparameter tuning
with the use of the algorithm Hyperband by [Li et al. 2016] and brought up the new algorithm called Asynchronous Successive Halving Algorithm. 

This github repository is about the implementation of this ASH algorithm.  It's divided into three sections. 

Firstly, THE ASH algorithm is implemented in with CPU workers. The program mimics the situation of 3 CPU workers and the implementation of the ASH algorithm based on that. 



