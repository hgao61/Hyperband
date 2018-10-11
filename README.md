## Implementation of the parallelized Hyperband algorithm

This repository is the implementation for hyperparameters with parallelized Hyperband, based on the paper:
MASSIVELY PARALLEL HYPERPARAMETER TUNING *https://openreview.net/forum?id=S1Y7OOlRZ*

### Description
Although machine learning models have recently achieved dramatic successes in a variety of practical applications, these models are highly sensitive to internal parameters, e.g. *hyperparameters*.

The traditional methods such like random search have the limitation of no early-stopping and no damatically allocation of computation resources. 

The paper *Massively parallel hyperparameter tuning* presented a new way of parallelized hyperparameter tuning
with the use of the algorithm *Hyperband* by [Li et al. 2016] and brought up the new algorithm called *Asynchronous Successive Halving Algorithm(ASH)*. 

### Project structure
This github repository is about the implementation of this ASH algorithm. There are three aspects of implementing this algorithm.  

Firstly, The ASH algorithm is implemented in with CPU workers. The program mimics the situation of 3 CPU workers and the implementation of the ASH algorithm based on that. 



Secondly, The folder 


### Requirements
keras==2.2.2
tensorflow-gpu==1.9.0
hyperopt==

