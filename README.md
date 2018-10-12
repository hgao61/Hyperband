## Implementation of the parallelized Hyperband algorithm

This repository is the implementation for hyperparameters with parallelized Hyperband, based on the paper:
[MASSIVELY PARALLEL HYPERPARAMETER TUNING](https://openreview.net/forum?id=S1Y7OOlRZ), published by Li et al, in 2018

### Description
Although machine learning models have recently achieved dramatic successes in a variety of practical applications, these models are highly sensitive to internal parameters, e.g. *hyperparameters*.

The traditional methods such like random search, grid search have the limitation of no early-stopping and no adaptively allocation of computation resources. 

The paper *Massively parallel hyperparameter tuning* showed a new way of parallelized hyperparameter tuning with the use of the algorithm *Hyperband* presented by [Li et al. 2016] in the paper [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560) and brought up the new algorithm called *Asynchronous Successive Halving Algorithm(ASH)*. 

### Project structure
To implement this ASH algorithm, I break down the task into 3 mini projects representing 3 steps to achieve the goal. And here are the details of implementing those mini projects:

#### CPU ASH algorithm
Firstly, the ASH algorithm is implemented with CPU workers(source code is at CPU_ASH subfolder). The program demonstrated the use of 3 CPU workers and buit up the framwork on which final implementation of the ASH algorithm is based. The focus on this CPU implementation is to explore the multi-processing scheme in Python, the concurrent task assignment and dispatching of CPU workers; the define of the framework's data structure.

This is the flow of the implementation:

1. A pool of configurations, the Hyperparameter combinations, for the targeted neural network is generated.

2. An empty rung pool is created as a list. Each of its element will be a list of configurations with the same rung ranking. For example, if the maximum rung number is 5, the rung pool will have 5 empty lists in it. And the first empty list will be for configurations with rank #0, the 2nd empty list will be for configurations with rank #1, and so on.

3. The first 3 configurations will be picked from the config pool, ranked as ranking 0, and assigned to the 3 available CPU workers as a start point. Some additional information such as worker number, the ranking number, will be added as well.

4. The assigned CPU worker is given a random generated number between 1 and 100 and will sleep for this period of time when called. Then it will return a random generated floating point number within range of (0,1). This is to simulate the time duration of each training is different and the different loss value you got from train to train. 

5. Upon the finish of the first task, the freed CPU worker will return the finished task to the rung pool at the specified ranking. 

6. Then it will look through the rung pool from highest rank to lowest rank to look for the promotable configuration. A configuration is considered as promotable if it's with the lowest loss configuration in its particular ranking, and there are more than eta number of configurations in that ranking.

7. If there is a promotable configuration available, it's rank number will be increase by 1, and given to the CPU worker for processing. And the steps from 4 to 7 will be repeated again.

8. The above process repeats until all the elements in the cong pool are gone. 


The command to run this program is:
*python cpu_multitask.py*

#### Hyperband with parallel GPU workers
Secondly, the implementation of the Hyperband algorithm running with multiple GPU workers in parallel. The source code for the basic Hyperband algorithm implementation is forked from here:
https://github.com/zygmuntz/hyperband

Based on that, I implemented the multiprocessing of utilization of multiple GPU workers in parallel. 
The parallellization utilizes multiple GPUs, each GPU worker works individually on a given random generated configuration of a neural network. 

All the generated configurations waiting to be trained are put in a pool for multi-process processing by the GPU worker. 

Currently there are two type of neural networks are supported:
The multilayer perceptron network with the [KIN8NM](https://www.openml.org/d/189) dataset.
The VGG network with the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
The command to run this program is:
*python main.py --gpuid '0,1,2' specifies the number of GPUs*
               *--network 'MLP' or 'CNN' specifies the neural network to use*
               *--mode 'hyperband' or 'random_search' by default it's 'hyperband', 'random_search' is only for getting the benchmark time for comparison.

#### Hyperband with asynchronous successive halving algorithm

Finally, by combining the CPU-ash algorithm and the parallelization of GPU workers together, we have the algorithm running. 

The command to run this program 
*python main.py --gpuid '0,1,2'*
### Requirements
keras==2.2.2

tensorflow-gpu==1.9.0

hyperopt==0.2

