# Parallelized Hyperband
## Background
This is the implementation of parallelized Hyperband based on paper of " Massively
 Parallel hyperparameter Tuning" which is under double-blind review at ICLR 2018.

 The new proposed method is  based on the [Li et al] 's paper at 2016 on Hyperband,
 a creative way of hyperparameter tuning algorithm which use early-stopping to balance
 the resource usage and training time on configurations'

## Parallelized Hyperband
Algorithm 1: Asynchronous Successive Halving Algorithm
Input: r, eta(default eta = 3), s
Algorithm async_SHA()
    repeat
        for free worker do
            (theta, k) = get_job()
            worker_performs run_then_return_val_loss(theta, r*eta **(s+k))
        end
        for completed job(theta, k) with loss l do
            Update configuration theta in rung k with loss l.
        end

Procedure get_job()
    //A configuration in a given rung is "promotable" if its validation
    loss places in the top 1/eta fraction of completed configurations in its rung and 
    it has not already been promoted.
