import tensorflow as tf
import multiprocessing
import datetime
import time
import random

def hello(taskq, resultq):
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    while True:
        name = taskq.get()
        start_time = time.time()
        res = sess.run(tf.constant('hello ' + name))
        time.sleep(random.randint(1,101))
        endtime = time.time()
        durtime = int(round(endtime-start_time))
        
        resultq.put([start_time, endtime, durtime ])

if __name__ == '__main__':
    cong_pool = []
    for i in range(10):
        cong_pool.append([0,random.randint(1,1001),{}])
    print (cong_pool)
        
    taskq = multiprocessing.Queue()
    resultq = multiprocessing.Queue()
    p = multiprocessing.Process(target=hello, args=(taskq, resultq))
    p.start()
    
    

    taskq.put('world')
    time.sleep(10)
    
    taskq.put('abcdabcd987')
    for i in range(10):
        taskq.put('abcdabcd'+ str(i))
    taskq.close()
    for i in range(12):
        print(resultq.get())
    #print(resultq.get())

    p.terminate()
    p.join()