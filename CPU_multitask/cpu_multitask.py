from concurrent.futures import ThreadPoolExecutor
import time
import random
import pdb

from operator import itemgetter
MAX_WORKERS = 3
MAX_RUNG = 5
knobSize = 3

def get_job(rungPool, congPool):
    for i in reversed(range(MAX_RUNG)): #4,3,2,1,0
        if len(rungPool[i]) < knobSize:
            continue
        #else: #promote the rung to higher rank
        elif (len(rungPool[i]) >= knobSize) and (i != MAX_RUNG -1): #promote the rung to higher rank
            to_do = rungPool[i].pop(0)
            to_do[0] +=1
            #rungPool[i][0][0] += 1
            #pring
            return to_do, rungPool, congPool
    return congPool.pop() , rungPool, congPool
        
def update (rungPool, result):
    #print('in update, the input is {}'.format(result))
    rungNumber = result[0] #get the rung number
    rungPool[rungNumber].append(result) #Add trained result to the corresponding rung
    print('current rungPool is {}'.format(rungPool))
    
    if len(rungPool[rungNumber])>1:
        #rungPool[rungNumber] = rungPool[rungNumber].sort(key=itemgetter(1))
        rungPool[rungNumber].sort(key=itemgetter(1))
        #rungPool[rungNumber].reverse()      #reverse the rung so that smallest loss is at right most.
        print('performed sorting on rungPool {}'.format(rungPool))
    return rungPool
            
def return_future_result(message):
    start_time = time.time()
    print('sleep for time {}'.format(message[2]))
    time.sleep(message[2])
    endtime = time.time()
    durtime = int(round(endtime-start_time))
    if message [1] != 0:
        message[1] = random.uniform(0, message[1])
    else:
        message[1] = random.uniform(0, 1)
    message.append([start_time, endtime, durtime])
    print ('message = {}'.format(message))
    return message
    
pool = ThreadPoolExecutor(max_workers = MAX_WORKERS)
#create of cong_pool
cong_pool =[]
for i in range(23):
    cong_pool.append([0,0,random.randint(1,100),{}])
print ('cong_pool is created: {}'.format(cong_pool))

rung_pool = []
for i in range(MAX_RUNG):
    rung_pool.append([])
print ('rung_pool is created: {}'.format(rung_pool))
    
#workerDict = {}
#for i in range(MAX_WORKERS):
#    target = 'future' + str(i)
#    workerDict[target] = pool.submit(return_future_result, (cong_pool.pop()))
future1 = pool.submit(return_future_result, (cong_pool.pop()))
future2 = pool.submit(return_future_result, (cong_pool.pop()))
future3 = pool.submit(return_future_result, (cong_pool.pop()))

while cong_pool:
      
    if future1.done():
        #print('the future1.result() is {} '.format(future1.result()))
        rung_pool = update(rung_pool, future1.result())
        #pdb.set_trace()
        #print('rung_pool is {}'.format(rung_pool))
        message1, rung_pool, cong_pool = get_job(rung_pool, cong_pool)
        print ('message1 is {}, rung_pool is {}, cong_pool is {}'.format(message1, rung_pool, cong_pool))
        #pdb.set_trace()
        future1 = pool.submit(return_future_result, (message1))
        #pdb.set_trace()
    elif future2.done():
        #print('the future2.result() is {} '.format(future2.result()))
        rung_pool = update(rung_pool, future2.result())
        #print('rung_pool is {}'.format(rung_pool))
        message2, rung_pool, cong_pool = get_job(rung_pool, cong_pool)
        print ('message2 is {}, rung_pool is {}, cong_pool is {}'.format(message2, rung_pool, cong_pool))
        future2 = pool.submit(return_future_result, (message2))
    elif future3.done():
        #print('the future3.result() is {} '.format(future3.result()))
        rung_pool = update(rung_pool, future3.result())
        #print('rung_pool is {}'.format(rung_pool))
        message3, rung_pool, cong_pool = get_job(rung_pool, cong_pool)
        print ('message3 is {}, rung_pool is {}, cong_pool is {}'.format(message3, rung_pool, cong_pool))
        future3 = pool.submit(return_future_result, (message3))
    time.sleep(0.5)
    #print("checking")
print('last future1 is {}'.format(future1.result()))
print('last future2 is {}'.format(future2.result()))
print('last future3 is {}'.format(future3.result()))
print('the final rung_pool is  {}'.format(rung_pool))
print('the final cong_pool is  {}'.format(cong_pool))

