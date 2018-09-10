# -*- coding: utf-8 -*-
"""
Created on Fri Sep  10 10:45:49 2018

@author: Wei Wan

retrieve results in a p0articular order
"""
import multiprocessing as mp
import random
import string

#Define an output queue
output = mp.Queue()

# dfine a example function
def rand_string(length, pos, output):
    """ Generates a random string of numbers, lower - and uppercase chars. """
    rand_str = ''.join(random.choice(
            string.ascii_lowercase + string.ascii_uppercase + string.digits)
        for i in range(length))
    output.put((pos, rand_str))
    #return (output)

if __name__== '__main__':
    # Setup a list of processes that we want to run
    processes = [mp.Process(target=rand_string, args = (5, x, output)) for x in range (4)]

    #Run processes
    for p in processes:
        p.start()
 
    # Exit the completed processes
    for p in processes:
        p.join()    

    # Get process results from the output queue
    results = [output.get() for p in processes]
    # sort the results.
    results.sort()
    results = [r for r in results]
    orderR = [r[1] for r in results]
    print (results, orderR)