import timeit
import numpy as np 
import multiprocessing as mp
import matplotlib.pyplot as plt
import platform
from time import time

def parzen_estimation(x_samples, point_x, h):
    """
    Implementation of a hypercube kernel for Parzen-window estimation.

    Keyword arguments:
        x_sample: training sample, 'd x 1' -dimensional numpy  array
        x: point x for density estimation, 'd x 1' -dimensional numpy array
        h: window width

    Returns the predicted pdf (probability density function) as float. 

    """
    k_n = 0
    for row in x_samples:
        x_i = (point_x - row[:,np.newaxis]) / (h)
        for row in x_i:
            if np.abs(row) > (1/2):
                break
        else:  # "completion-else" 
                k_n += 1
    
    return (h, (k_n /len(x_samples))/(h**point_x.shape[1]))

def serial(samples, x, widths):
    return [parzen_estimation(samples,x,w) for w in widths ]

def multiprocess(processNo, samples, x, widths):
    pool = mp.Pool(processes = processNo)
    results = [pool.apply_async(parzen_estimation,args = (samples,x,w)) for w in widths]
    results = [p.get() for p in results]
    results.sort()   # to sort the results by input window width
    return results

def print_sysinfo():
    print('\nPython version :', platform.python_version())
    print('compiler :         ', platform.python_compiler())
    print('\nsystem :        ', platform.system())
    print('release :          ', platform.release())
    print('machine :          ', platform.machine())
    print('processor :        ', platform.processor())
    print('CPU count :        ', mp.cpu_count())
    print('interpreter :      ', platform.architecture()[0])
    print('\n\n')

def plot_results():
    bar_labels =['serial','2','3','4','6']

    plt.figure(figsize=(10,8))

    # plot bars
    y_pos = np.arange(len(benchmarks))
    plt.yticks(y_pos, bar_labels, fontsize = 16)
    bars = plt.barh(y_pos,benchmarks, align='center', alpha = 0.4, color = 'g')

    # annotation and Lables
    for ba, be in zip(bars, benchmarks):
        plt.text(ba.get_width()+2, ba.get_y()+ ba.get_height()/2, '{0:.2%}'.format(benchmarks[0]/be), ha = 'center', va ='bottom', fontsize = 12)

    plt.xlabel('time in seconds for n=%s' %n, fontsize = 14)
    plt.ylabel('number of processes', fontsize = 14)

    plt.title('Serial vs. Multiprocessing via Parzen-window estimation', fontsize = 18)

    plt.ylim([-1, len(benchmarks)+0.5])
    plt.xlim([0, max(benchmarks)*1.1])

    plt.vlines(benchmarks[0],-1, len(benchmarks) + 0.5, linestyle ='dashed')
    plt.grid()

    plt.show()

if __name__ == '__main__':
    # Generate random 2D-patterns
    mu_vec = np.array([0,0])
    cov_mat = np.array([[1,0],[0,1]])
    n = 100000

    x_2Dgauss = np.random.multivariate_normal(mu_vec, cov_mat, n)

    widths = np.linspace(1.0 ,1.2, 100)
    point_x = np.array([[0],[0]])
    results = []

    results = multiprocess(4, x_2Dgauss,point_x,widths)

    benchmarks = []

    t0 = time()

    benchmarks.append(timeit.Timer('serial(x_2Dgauss, point_x, widths)', 'from __main__ import serial, x_2Dgauss, point_x, widths').timeit(number=1))
    t1 = time()
    
    benchmarks.append(timeit.Timer('multiprocess(2, x_2Dgauss, point_x, widths)', 'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))
    t2 = time()

    benchmarks.append(timeit.Timer('multiprocess(3, x_2Dgauss, point_x, widths)', 'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))
    t3 = time()

    benchmarks.append(timeit.Timer('multiprocess(4, x_2Dgauss, point_x, widths)', 'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))
    t4 = time()

    benchmarks.append(timeit.Timer('multiprocess(6, x_2Dgauss, point_x, widths)', 'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))
    t5 = time()

    print ('serial execution time:    ', t1 - t0)
    print ('2 process execution time: ', t2 - t1 )
    print ('3 process execution time: ', t3 - t2 )
    print ('4 process execution time: ', t4 - t3 )
    print ('6 process execution time: ', t5 - t4 )


    plot_results()
    #print_sysinfo()