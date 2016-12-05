#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import math
from subprocess import call
from scipy.stats import rankdata
#import analytical_model as model
#import trimodal
#import traceGen 
#import cache 

np.set_printoptions(precision=2)

def value_iteration(values,p,d,h,e,s,drag,allow_plot=False):
    
    threshold = 1e-4

    v = np.zeros((2,n))
    v[1,:len(values)] = values

    delta = np.zeros(n)

    h = np.where(h < 1e-5, np.zeros_like(h), h)
    boundry = np.argmax(np.cumsum(h))

    cumevents = np.cumsum((h+e)[::-1])[::-1]
    cumevents = np.where(cumevents < 1e-5, np.ones_like(cumevents), cumevents)

    if allow_plot:
        plt.close('all')
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(values)
        plt.title('values')
        plt.subplot(3,1,2)
        plt.plot(h)
        plt.title('hit')
        plt.subplot(3,1,3)
        plt.plot(e)
        plt.title('eviction')
        plt.show()

        plt.close('all')
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1,1,1)
        # simple = model.analysis_modal(p,d,h,e,False)
        # ax.plot(simple)
        line, = ax.plot(v[0])
        plt.ion()

    for i in range(20000):
        this = v[i%2]
        that = v[(i+1)%2]

        for j in range(0, n):
            this[j] = 0

            # hit probability
            this[j] += h[j] / cumevents[j] * (1 + drag * that[0])

            # eviction probability
            this[j] += e[j] / cumevents[j] * (0 + drag * that[0])

            # otherwise
            if j+1 < n:
                leftover = (cumevents[j] - h[j] - e[j]) / cumevents[j]
                if leftover > 1e-5:
                    this[j] += leftover * (0 + drag * that[j+1])

            if j>=0 and j <= boundry:
                delta[j] = abs(this[j]-this[0]-(that[j]-that[0])) 

        if max(delta) < threshold: 
            if allow_plot:
                plt.ioff()
            break

        if i%50 != 0 or not allow_plot: continue
        ax.set_title('After iteration %d' % i)
        yplot = this - this[0]
        line.set_ydata(yplot)
        ax.set_ylim(np.min(yplot[0:boundry]), np.max(yplot[0:boundry+1]))
        ax.set_xlim(0,boundry+1)
        ax.relim()
        ax.autoscale_view()
        plt.draw() # tell pyplot data has changed
        plt.pause(0.000001) # it won't actually redraw until you call pause!!!

    return this - this[0]

def policy_iteration(p,d,s,drag=1,allow_plot=False):
    ed = round(np.dot(p,d))
    # step = - int(ed/8.)
    # assert abs(step) >= 1
    # cache_size = range(int(ed),s,step) + [s,]
    cache_size = [s] * 3
    # values = np.zeros(max(d)+10)
    values = np.arange(n)
    ranks = np.zeros_like(values)
    for s in cache_size:
        # need to detect whether or not it's converged

        h,e = parse_policy(values,p,d,s)

        h = fill_rdd(p,d,h)

        values = value_iteration(values,p,d,h,e,s,drag,allow_plot)

    return values

def fill_rdd(p,d,h):
    w = 0.05
    new_h = (1-w) * h + w * rdd

    return new_h

def parse_policy(values,p,d,s):
    critical_values = np.arange(len(values),dtype=float)
    critical_values[max(d)+1:] = 0
    for i in range(len(d)):
        critical_values[d[i]+1:] -= abs(critical_values[d[i]+1] - values[d[i]+1])

    ranks = values_to_ranks(critical_values)
    log_array(ranks,'ranks')
    plt.close("all")
    plt.subplot(2,1,1)
    plt.plot(critical_values)
    plt.title('critical values')
    plt.subplot(2,1,2)
    plt.plot(ranks)
    plt.title('ranks')
    plt.show()
    print "log the ranks"

    call('./compute '+str(s),shell=True)
    h = read_distribution('hit.out')
    e = read_distribution('evict.out')

    return h, e

def values_to_ranks(values):
    ranks = rankdata(-values,'min')
    return ranks
    
def log_values(values,filename):
    f = open(filename+'.out','w+')
    f.write(str(len(values))+'\n')
    for v in values:
        f.write("%.2f\n" %v )
    f.close()

def log_array(array,filename):
    f = open(filename+'.out', 'w+')
    f.write(str(len(array))+'\n')
    for a in array:
        f.write("%d\n" %a)
    f.close()

def read_distribution(filename):
    h = np.zeros(n,dtype=float)
    f = open(filename)
    for i,line in enumerate(f):
        if i<n:
            h[i] = line
        else:
            break
    return h

def load_rdd(filename):
    n_line = 50
    p = np.zeros(n_line,dtype=float)
    d = np.zeros(n_line,dtype=int)

    rdd = np.zeros(n,dtype=float)
    f = open(filename)
    for i,line in enumerate(f):
        d[i],p[i] = line.split(' ')   
        rdd[d[i]] =  p[i]
    f.close()
    return p,d,rdd

n = 1200
if __name__ == '__main__':

    p,d,rdd = load_rdd("rdd.out")
    ed = round(np.dot(p,d))
    print "expected reuse distance = %d" %ed
    plt.plot(rdd)
    plt.show()

    s = 10
    print "size = %d" %s

    drag = 0.9999
    values = policy_iteration(p,d,s,drag,True)
    log_values(values,'values-'+str(s))
    plt.plot(values)
    plt.show()
