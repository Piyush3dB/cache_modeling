#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import math
from subprocess import call
#import analytical_model as model
#import trimodal
#import traceGen 
#import cache 

np.set_printoptions(precision=2)

def value_iteration(values,p,d,h,e,s,drag,allow_plot=False):
    
    threshold = 1e-4

    [d1,d2,d3] = d[0:3]
    v = np.zeros((2,n))
    v[1,:len(values)] = values

    delta = np.zeros(n)

    h = np.where(h < 1e-5, np.zeros_like(h), h)
    boundry = np.argmax(np.cumsum(h))
    print boundry

    cumevents = np.cumsum((h+e)[::-1])[::-1]
    cumevents = np.where(cumevents < 1e-5, np.ones_like(cumevents), cumevents)

    if allow_plot:
        plt.close('all')
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(h)
        plt.title('hit')
        plt.subplot(3,1,2)
        plt.plot(e)
        plt.title('eviction')
        plt.subplot(3,1,3)
        plt.plot(cumevents)
        plt.title('cumevents')
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

def policy_iteration(p,d,s,drag=1):
    ed = round(np.dot(p,d))
    # step = - int(ed/8.)
    # assert abs(step) >= 1
    # cache_size = range(int(ed),s,step) + [s,]
    cache_size = [s] * 2
    # values = np.zeros(max(d)+10)
    values = np.arange(n)
    ranks = np.zeros_like(values)
    for s in cache_size:

        # [opt,opt_miss_rate] = trimodal.opt_policy(p,d,s)

        # [h,e] = parse_policy(values,p,d,s)[1:3]
        # [h,e] = sim_policy(values,p,d,s)[1:3]

        # call the modeling code
        ranks = values_to_ranks(values)
        log_array(ranks,'ranks')
        print "log the ranks"

        call('./examples')
        h = read_distribution('hit.out')
        e = read_distribution('evict.out')
        
        # h = fill_rdd(p,d,h)

        values = value_iteration(values,p,d,h,e,s,drag,False)
    return values

def fill_rdd(p,d,h):
    w = 0.05
    new_h = (1-w) * h + w * rdd

    return new_h

def sim_policy(values,p,d,s):

    idealRdDist = [(p[i],d[i]) for i in range(len(p))]
    trace = traceGen.TraceDistribution(idealRdDist)
    trace.generate(10000)

    plt.close('all')
    plt.figure()
    def idealRdDistNonSparse():
        cump = 0.
        i = 0
        rdd = np.zeros_like(trace.rdDist)
        for d in range(len(rdd)):
            while i < len(idealRdDist) and idealRdDist[i][1] <= d:
                cump += idealRdDist[i][0]
                i += 1
            rdd[d] = cump * len(trace.trace)
        return rdd

    plt.plot(idealRdDistNonSparse(), label='Ideal RD dist')
    plt.plot(np.cumsum(trace.rdDist), label='Actual RD dist')
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.plot(values)
    plt.xlabel('age')
    plt.ylabel('value')
    plt.ylim([min(values[0:max(d)]),max(values[0:max(d)])])
    plt.show()

    my_cache = cache.Cache(s,values)

    for i in range(len(trace.trace)):
        my_cache.lookup(trace.trace[i])

    miss_rate = 1.0 - float(my_cache.get_hit_rate())
    events = float(sum(my_cache.get_hit_ages()) + sum(my_cache.get_evict_ages()))
    h = my_cache.get_hit_ages() / events
    e = my_cache.get_evict_ages() / events
    # plot hit and eviction distribution
    plt.figure()
    plt.subplot(2,1,1,title='hit age distribution')
    plt.plot(np.cumsum(my_cache.get_hit_ages()/events))
    plt.subplot(2,1,2,title='evict age distribution')
    plt.plot(np.cumsum(my_cache.get_evict_ages()/events))
    plt.show()

    return miss_rate, h, e

def parse_policy(values,p,d,s):
    critical_point = [0,] + d[0:-1]
    policy = np.argsort([values[critical_point[i]+1] for i in range(len(critical_point))])
    
    if policy[0] == 0:
        miss_rate = trimodal.miss_rate_mru(p,d,s)
        h,e = trimodal.hit_rate_mru(p,d,s)
    elif policy[0] == 1:
        miss_rate = trimodal.miss_rate_d1(p,d,s)
        h,e = trimodal.hit_rate_d1(p,d,s)
    elif policy[0] == 2:
        if policy[1] == 0:
            miss_rate = trimodal.miss_rate_d2(p,d,s)
            h,e = trimodal.hit_rate_d2(p,d,s)
        elif policy[1] == 1:
            miss_rate = trimodal.miss_rate_d2_d1(p,d,s)
            h,e = trimodal.hit_rate_d2_d1(p,d,s)

    return miss_rate, h, e

def values_to_ranks(values):
    temp = values.argsort()
    ranks = np.empty(len(values),int) 
    ranks[temp] = np.arange(len(values),0,-1)
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
    rdd = np.zeros(n,dtype=float)
    f = open(filename)
    for line in f:
        d,p = line.split(' ')   
        d = int(float(d))
        p = float(p)
        rdd[d] =  p
    f.close()
    return rdd

n = 1200
if __name__ == '__main__':

    # trimodal.analysis(p,d)
    drag = 0.9999
    rdd = load_rdd("rdd.out")
    ed = 0
    for d,p in enumerate(rdd):
        ed += p*d
    print "expected reuse distance = %d" %ed
    plt.plot(rdd)
    plt.show()
    p = [0.25, 0.25, 0.5]
    d = [40,80,320]

    s = 550
    print "size = %d" %s
    values = policy_iteration(p,d,s,drag)
    #values = value_iteration(values,p,d,h,e,s,drag,True)
    log_values(values,'values-'+str(s))
