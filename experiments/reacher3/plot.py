import sys, imp
import os.path
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer; debug_here = Tracer()

# Add gps/python to path so that imports work.
# (should be run from ~/gps)
dirname = '/'.join(str.split(os.path.abspath(str(__file__)), '/')[:-1])
os.chdir(os.path.expanduser('~/gps'))
sys.path.append(os.path.abspath('python'))
from gps.sample.sample_list import SampleList
from gps.algorithm.cost.cost_sum import CostSum
os.chdir(dirname)

iters = 15
seeds = [0, 1, 2]

def pickle_dist(expt, seed, itr):
    print "Pickling %s, seed %s, itr %s" % (expt, seed, itr)

    # extract samples
    dirname = "%s/%s/data_files" % (expt, seed)
    if expt == 'lqr':
        fname = "%s/traj_sample_itr_%02d.pkl" % (dirname, itr)
    else:
        fname = "%s/pol_sample_itr_%02d.pkl" % (dirname, itr)

    with open(fname, 'rb') as f:
        samples = cPickle.load(f)

    # magic dist extraction (get first dist position of eepts)
    dist = np.array([[s._data[3][-1][2] for s in ss] for ss in samples]) #(M, N)

    # save dist for faster replotting
    fname = "%s/dist_itr_%02d.pkl" % (dirname, itr)
    with open(fname, 'wb') as f:
        cPickle.dump(dist, f, -1)

def unpickle_dist(expt, seed, itr):
    print "Unpickling %s, seed %s, itr %s" % (expt, seed, itr)

    dirname = "%s/%s/data_files" % (expt, seed)
    fname = "%s/dist_itr_%02d.pkl" % (dirname, itr)
    with open(fname, 'rb') as f:
        dist = cPickle.load(f)
    return dist
    
def process_dist(expt, itr):
    print "Processing %s, itr %s" % (expt, itr)

    dist = []
    for seed in seeds:
        dirname = "%s/%s/data_files" % (expt, seed)

        # skip if the seed hasn't been run
        fname = "%s/algorithm_itr_%02d.pkl" % (dirname, itr)
        if not os.path.exists(fname):
            continue

        # pickle if the seed hasn't been pickled
        fname = "%s/dist_itr_%02d.pkl" % (dirname, itr)
        if not os.path.exists(fname):
            pickle_dist(expt, seed, itr)

        dist.append(unpickle_dist(expt, seed, itr))

    dist = np.array(dist)
    return np.mean(dist), np.std(dist)

def plot_expts(expts, colors):
    for expt, color in zip(expts, colors):
        print "Plotting %s" % expt

        D = np.zeros((iters, 2))
        for itr in range(iters):
            D[itr] = process_dist(expt, itr)

        plt.errorbar(range(iters), D[:,0], D[:,1], c=color, label=expt)

    plt.legend()
    plt.xlabel('Iterations')
    plt.xlim((0, iters-1))
    plt.ylabel('Distance to target')
    plt.ylim((0, 0.5))


expts = ['lqr', 'badmm', 'mdgps_lqr', 'mdgps_lqr_new', 'mdgps_nn_new']
colors = ['k', 'r', 'b', 'm', 'g']

plot_expts(expts, colors)
plt.title("Reacher3 (old step rule)")
plt.savefig("reacher3.png")
plt.clf()
