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

iters = 12

def pickle_z(expt, itr):
    print "Pickling %s, itr %s" % (expt, itr)

    # extract samples
    dirname = "%s/data_files" % (expt)
    if expt == 'lqr':
        fname = "%s/traj_sample_itr_%02d.pkl" % (dirname, itr)
    else:
        fname = "%s/pol_sample_itr_%02d.pkl" % (dirname, itr)

    with open(fname, 'rb') as f:
        samples = cPickle.load(f)

    # magic z extraction (get first z position of eepts)
    z = np.array([[s._data[3][-1][2] for s in ss] for ss in samples]) #(M, N)

    # save z for faster replotting
    fname = "%s/z_itr_%02d.pkl" % (dirname, itr)
    with open(fname, 'wb') as f:
        cPickle.dump(z, f, -1)

def unpickle_z(expt, itr):
    dirname = "%s/data_files" % (expt)
    fname = "%s/z_itr_%02d.pkl" % (dirname, itr)
    with open(fname, 'rb') as f:
        z = cPickle.load(f)
    return z
    
def process_z(expt, itr):
    print "Processing %s, itr %s" % (expt, itr)

    dirname = "%s/data_files" % (expt)

    # skip if not done
    fname = "%s/algorithm_itr_%02d.pkl" % (dirname, itr)
    if not os.path.exists(fname):
        return

    # pickle if hasn't been pickled
    fname = "%s/z_itr_%02d.pkl" % (dirname, itr)
    if not os.path.exists(fname):
        pickle_z(expt, itr)

    z = unpickle_z(expt, itr)
    return np.mean(z), np.std(z)

def plot_expts(expts, colors):
    for expt, color in zip(expts, colors):
        print "Plotting %s" % expt

        Z = np.zeros((iters, 2))
        for itr in range(iters):
            Z[itr] = process_z(expt, itr)

        plt.errorbar(range(iters), Z[:,0] + 0.5, Z[:,1], c=color, label=expt)

    height = 0.1*np.ones(iters)
    plt.plot(range(iters), height, 'k--')
    plt.legend()
    plt.xlabel('Iterations')
    plt.xlim((0, iters-1))
    plt.ylabel('Distance to target')
    plt.ylim((0, 0.5))


old_expts = ['lqr', 'badmm', 'mdgps_lqr', 'mdgps_nn']
new_expts = ['lqr', 'badmm', 'mdgps_lqr_new', 'mdgps_nn_new']
colors = ['k', 'r', 'b', 'g']

plot_expts(old_expts, colors)
plt.title("Peg Insertion (hard, old step size)")
plt.savefig("hard_peg_debug_old.png")
plt.clf()

plot_expts(new_expts, colors)
plt.title("Peg Insertion (hard, new step size)")
plt.savefig("hard_peg_debug_new.png")
plt.clf()
