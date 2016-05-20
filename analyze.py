import sys, imp
import os.path
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer; debug_here = Tracer()

# Add gps/python to path so that imports work.
# (should be run from ~/gps)
sys.path.append(os.path.abspath('python'))
from gps.sample.sample_list import SampleList

tasks = ['reacher3', 'peg', 'peg_blind_big']
expts = ['badmm', 'mdgps_lqr', 'mdgps_nn', 'mdgps_lqr_new', 'mdgps_nn_new']
labels = ['BADMM', 'classic, off policy', 'classic, on policy', 'global, off policy', 'global, on policy']
seeds = [0, 1, 2]
iters = range(30)
colors = ['k', 'r', 'b', 'm', 'g']

SUCCESS_THRESHOLD = -0.5 + 0.06

def pickle_final_eepts(task, expt, seed, itr):
    print "Pickling task %s, expt %s, seed %s, itr %s" % (task, expt, seed, itr)

    # extract samples
    dirname = "experiments/%s/%s/%s/data_files" % (task, expt, seed)
    if expt == 'lqr':
        fname = "%s/traj_sample_itr_%02d.pkl" % (dirname, itr)
    else:
        fname = "%s/pol_sample_itr_%02d.pkl" % (dirname, itr)

    with open(fname, 'rb') as f:
        samples = cPickle.load(f)

    # magic final_eepts extraction (get first final_eepts position of eepts)
    final_eepts = np.array([[s._data[3][-1] for s in ss] for ss in samples]) #(M, N, num_eepts)

    # save final_eepts for faster replotting
    fname = "%s/final_eepts_itr_%02d.pkl" % (dirname, itr)
    with open(fname, 'wb') as f:
        cPickle.dump(final_eepts, f, -1)

def unpickle_final_eepts(task, expt, seed, itr):
    print "Unpickling task %s, expt %s, seed %s, itr %s" % (task, expt, seed, itr)
    dirname = "experiments/%s/%s/%s/data_files" % (task, expt, seed)
    fname = "%s/final_eepts_itr_%02d.pkl" % (dirname, itr)
    with open(fname, 'rb') as f:
        final_eepts = cPickle.load(f)
    return final_eepts
    
def get_final_eepts(task, expt, itr):
    print "Processing task %s, expt %s, itr %s" % (task, expt, itr)

    eepts = []
    for seed in seeds:
        dirname = "experiments/%s/%s/%s/data_files" % (task, expt, seed)

        # skip if the seed hasn't been run
        fname = "%s/algorithm_itr_%02d.pkl" % (dirname, itr)
        if not os.path.exists(fname):
            print "Skipping, not done, task %s, expt %s, seed %s, itr %s" % (task, expt, seed, itr)
            continue

        # pickle if the seed hasn't been pickled
        fname = "%s/final_eepts_itr_%02d.pkl" % (dirname, itr)
        if not os.path.exists(fname):
            pickle_final_eepts(task, expt, seed, itr)

        pts = unpickle_final_eepts(task, expt, seed, itr)
        if np.isnan(pts).any() or np.isinf(pts).any():
            #debug_here()
            print "Skipping, NaN/Inf, task %s, expt %s, seed %s, itr %s" % (task, expt, seed, itr)
            continue
        eepts.append(pts)

    return np.array(eepts) # (num_seeds, M, N, num_eepts)

def write_success_pcts(task, fname):
    with open(fname, 'w') as f:
        f.write('\hline\n')
        f.write('Iteration & ' + ' & '.join(labels) + '\\\\\n')
        f.write('\hline\n')
        for i in [5, 10, 15]:
            line = "%d" % i
            for expt in expts:
                eepts = get_final_eepts(task, expt, i-1)
                zpos = eepts[:, :, :, 2].flatten()
                pct = 100*np.mean(zpos < SUCCESS_THRESHOLD)
                line += ("& %.2f" % pct) #+ "~\% "
            line += "\\\\ \n"
            f.write(line)
        f.write('\hline\n')
#write_success_pcts('peg', 'peg_success.txt')
#write_success_pcts('peg_blind_big', 'peg_blind_big_success.txt')


plt.figure(figsize=(16,5))

plt.subplot(131)
plt.title("Obstacle")
task = 'obstacle'
iters = 12
for expt, color, label in zip(expts, colors, labels):
    means = []
    stdevs = []
    for itr in range(iters):
        eepts = get_final_eepts(task, expt, itr)
        if eepts.shape[0] == 0: # no more afterwards
            break
        diffs = eepts - np.array([2.5, 0, 0])
        dists = np.sqrt(np.sum(diffs**2, axis=3))

        # average over seeds first
        dists = dists.mean(axis=1)
        means.append(dists.mean())
        stdevs.append(dists.std())
    itrs = range(len(means))
    plt.errorbar(itrs, means, stdevs, c=color, label=label)
plt.legend()
plt.xlabel('Iterations')
plt.xlim((0, iters-1))
plt.ylabel('Distance to target')
plt.ylim((0, 3.0))

## Peg plot
plt.subplot(132)
plt.title("Peg Insertion")
task = 'peg'
iters = 15
for expt, color, label in zip(expts, colors, labels):
    means = []
    stdevs = []
    for itr in range(iters):
        eepts = get_final_eepts(task, expt, itr)
#        if eepts.shape[0] == 0: # no more afterwards
#            break

        # average over seeds first
        zs = eepts[:, :, :, 2].mean(axis=1)
        means.append(zs.mean() + 0.5)
        stdevs.append(zs.std())
    itrs = range(len(means))
    plt.errorbar(itrs, means, stdevs, c=color, label=label)

height = 0.1*np.ones(iters)
plt.plot(range(iters), height, 'k--')
plt.legend()
plt.xlabel('Iterations')
plt.xlim((0, iters-1))
plt.ylabel('Distance to target')
plt.ylim((0, 0.5))

## Blind peg plot
plt.subplot(133)
plt.title("Blind Peg Insertion")
task = 'peg_blind_big'
iters = 15
for expt, color, label in zip(expts, colors, labels):
    means = []
    stdevs = []
    for itr in range(iters):
        eepts = get_final_eepts(task, expt, itr)
#        if eepts.shape[0] == 0: # no more afterwards
#            break

        # average over seeds first
        zs = eepts[:, :, :, 2].mean(axis=1)
        means.append(zs.mean() + 0.5)
        stdevs.append(zs.std())
    itrs = range(len(means))
    plt.errorbar(itrs, means, stdevs, c=color, label=label)

height = 0.1*np.ones(iters)
plt.plot(range(iters), height, 'k--')
plt.legend()
plt.xlabel('Iterations')
plt.xlim((0, iters-1))
plt.ylabel('Distance to target')
plt.ylim((0, 0.5))

plt.tight_layout()
plt.show()
plt.savefig('results.png')

### Reacher plot
#plt.subplot(131)
#plt.title("Reaching")
#task = 'reacher3'
#iters = 15
#for expt, color, label in zip(expts, colors, labels):
#    means = []
#    stdevs = []
#    for itr in range(iters):
#        eepts = get_final_eepts(task, expt, itr)
#        if eepts.shape[0] == 0: # no more afterwards
#            break
#        dists = np.sqrt( np.sum(eepts[:, :, :, :3]**2, axis=3) ).flatten()
#        means.append(dists.mean())
#        stdevs.append(dists.std())
#    itrs = range(len(means))
#    plt.errorbar(itrs, means, stdevs, c=color, label=label)
#
#plt.legend()
#plt.xlabel('Iterations')
#plt.xlim((0, iters-1))
#plt.ylabel('Distance to target')
#plt.ylim((0, 0.4))

