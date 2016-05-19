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

tasks = ['peg', 'peg_blind', 'peg_blind_big']
expts = ['lqr', 'badmm', 'mdgps_lqr', 'mdgps_nn', 'mdgps_lqr_new', 'mdgps_nn_new']
seeds = [0, 1, 2]
iters = range(30)
colors = ['k', 'r', 'b', 'c', 'm', 'g']

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
            continue

        # pickle if the seed hasn't been pickled
        fname = "%s/final_eepts_itr_%02d.pkl" % (dirname, itr)
        if not os.path.exists(fname):
            pickle_final_eepts(task, expt, seed, itr)

        eepts.append(unpickle_final_eepts(task, expt, seed, itr))

    return np.array(eepts) # (num_seeds, M, N, num_eepts)

task = tasks[0]
csv = open('peg.csv', 'w')
csv.write("iteration\t")
[csv.write("%12s\t" % expt) for expt in expts]
csv.write("\n")
for i in [10]:
    csv.write("%8s\t" % i)
    for expt in expts:
        eepts = get_final_eepts(task, expt, i-1)
        if eepts.shape[0] == 0:
            csv.write("%12s\t" % "N/A")
            continue
        zpos = eepts[:, :, :, 2].flatten()
        pct = np.mean(zpos < SUCCESS_THRESHOLD)
        csv.write("%12f\t" % pct)
    csv.write('\n')

'''
task = tasks[1]
csv = open('peg_blind.csv', 'w')
csv.write("iteration\t")
[csv.write("%12s\t" % expt) for expt in expts]
csv.write("\n")
for i in [10, 20]:
    csv.write("%8s\t" % i)
    for expt in expts:
        eepts = get_final_eepts(task, expt, i-1)
        if eepts.shape[0] == 0:
            csv.write("%12s\t" % "N/A")
            continue
        zpos = eepts[:, :, :, 2].flatten()
        pct = np.mean(zpos < SUCCESS_THRESHOLD)
        csv.write("%12f\t" % pct)
    csv.write('\n')
'''

task = tasks[2]
csv = open('peg_blind_big.csv', 'w')
csv.write("iteration\t")
[csv.write("%12s\t" % expt) for expt in expts]
csv.write("\n")
for i in [10, 20, 30]:
    csv.write("%8s\t" % i)
    for expt in expts:
        eepts = get_final_eepts(task, expt, i-1)
        if eepts.shape[0] == 0:
            csv.write("%12s\t" % "N/A")
            continue
        zpos = eepts[:, :, :, 2].flatten()
        pct = np.mean(zpos < SUCCESS_THRESHOLD)
        csv.write("%12f\t" % pct)
    csv.write('\n')


#def plot_expts(expts, colors):
#    for expt, color in zip(expts, colors):
#        print "Plotting %s" % expt
#
#        Z = np.zeros((iters, 2))
#        for itr in range(iters):
#            Z[itr] = process_z(expt, itr)
#
#        plt.errorbar(range(iters), Z[:,0] + 0.5, Z[:,1], c=color, label=expt)
#
#    height = 0.1*np.ones(iters)
#    plt.plot(range(iters), height, 'k--')
#    plt.legend()
#    plt.xlabel('Iterations')
#    plt.xlim((0, iters-1))
#    plt.ylabel('Distance to target')
#    plt.ylim((0, 0.5))
#
#
#
#plot_expts(expts, colors)
#plt.title("Blind Peg Insertion")
#plt.savefig("peg_blind_big.png")
#plt.clf()
