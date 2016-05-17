import sys, imp
import os.path
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer; debug_here = Tracer()

# Add gps/python to path so that imports work.
# (should be run from ~/gps)
os.chdir(os.path.expanduser('~/gps'))
sys.path.append(os.path.abspath('python'))
from gps.sample.sample_list import SampleList
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum

os.chdir('experiments/peg3D')
expts = ['lqr', 'badmm', 'mdgps_lqr', 'mdgps_nn']
colors = ['k-', 'r-', 'b-', 'g-']
iters = 12

for i, expt in enumerate(expts):
    fname = '%s/log.txt' % expt
    with open(fname) as f:
        lines = f.readlines()[-iters:]

    # ugly log.txt parsing
    lines = [l.split('|')[1] for l in lines]
    if expt == 'lqr':
        costs = [float(l) for l in lines]
    else:
        costs = [float(l.split()[1]) for l in lines]
    plt.plot(costs, colors[i], label=expt)

plt.title("3D Peg Insertion Task")
plt.legend()
plt.xlim((0, iters-1))
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.savefig('peg3D.png')
