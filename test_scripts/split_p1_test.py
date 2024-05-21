import sys
import os
# https://stackoverflow.com/questions/5180215/importing-from-subdirectories-in-python
source_path = '../src'
sys.path.append(source_path)

from graphdyn import *
from parameter_sweep import *
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import datetime


if __name__ == '__main__':
    # https://stackoverflow.com/questions/415511/how-do-i-get-the-current-time-in-python
    print(datetime.datetime.now())
    # Print source code version details
    sim_path = os.getcwd()
    os.chdir(source_path)
    os.system('git log | head -n 3')
    os.system('git status')
    os.chdir(sim_path)

    leak = np.array([1.0]) # Rate of application of lindblad operator
    N = 64 # Number of network nodes
    J = np.array([0.05]) # Coherent coupling constant
    # sweep over disorder strength
    sigmaList = np.array([2.0, 1.0, 0.5, 0.05, 0.001])
    n_samp = 5
    dynamics = (0,0.0)
    base_path = 'img/split'

    # star graph simulations
    graph = nx.star_graph(N-1)
    nx.draw(graph)
    plt.savefig(base_path + '.png')
    plt.close()

    # run independent lowering
    c_op = independent_lowering(N, 1.0)

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            dynamics=dynamics, n_samp=n_samp, spectrum=False, save_L='full',
            L_root='L_data/split')
    sweep.run()
    sweep.save_file(base_path + '.npz')
    del sweep

    print(datetime.datetime.now())
