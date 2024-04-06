import sys
# https://stackoverflow.com/questions/5180215/importing-from-subdirectories-in-python
sys.path.append('../src')
from graphdyn import *
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import qutip as qt
import networkx as nx
import scipy as sp

if __name__ == '__main__':
    leak = np.array([1.0, 0.1]) # Rate of application of lindblad operator
    N = 10 # Number of network nodes
    J = np.array([0.1, 0.02]) # Coherent coupling constant
    # sweep over disorder strength
    sigmaList = np.array([2.0, 1.5, 1.0, 0.5, 0.25, 0.05, 0.001])

    graph = nx.star_graph(N-1)
    nx.draw(graph)
    plt.savefig('star.png')
    plt.close()

    c_op = np.array(collective_lowering(N))

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak)
    
    sweep.run()

    sweep.make_disorder_fig(fname_root='star')
    sweep.make_J_fig(fname_root='star')
    sweep.make_rate_fig(fname_root='star')
    
    sweep.save_file('star_save.npz')

    reloaded = param_sweep_from_file('star_save.npz')
   
    reloaded.make_disorder_fig(fname_root='reloaded')

    graph = nx.cycle_graph(N)
    nx.draw(graph)
    plt.savefig('nn.png')
    plt.close()

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak)
    
    sweep.run()
    
    sweep.make_disorder_fig(fname_root='nn')
    sweep.make_J_fig(fname_root='nn')
    sweep.make_rate_fig(fname_root='nn')
    
    sweep.save_file('nn_save.npz')
