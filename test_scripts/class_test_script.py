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
    leak = np.array([0.3, 1.0])**2 # Base rate of application of lindblad operator
    N = 10 # Number of network nodes
    J = np.array([0.02, 0.2]) # Base coherent coupling constant
    sigmaList = 0.02*np.array([2.0, 1.5, 1.0, 0.5, 0.25, 0.05, 0.0]) # sweep over disorder strength

    graph = nx.star_graph(N-1)
    nx.draw(graph)
    plt.savefig('test_graph.png')
    plt.close()

    c_op = np.array(collective_lowering(N))

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak)
    
    sweep.run()

    sweep.make_disorder_fig(fname_root='test')
    
    sweep.save_file('test_save.npz')

    reloaded = param_sweep_from_file('test_save.npz')
   
    reloaded.make_disorder_fig(fname_root='reloaded')
