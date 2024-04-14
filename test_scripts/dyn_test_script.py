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
    leak = np.array([10.0, 1.0, 0.1]) # Rate of application of lindblad operator
    N = 20 # Number of network nodes
    J = np.array([0.1, 0.02]) # Coherent coupling constant
    # sweep over disorder strength
    sigmaList = np.array([2.0, 1.5, 1.0, 0.5, 0.25, 0.05, 0.001])

    # star graph simulations
    graph = nx.star_graph(N-1)

    # run collective lowering
    c_op = np.array(collective_lowering(N))
    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak, spectrum=False,
            dynamics=(1000,400.0), n_samp=5)
    sweep.run()
    sweep.save_file('dyn_star_save.npz')

    # run independent lowering
    c_op = np.array(independent_lowering(N, 1.0))
    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak, spectrum=False,
            dynamics=(1000,400.0), n_samp=5)
    sweep.run()
    sweep.save_file('dyn_ind_star_save.npz')

    # nearest neighbor graph
    graph = nx.cycle_graph(N)
    c_op = np.array(collective_lowering(N))
    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak, spectrum=False,
            dynamics=(1000,400.0), n_samp=5)
    sweep.run()
    sweep.save_file('dyn_nn_save.npz')
