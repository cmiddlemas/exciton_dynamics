import sys
# https://stackoverflow.com/questions/5180215/importing-from-subdirectories-in-python
sys.path.append('../src')
from graphdyn import *
from parameter_sweep import *
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import qutip as qt
import networkx as nx
import scipy as sp
import datetime

N = 64

def graph_builder(rng):
    return nx.gnp_random_graph(N, 1.0/N, seed=rng)

if __name__ == '__main__':
    # https://stackoverflow.com/questions/415511/how-do-i-get-the-current-time-in-python
    print(datetime.datetime.now())
    leak = np.array([10.0, 1.0, 0.1]) # Rate of application of lindblad operator
    J = np.array([0.05])/np.sqrt(N-1) # Coherent coupling constant
    # sweep over disorder strength
    sigmaList = np.array([0.0])
    n_samp = 50
    dynamics = (1000,400.0)
    
    # set up a dummy graph of the same order
    graph = nx.cycle_graph(N)
    
    # bind the graph building function
    g_func = graph_builder

    # run collective lowering
    c_op = [collective_lowering(N)]

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            dynamics=dynamics, g_func=g_func, n_samp=n_samp)
    sweep.run()
    sweep.make_disorder_fig(fname_root='img/gnp')
    sweep.make_J_fig(fname_root='img/gnp')
    sweep.make_rate_fig(fname_root='img/gnp')
    sweep.make_dynamics_fig(fname_root='img/gnp')
    sweep.save_file('gnp_save.npz')
    del sweep

    print(datetime.datetime.now())
