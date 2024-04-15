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

if __name__ == '__main__':
    # https://stackoverflow.com/questions/415511/how-do-i-get-the-current-time-in-python
    print(datetime.datetime.now())
    leak = np.array([10.0, 1.0, 0.1]) # Rate of application of lindblad operator
    N = 10 # Number of network nodes
    J = np.array([0.2, 0.02]) # Coherent coupling constant
    # sweep over disorder strength
    sigmaList = np.array([2.0, 1.5, 1.0, 0.5, 0.25, 0.05, 0.001])
    n_samp = 5
    dynamics = (1000,400.0)

    # star graph simulations
    graph = nx.star_graph(N-1)
    nx.draw(graph)
    plt.savefig('img/star.png')
    plt.close()

    # run collective lowering
    c_op = [collective_lowering(N)]

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            dynamics=dynamics, n_samp=n_samp)
    sweep.run()
    sweep.make_disorder_fig(fname_root='img/star')
    sweep.make_J_fig(fname_root='img/star')
    sweep.make_rate_fig(fname_root='img/star')
    sweep.save_file('star_save.npz')
    del sweep

    reloaded = param_sweep_from_file('star_save.npz')
    reloaded.make_disorder_fig(fname_root='img/reloaded')
    del reloaded

    # run independent lowering
    c_op = independent_lowering(N, 1.0)
    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            dynamics=dynamics, n_samp=n_samp)
    sweep.run()
    sweep.make_disorder_fig(fname_root='img/ind_star')
    sweep.save_file('ind_star_save.npz')
    del sweep

    # nearest neighbor graph
    graph = nx.cycle_graph(N)
    nx.draw(graph)
    plt.savefig('img/nn.png')
    plt.close()

    c_op = [collective_lowering(N)]

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            dynamics=dynamics, n_samp=n_samp)
    
    sweep.run()
    
    sweep.make_disorder_fig(fname_root='img/nn')
    sweep.make_J_fig(fname_root='img/nn')
    sweep.make_rate_fig(fname_root='img/nn')
    
    sweep.save_file('nn_save.npz')

    print(datetime.datetime.now())
