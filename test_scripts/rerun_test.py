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
    leak = np.array([1.0]) # Rate of application of lindblad operator
    N = 16 # Number of network nodes
    J = np.array([0.05]) # Coherent coupling constant
    # sweep over disorder strength
    sigmaList = np.array([2.0, 1.0, 0.5, 0.05, 0.001])
    n_samp = 50
    dynamics = (1000,400.0)
    base_path = 'img/rerun'

    # star graph simulations
    graph = nx.star_graph(N-1)
    nx.draw(graph)
    plt.savefig(base_path + '.png')
    plt.close()

    # run collective lowering
    c_op = [collective_lowering(N)]

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            dynamics=dynamics, n_samp=n_samp)
    sweep.run()
    sweep.make_disorder_fig(fname_root=base_path)
    sweep.make_dynamics_fig(fname_root=base_path)
    sweep.save_file(base_path + '.npz')
    del sweep

    # reload the simulation and delete the old results
    reloaded = param_sweep_from_file(base_path + '.npz')
    reloaded.h_eigvals = np.array([])
    reloaded.l_eigvals = np.array([])
    reloaded.expects = np.array([])
    # Double the dynamical simulation length
    dynamics2 = (2000,800.0)
    reloaded.dynamics = dynamics2
    # Rerun the simulation, using the old samples of H and c_ops
    reloaded.run()
    reloaded.make_disorder_fig(fname_root=base_path+'2')
    reloaded.make_dynamics_fig(fname_root=base_path+'2')
    reloaded.save_file(base_path + '2' + '.npz')
    del reloaded


    print(datetime.datetime.now())
