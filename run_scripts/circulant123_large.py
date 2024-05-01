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
    data_path = '/scratch/gpfs/tm17/graphdyn/circulant123_large/'
    base_name = 'circulant123_large'
    base_path = data_path + base_name
    print('Running: ' + base_name)
    
    # https://stackoverflow.com/questions/415511/how-do-i-get-the-current-time-in-python
    print(datetime.datetime.now())
    # Rate of application of lindblad operator
    leak = np.array([10.0, 1.0, 0.1])
    N = 64 # Number of network nodes
    # Coherent coupling constant
    J = np.array([0.05, 0.005])/np.sqrt(N-1)
    # sweep over disorder strength
    sigmaList = np.array([2.0, 1.5, 1.0, 0.5, 0.25, 0.05, 0.001])
    n_samp = 50
    dynamics = (1000,400.0)

    # star graph simulations
    graph = nx.circulant_graph(N, [1,2,3])
    nx.draw(graph)
    plt.savefig(base_path + '.png')
    plt.close()

    # run collective lowering
    print('running cl')
    c_op = [collective_lowering(N)]
    cl_path = base_path + '_cl'

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            dynamics=dynamics, n_samp=n_samp)
    sweep.run()
    sweep.make_disorder_fig(fname_root=cl_path)
    sweep.make_J_fig(fname_root=cl_path)
    sweep.make_rate_fig(fname_root=cl_path)
    sweep.make_dynamics_fig(fname_root=cl_path)
    sweep.save_file(cl_path + '.npz')
    del sweep

    # run independent lowering
    print('running il')
    c_op = independent_lowering(N, 1.0)
    il_path = base_path + '_il'

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            dynamics=dynamics, n_samp=n_samp)
    sweep.run()
    sweep.make_disorder_fig(fname_root=il_path)
    sweep.make_J_fig(fname_root=il_path)
    sweep.make_rate_fig(fname_root=il_path)
    sweep.make_dynamics_fig(fname_root=il_path)
    sweep.save_file(il_path + '.npz')
    del sweep

    # run collective dephasing
    print('running cd')
    c_op = [collective_dephasing(N)]
    cd_path = base_path + '_cd'

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            dynamics=dynamics, n_samp=n_samp)
    sweep.run()
    sweep.make_disorder_fig(fname_root=cd_path)
    sweep.make_J_fig(fname_root=cd_path)
    sweep.make_rate_fig(fname_root=cd_path)
    sweep.make_dynamics_fig(fname_root=cd_path)
    sweep.save_file(cd_path + '.npz')
    del sweep

    # run independent dephasing
    print('running id')
    c_op = independent_dephasing(N, 1.0)
    id_path = base_path + '_id'

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            dynamics=dynamics, n_samp=n_samp)
    sweep.run()
    sweep.make_disorder_fig(fname_root=id_path)
    sweep.make_J_fig(fname_root=id_path)
    sweep.make_rate_fig(fname_root=id_path)
    sweep.make_dynamics_fig(fname_root=id_path)
    sweep.save_file(id_path + '.npz')
    del sweep

    print(datetime.datetime.now())
