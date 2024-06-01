import sys
import os
# https://stackoverflow.com/questions/5180215/importing-from-subdirectories-in-python
source_path = '/home/tm17/source_code/exciton_dynamics/src'
sys.path.append(source_path)

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
    data_path = './'
    base_name = 'star_cgue'
    base_path = data_path + base_name
    print('Running: ' + base_name)

    # https://stackoverflow.com/questions/415511/how-do-i-get-the-current-time-in-python
    print(datetime.datetime.now())
    # Print source code version details
    sim_path = os.getcwd()
    os.chdir(source_path)
    os.system('git log | head -n 3')
    os.system('git status')
    os.chdir(sim_path)

    # Rate of application of lindblad operator
    leak = np.array([1.0])
    N = 64 # Number of network nodes
    # Coherent coupling constant
    J = np.array([0.05])/np.sqrt(N-1)
    # sweep over disorder strength
    sigmaList = np.array([2.0, 1.5, 1.0, 0.5, 0.25, 0.05, 0.001])
    n_samp = 50
    dynamics = (1000,400.0)
    # disorder type
    disorder_type = 'coherent-gue'

    # star graph simulations
    graph = nx.star_graph(N-1)

    # run collective lowering
    print('running cl')
    c_op = [collective_lowering(N)]
    current_path = base_path + '_cl'

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            n_samp=n_samp, dynamics=dynamics, disorder_type=disorder_type)
    sweep.run()
    sweep.make_disorder_fig(fname_root=current_path)
    sweep.make_J_fig(fname_root=current_path)
    sweep.make_rate_fig(fname_root=current_path)
    sweep.make_dynamics_fig(fname_root=current_path)
    sweep.save_file(current_path + '.npz')
    del sweep

    # run independent lowering
    print('running il')
    c_op = independent_lowering(N, 1.0)
    current_path = base_path + '_il'

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            n_samp=n_samp, dynamics=dynamics, disorder_type=disorder_type)
    sweep.run()
    sweep.make_disorder_fig(fname_root=current_path)
    sweep.make_J_fig(fname_root=current_path)
    sweep.make_rate_fig(fname_root=current_path)
    sweep.make_dynamics_fig(fname_root=current_path)
    sweep.save_file(current_path + '.npz')
    del sweep

    # run collective dephasing
    print('running cd')
    c_op = [collective_dephasing(N)]
    current_path = base_path + '_cd'

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            n_samp=n_samp, dynamics=dynamics, disorder_type=disorder_type)
    sweep.run()
    sweep.make_disorder_fig(fname_root=current_path)
    sweep.make_J_fig(fname_root=current_path)
    sweep.make_rate_fig(fname_root=current_path)
    sweep.make_dynamics_fig(fname_root=current_path)
    sweep.save_file(current_path + '.npz')
    del sweep

    # run independent dephasing
    print('running id')
    c_op = independent_dephasing(N, 1.0)
    current_path = base_path + '_id'

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            n_samp=n_samp, dynamics=dynamics, disorder_type=disorder_type)
    sweep.run()
    sweep.make_disorder_fig(fname_root=current_path)
    sweep.make_J_fig(fname_root=current_path)
    sweep.make_rate_fig(fname_root=current_path)
    sweep.make_dynamics_fig(fname_root=current_path)
    sweep.save_file(current_path + '.npz')
    del sweep

    print(datetime.datetime.now())
