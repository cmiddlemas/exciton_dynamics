import sys
import os
# https://stackoverflow.com/questions/5180215/importing-from-subdirectories-in-python
source_path = '/home/tm17/source_code/exciton_dynamics/src'
sys.path.append(source_path)

from graphdyn import *
from parameter_sweep import *
import numpy as np
import networkx as nx
import datetime

N = 64 # Number of network nodes
def graph_builder(rng):
    return nx.gnm_random_graph(N, 3, seed=rng)

if __name__ == '__main__':
    data_path = './'
    base_name = 'gnm_m3'
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
    leak = np.array([10.0, 1.0, 0.1])
    # Coherent coupling constant
    J = np.array([0.05])/np.sqrt(N-1)
    # sweep over disorder strength
    sigmaList = np.array([2.0])
    n_samp = 50
    dynamics = (1000,400.0)

    # make a dummy graph with the right number of vertices
    graph = nx.wheel_graph(N)

    # run collective lowering
    print('running cl')
    c_op = [collective_lowering(N)]
    current_path = base_path + '_cl'

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            dynamics=dynamics, n_samp=n_samp, g_func=graph_builder)
    sweep.run()
    sweep.make_disorder_fig(fname_root=current_path)
    sweep.make_rate_fig(fname_root=current_path)
    sweep.make_dynamics_fig(fname_root=current_path)
    sweep.save_file(current_path + '.npz')
    del sweep

    # run independent lowering
    print('running il')
    c_op = independent_lowering(N, 1.0)
    current_path = base_path + '_il'

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            dynamics=dynamics, n_samp=n_samp, g_func=graph_builder)
    sweep.run()
    sweep.make_disorder_fig(fname_root=current_path)
    sweep.make_rate_fig(fname_root=current_path)
    sweep.make_dynamics_fig(fname_root=current_path)
    sweep.save_file(current_path + '.npz')
    del sweep

    # run collective dephasing
    print('running cd')
    c_op = [collective_dephasing(N)]
    current_path = base_path + '_cd'

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            dynamics=dynamics, n_samp=n_samp, g_func=graph_builder)
    sweep.run()
    sweep.make_disorder_fig(fname_root=current_path)
    sweep.make_rate_fig(fname_root=current_path)
    sweep.make_dynamics_fig(fname_root=current_path)
    sweep.save_file(current_path + '.npz')
    del sweep

    # run indepdendent dephasing
    print('running id')
    c_op = independent_dephasing(N, 1.0)
    current_path = base_path + '_id'

    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            dynamics=dynamics, n_samp=n_samp, g_func=graph_builder)
    sweep.run()
    sweep.make_disorder_fig(fname_root=current_path)
    sweep.make_rate_fig(fname_root=current_path)
    sweep.make_dynamics_fig(fname_root=current_path)
    sweep.save_file(current_path + '.npz')
    del sweep

    print(datetime.datetime.now())
