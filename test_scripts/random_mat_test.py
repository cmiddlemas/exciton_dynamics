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
    N = 16 # Number of nodes
    leak = np.array([10.0, 1.0, 0.1]) # Rate of application of lindblad operator
    J = np.array([0.05])/np.sqrt(N-1) # Coherent coupling constant
    # sweep over disorder strength
    sigmaList = np.array([2.0, 1.0, 0.5, 0.025, 0.005])
    n_samp = 250

    # Where to save figures
    base_file = 'img/dtype'
    
    # Test on cycle graph
    graph = nx.cycle_graph(N)

    # base c_op is collective lowering
    c_op = [collective_lowering(N)]
    
    # run diagonal disorder
    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            disorder_type='diagonal', n_samp=n_samp)
    sweep.run()
    sweep.make_disorder_fig(fname_root=base_file+'_diag')
    sweep.make_rate_fig(fname_root=base_file+'_diag')
    sweep.save_file('dtype_diag.npz')
    del sweep

    # run coherent-gue
    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            disorder_type='coherent-gue', n_samp=n_samp)
    sweep.run()
    sweep.make_disorder_fig(fname_root=base_file+'_cgue')
    sweep.make_rate_fig(fname_root=base_file+'_cgue')
    sweep.save_file('dtype_cgue.npz')
    del sweep

    # run coherent-goe
    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            disorder_type='coherent-goe', n_samp=n_samp)
    sweep.run()
    sweep.make_disorder_fig(fname_root=base_file+'_cgoe')
    sweep.make_rate_fig(fname_root=base_file+'_cgoe')
    sweep.save_file('dtype_cgoe.npz')
    del sweep
    
    # run dephase-gue
    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            disorder_type='dephase-gue', n_samp=n_samp)
    sweep.run()
    sweep.make_disorder_fig(fname_root=base_file+'_dgue')
    sweep.make_rate_fig(fname_root=base_file+'_dgue')
    sweep.save_file('dtype_dgue.npz')
    del sweep

    # run dephase-goe
    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            disorder_type='dephase-goe', n_samp=n_samp)
    sweep.run()
    sweep.make_disorder_fig(fname_root=base_file+'_dgoe')
    sweep.make_rate_fig(fname_root=base_file+'_dgoe')
    sweep.save_file('dtype_dgoe.npz')
    del sweep
    
    # run loss-gue
    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            disorder_type='loss-gue', n_samp=n_samp)
    sweep.run()
    sweep.make_disorder_fig(fname_root=base_file+'_lgue')
    sweep.make_rate_fig(fname_root=base_file+'_lgue')
    sweep.save_file('dtype_lgue.npz')
    del sweep

    # run loss-goe
    sweep = ParameterSweep(graph, c_op, sigmaList, J, leak,
            disorder_type='loss-goe', n_samp=n_samp)
    sweep.run()
    sweep.make_disorder_fig(fname_root=base_file+'_lgoe')
    sweep.make_rate_fig(fname_root=base_file+'_lgoe')
    sweep.save_file('dtype_lgoe.npz')
    del sweep

    print(datetime.datetime.now())
