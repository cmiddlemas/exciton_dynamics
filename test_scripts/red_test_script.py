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
    leak = 0.3 # Base rate of application of lindblad operator
    N = 10 # Number of network nodes
    J = 0.02 # Base coherent coupling constant
    sigmaList = 0.02*np.array([2.0, 1.5, 1.0, 0.5, 0.25]) # sweep over disorder strength
    nSamp = 50 # Number of samples in the ensemble

    graph = nx.star_graph(N-1)
    C = make_coupling_H(graph)
    nx.draw(graph)
    plt.savefig('test_graph.png')
    plt.close()

    H = diagonal_H(N) + J*C
    c_op = [leak*collective_lowering(N)]

    Heig, Leig, l_null, fig = diagonal_disorder_sweep(H, c_op, sigmaList,
            n_samp=nSamp, J=J)
    
    fig.savefig('test_script.png')

    save_sweep('test_save.npz', H, c_op, Heig, Leig, sigmaList, J)

    reloaded = np.load('test_save.npz')
    print(reloaded.files)
    print(reloaded['H'])
    print(reloaded['c_op'])
    print(reloaded['h_eigvals'])
    
    fig2 = make_sweep_fig(reloaded['h_eigvals'], reloaded['l_eigvals'],
            reloaded['sigma_list'], reloaded['J'])
    fig2.savefig('test_reloaded.png')
