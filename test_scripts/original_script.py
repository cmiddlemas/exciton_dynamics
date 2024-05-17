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
    plt.rcParams['text.usetex'] = True
    # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    plt.rcParams.update({'font.size': 16})

    leak = 0.3 # Base rate of application of lindblad operator
    N = 10 # Number of network nodes
    J = 0.02 # Base coherent coupling constant
    sigma = 0.02 # Base disorder strength
    sigmaMod = [2.0, 1.5, 1.0, 0.5, 0.25] # sweep over disorder strength
    nSamp = 50 # Number of samples in the ensemble
    hamStore = [] # tell the notebook to keep around the list of computed eigenvalues for some post-processing
    lioStore = [] # ham = hamiltonian, lio = liouvillian

    graph = nx.star_graph(N-1)
    C = make_coupling_H(graph)
    nx.draw(graph)

    Leig = []
    Heig = []
    for mod in sigmaMod:
        print(mod)
        Lacc = []
        Hacc = []
        for i in range(nSamp):
            H = diagonal_H(N) + diagonal_disorder(N, mod*sigma) + J*C
            L = qt.liouvillian(H, leak*collective_lowering(N))
            NSpace = sp.linalg.null_space(L.full())
            print(np.shape(NSpace)) # Check whether we have unique zero eigenvector (i.e. choice of stable base)
            Lacc += list(L.eigenenergies())
            Hacc += list(H.eigenenergies())
        Leig += [Lacc]
        Heig += [Hacc]
    lioStore += [Leig]
    hamStore += [Heig]
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/errorbar_subsample.html#sphx-glr-gallery-lines-bars-and-markers-errorbar-subsample-py
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize= [6, 9])

    ax1.hist(Heig, 100,density=True, histtype='step',
             label=["$\sigma/J =$ 2.0", "$\sigma/J =$ 1.5", "$\sigma/J =$ 1.0",
                    "$\sigma/J =$ 0.5", "$\sigma/J =$ 0.25"])
    ax1.set_title("Hamiltonian Spectrum")
    ax1.set_xlabel(r'$E$')
    ax1.set_ylabel(r'$\sigma(E)$')
    ax1.legend(loc = 'upper left')

    for l in Leig:
        ax2.scatter(np.real(l), np.imag(l), s=2)
    ax2.set_title("Liouvillian Spectrum")
    ax2.set_xlabel(r'Re$(\lambda)$')
    ax2.set_ylabel(r'Im$(\lambda)$')

    plt.tight_layout()
    plt.savefig('test_script.png')
