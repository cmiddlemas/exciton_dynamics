"""
graphdyn.py

Allows a user to take a NetworkX graph and convert it to a QuTiP superoperator
in the single-excitation + ground-state subspace. The convention is that the
excited states occupy the first n dimensions of the underlying ket space and the
ground state occupies the last.

Also provides helper functions for setting up common analyses such as tracking
eigenvalues movement as afunction of diagonal disorder and setting up common
plots, which can then be edited in the calling notebook or script.
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import qutip as qt
import networkx as nx
import scipy as sp

#-------------------------------------------------------------------------------
# Helper functions for creating Hamiltonians and collaps operators
#-------------------------------------------------------------------------------

def lowering_L(n, i):
    """Creates a lowering operator on a network of n nodes at site i"""
    M = np.zeros((n+1, n+1))
    M[-1,i] = 1.0
    return qt.Qobj(M)

def dephasing_L(n, i):
    """Creates a dephasing operator on a network of n nodes at site i"""
    M = np.zeros((n+1, n+1))
    M[i,i] = 1.0
    return qt.Qobj(M)

def collective_lowering(n):
    """
    Creates a collective (i.e. acting symmetrically on all sites) lowering
    operator on a network of n nodes
    """
    return sum([lowering_L(n, i) for i in range(n)])/np.sqrt(n)

def collective_raising(n):
    """
    Creates a collective (i.e. acting symmetrically on all sites) raising
    operator on a network of n nodes
    """
    return sum([lowering_L(n, i).dag() for i in range(n)])/np.sqrt(n)

def independent_lowering(n, sqrt_rate):
    """
    Creates a list of independent, identical lowering operators on a network of
    n nodes with coefficient sqrt_rate
    """
    return [sqrt_rate*lowering_L(n, i) for i in range(n)]

def independent_dephasing(n, sqrt_rate):
    """
    Creates a list of independent, identical dephasing operators on a network of
    n nodes with coefficient sqrt_rate
    """
    return [sqrt_rate*dephasing_L(n, i) for i in range(n)]

def make_vac(array):
    """
    Takes an array representing a nxn matrix representing the Hamiltonian of the
    excited states and returns a (n+1)x(n+1) matrix representing the addition of
    the ground state.
    """
    n = array.shape[0]
    temp = np.concatenate((array, np.zeros((1, n))))
    return np.concatenate((temp, np.zeros((n+1, 1))), axis=1)

def diagonal_H(n):
    """
    Returns a Hamiltonian that is an identity on the subspace of n excited
    states and zero on the ground state.
    """
    M = np.identity(n+1)
    M[-1,-1] = 0.0
    return qt.Qobj(M)

def make_coupling_H(G):
    """
    Takes a NetworkX graph G and returns the corresponding QuTiP Hamiltonian in
    the single-excitation + ground-state subspace.
    """
    M = nx.to_numpy_array(G)
    return qt.Qobj(make_vac(M))

def diagonal_disorder(n, sigma, rng=np.random.default_rng()):
    """
    Returns a Hamiltonian that is diagonal with entries normally distributed
    with standard deviation sigma on the subspace of n excited states and zero
    on the ground state. Also allows you to specify an instantiation of an rng,
    provided this rng provides rng.normal(offset, sigma, size). 
    """
    diag = rng.normal(0.0, sigma, n+1)
    diag[-1] = 0.0
    return(qt.Qobj(np.diag(diag)))

#-------------------------------------------------------------------------------
# Helper functions for sweeping over diagonal disorder
#-------------------------------------------------------------------------------

def make_sweep_fig(h_eigvals, l_eigvals, sigma_list, J=None):
    """Creates the appropriate eigenvalue plot"""
    plt.rcParams['text.usetex'] = True
    # https://stackoverflow.com/questions/3899980/
    # how-to-change-the-font-size-on-a-matplotlib-plot
    plt.rcParams.update({'font.size': 16})
    # https://matplotlib.org/stable/gallery/
    # lines_bars_and_markers/errorbar_subsample.html
    # #sphx-glr-gallery-lines-bars-and-markers-errorbar-subsample-py
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize= [6, 9])

    if J is None:
        label=["$\sigma =$ " + str(x) for x in sigma_list]
        ax1.hist(h_eigvals.transpose(), 'auto', density=True, histtype='step', label=label)
    else:
        label=["$\sigma/J =$ " + str(x/J) for x in sigma_list]
        ax1.hist(h_eigvals.transpose(), 'auto', density=True, histtype='step', label=label)

    ax1.set_title("Hamiltonian Spectrum")
    ax1.set_xlabel(r'$E$')
    ax1.set_ylabel(r'$\sigma(E)$')
    ax1.legend(loc = 'upper left')

    for l in l_eigvals:
        ax2.scatter(np.real(l), np.imag(l), s=2)
    ax2.set_title("Liouvillian Spectrum")
    ax2.set_xlabel(r'Re$(\lambda)$')
    ax2.set_ylabel(r'Im$(\lambda)$')

    fig.tight_layout()

    return fig

def save_sweep(fname, H, c_op, h_eigvals, l_eigvals, sigma_list, J=None):
    """Saves out the information needed to remake an eigenvalue plot"""
    H_arr = H.full()
    c_op_arr = np.array([c.full() for c in c_op])
    if J is None:
        J_arr = np.array(0.0)
    else:
        J_arr = np.array(J)

    np.savez(fname, H=H_arr, c_op=c_op_arr, h_eigvals=h_eigvals,
            l_eigvals=l_eigvals, sigma_list=sigma_list, J=J_arr)
    
    return

def diagonal_disorder_sweep(H, c_op, sigma_list, n_samp=50, J=None, rng=None):
    """
    Sweeps over the disorder strengths listed in sigma_list (1d np.array),
    for an unperturbed system defined by H (Hamilitonian) and c_op (list of
    collapse operators).  Will use n_samp realizations at each disorder
    strength, collect all resultant Hamiltonian and Liouvillian eigenvalues.

    Optional parameters:
        J = scale of coupling in H, if provided will use this information in
            legend titles

        rng = rng instance providing rng.normal

    Returns (h_eigvals, l_eigvals, l_null, fig),
    where
        h_eigvals = structured list of Hamiltonian eigenvalues
            np.array(np.array for sigma1, np.array for sigma2, ...)
        
        l_eigvals = list of Liouvillian eigenvalues
            np.array(np.array for sigma1, np.array for sigma2, ...)
        
        l_null = list of null spaces of the Liouvillian
            [list of matrices representing null spaces for sigma1, ...]
        
        fig = matplotlib figure plotting the eigenvalue data

    Developed with reference to the QuTiP implementations of eigendecomposition
    for quantum objects, but anticipate needing to modify calls for future
    features.
    """
    # Need to know how many eigenvalues we're getting to preallocate
    n_H = H.shape[0]
    n_L = n_H**2

    # https://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array
    # https://stackoverflow.com/questions/24439137/efficient-way-for-appending-numpy-array
    # Since we can preallocate though, we should
    # h_eigvals can be assumed real, l_eigvals cannot
    h_eigvals = np.zeros((len(sigma_list), n_samp*n_H), dtype=np.float64)
    l_eigvals = np.zeros((len(sigma_list), n_samp*n_L), dtype=np.complex128)
    l_null = []
    
    if not H.isherm:
        raise ValueError('Given Hamiltonian must be hermitian.')

    for i, sig in enumerate(sigma_list):
        print(sig) # good for keeping track of progress
        l_null_acc = []

        for j in range(n_samp):
            if rng is None:
                H_perturb = H + diagonal_disorder(n_H-1, sig)
            else:
                H_perturb = H + diagonal_disorder(n_H-1, sig, rng=rng)
                
            # Do an eigendecomposition on H
            h_eigvals[i, n_H*j : n_H*(j+1)] = H_perturb.eigenenergies()
            
            # Do an eigendecomposition on the Liouvillian defined by H and c_op
            L = qt.liouvillian(H_perturb, c_op)
            l_matrix = L.full()
            l_eigvals[i, n_L*j : n_L*(j+1)] = sp.linalg.eigvals(l_matrix)

            # Compute null space for L, giving steady states
            l_null_acc += [sp.linalg.null_space(l_matrix)]
            # Keep track of time and size of steady state manifold
            print(np.shape(l_null_acc[-1]))

        l_null += [l_null_acc]

    # Make the figure
    fig = make_sweep_fig(h_eigvals, l_eigvals, sigma_list, J)
    
    return h_eigvals, l_eigvals, l_null, fig
