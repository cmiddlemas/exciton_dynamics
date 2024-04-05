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
# ParameterSweep class to structure exploring numerical parameters
#-------------------------------------------------------------------------------

class ParameterSweep:
    """
    Holds the data needed to create plots and contains methods to run
    simulations
    """

    def __init__(self, G, c_op, sigma, J, rate, n_samp=50,
            rng=np.random.default_rng(), h_eigvals=np.array([]),
            l_eigvals=np.array([]), l_null=[]):
        """
        Parameters:
        G = nx.graph
        c_op = np.array of Qobj
        sigma = np.array
        J = np.array
        rate = np.array
        n_samp = int
        rng = instance of rng
        
        Simulation Results:
        h_eigvals = np.array
        l_eigvals = np.array
        l_null = list of np.array
        """
        self.G = G
        self.c_op = c_op
        self.sigma = sigma
        self.J = J
        self.rate = rate
        self.n_samp = n_samp
        self.rng = rng
        
        self.h_eigvals = h_eigvals
        self.l_eigvals = l_eigvals
        self.l_null = l_null

    def run(self):
        """
        Will sweep through all parameters for
        H = diag - J*C + disorder
        c_op = rate*c_op
        """
        # Make coupling matrix
        C = make_coupling_H(self.G)

        # Need to know how many eigenvalues we're getting to preallocate
        n_H = C.shape[0]
        n_L = n_H**2
        n_sigma = self.sigma.size
        n_J = self.J.size
        n_rate = self.rate.size

        # for a general overview of array/list strategies see:
        # https://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array
        # https://stackoverflow.com/questions/24439137/efficient-way-for-appending-numpy-array
        # Our case is dead simple though, since we can just preallocate
        self.h_eigvals = np.zeros((n_sigma, n_J, n_rate, self.n_samp*n_H),
                dtype=np.float64)
        self.l_eigvals = np.zeros((n_sigma, n_J, n_rate, self.n_samp*n_L),
                dtype=np.complex128)
        self.l_null = []

        for i, sig in enumerate(self.sigma):
            print(sig)
            
            for j, coup in enumerate(self.J):
                H = diagonal_H(n_H-1) - coup*C
                
                for k, gamma in enumerate(self.rate):
                    scaled_c_op = np.sqrt(gamma)*self.c_op
                    l_null_acc = []
                    
                    for m in range(self.n_samp):
                        H_perturb = H + diagonal_disorder(n_H-1, sig, rng=self.rng)
                
                        # Do an eigendecomposition on H
                        self.h_eigvals[i, j, k, n_H*m:n_H*(m+1)] = \
                            H_perturb.eigenenergies()
            
                        # Do an eigendecomposition on the Liouvillian defined by
                        # H and c_op
                        L = qt.liouvillian(H_perturb, scaled_c_op)
                        l_matrix = L.full()
                        self.l_eigvals[i, j, k, n_L*m:n_L*(m+1)] = \
                            sp.linalg.eigvals(l_matrix)

                        # Compute null space for L, giving steady states
                        l_null_acc += [sp.linalg.null_space(l_matrix)]
                        # Keep track of time and size of steady state manifold
                        #print(np.shape(l_null_acc[-1]))

                    self.l_null += [l_null_acc]


    def make_disorder_fig(self, fname_root=None):
        """
        Makes eigvalue plots that directly reveals variation over disorder
        strength
        """
        plt.rcParams['text.usetex'] = True
        # https://stackoverflow.com/questions/3899980/
        # how-to-change-the-font-size-on-a-matplotlib-plot
        plt.rcParams.update({'font.size': 16})
        # https://matplotlib.org/stable/gallery/
        # lines_bars_and_markers/errorbar_subsample.html
        # #sphx-glr-gallery-lines-bars-and-markers-errorbar-subsample-py

        fig_list = []

        for i, coup in enumerate(self.J):
            for j, gamma in enumerate(self.rate):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize= [6, 9])
                label=["$\sigma/J =$ " + "{:3g}".format(x/coup) for x in self.sigma]
                ax1.hist(self.h_eigvals[:,i,j,:].transpose(), 'auto',
                        density=True, histtype='step', label=label)

                ax1.set_title("Hamiltonian Spectrum")
                ax1.set_xlabel(r'$E$')
                ax1.set_ylabel(r'$\sigma(E)$')
                ax1.legend(loc = 'upper left')

                for l in self.l_eigvals[:,i,j,:]:
                    ax2.scatter(np.real(l), np.imag(l), s=2)
                    ax2.set_title("Liouvillian Spectrum")
                    ax2.set_xlabel(r'Re$(\lambda)$')
                    ax2.set_ylabel(r'Im$(\lambda)$')

                fig.tight_layout()

                if fname_root is not None:
                    fig.savefig(fname_root + '_J' + str(coup) + '_rate' +
                            str(gamma) + '.png')
                
                fig_list += [fig]

        return fig_list


    def make_J_fig(self, fname_root=None):
        return

    def make_rate_fig(self, fname_root=None):
        return

    def save_file(self, fname):
        """
        Saves a binary representation of the instance to a .pyz file. Keeps all
        information except rng and l_null, which cannot be saved in array
        format.
        """
        G_arr = nx.to_numpy_array(self.G)
        c_op_arr = np.array([c.full() for c in np.atleast_1d(self.c_op)])

        np.savez(fname, G=G_arr, c_op=c_op_arr, sigma=self.sigma, J=self.J,
                rate=self.rate, n_samp=self.n_samp, h_eigvals=self.h_eigvals,
                l_eigvals=self.l_eigvals)

        

def param_sweep_from_file(fname, rng=np.random.default_rng()):
    """
    Load binary representation made by ParameterSweep.save_file(fname). However,
    will need a new rng and will have an empty list of l_null, due to the
    inability to save those out as np.arrays.
    """
    raw = np.load(fname)
    G = nx.from_numpy_array(raw['G'])
    c_op = np.array([qt.Qobj(c) for c in raw['c_op']])
    
    return ParameterSweep(G, c_op, sigma=raw['sigma'], J=raw['J'],
            rate=raw['rate'], n_samp=raw['n_samp'].item(), rng=rng,
            h_eigvals=raw['h_eigvals'], l_eigvals=raw['l_eigvals'])
