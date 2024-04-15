"""
graphdyn.py

Allows a user to take a NetworkX graph and convert it to a QuTiP superoperator
in the single-excitation + ground-state subspace. The convention is that the
excited states occupy the first n dimensions of the underlying ket space and the
ground state occupies the last.

"""
import numpy as np
import qutip as qt
import networkx as nx

#-------------------------------------------------------------------------------
# Helper functions for creating Hamiltonians and collapse operators
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
