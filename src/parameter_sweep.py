"""
parameter_sweep.py

Provides a ParameterSweep class to structure exploring numerical parameters

"""
import numpy as np
from matplotlib import pyplot as plt
import qutip as qt
import networkx as nx
import scipy as sp
import gc
import graphdyn
import sys

class ParameterSweep:
    """
    Holds the data needed to create plots and contains methods to run
    simulations
    """

    def __init__(self,
                 G,
                 c_op,
                 sigma,
                 J,
                 rate,
                 n_samp=50,
                 rng=np.random.default_rng(),
                 spectrum=True,
                 dynamics=(0, 0.0),
                 g_func=None,
                 disorder_type='diagonal',
                 save_L='partial',
                 L_root='',
                 h_eigvals=np.array([]),
                 l_eigvals=np.array([]),
                 l_singvals=np.array([]),
                 l_null=[],
                 d_null=np.array([]),
                 cond=np.array([]),
                 depF=np.array([]),
                 times=np.array([]),
                 expects=np.array([]),
                 gaps=np.array([]),
                 h_instances=np.array([]),
                 c_instances=np.array([])):
        """
        Parameters:
        G = nx.graph representing connectivity graph
        c_op = Python list of Qobj, needs to be a list for QuTiP to work,
               represents collapse operators
        sigma = np.array of \sigma / J
        J = np.array of J, the coupling constant
        rate = np.array of dissipation rates (\gamma / J)
        n_samp = int, number of samples to take at each set of parameters
        rng = instance of rng
        spectrum = bool, do we compute spectra?
        dynamics = (int, float), (number of dynamics timesteps, time cutoff)
                   if number of steps is 0, skip dynamics calculations
                   time cutoff will be scaled by the dissipation rate
        g_func = None or function, if function, sample from the random graph
                 ensemble specified by the function. The sole parameter passed
                 to the function will be an rng.
        disorder_type = string, 'diagonal' for perturbing exciton-GS gap,
                        'coherent-gue' and 'coherent-goe' for perturbing
                        Hamiltonian in 1-exc subspace using appropriate random
                        matrix ensemble, 'dephase-gue' and 'dephase-goe' for
                        adding a Lindblad operator in 1-exc subspace, and
                        'loss-gue' and 'loss-goe' for adding a Lindblad operator
                        in the 0-1-exc subspace, 'read_H' to take the data from
                        self.h_instances and self.c_instances, and 'read_L' to
                        take the data from the L files
        save_L = string, 'none' saves nothing, 'partial' saves H and c_ops,
                 and 'full' saves H and c_ops, as well as L to files with base
                 path given by L_path
        L_root = string, specifies the path and basename used by
                 disorder_type='read_L'

        Simulation Results: Order of array dimensions is [sigma, J, rate, samp]
        h_eigvals = np.array of Hamiltonian eigenvalues
        l_eigvals = np.array of Liouvillian eigenvalues
        l_singvals = np.array of Liouvillian singular values
        l_null = list of np.array giving bases for Liouvillian null spaces
        d_null = np.array of dimensions of Liouvillian null spaces
        cond = np.array of condition numbers of Liouvillian eigenvector matrix
        depF = np.array of departures from normality in the Frobenious norm of
               the Liouvillian
        times = np.array of dynamics timepoints
        expects = np.array of dynamical expectation values
        gaps = np.array of Hamiltonian spectral gaps

        Information for rerunning analyses:
        h_instances = np.array of realizations of H
        c_instances = np.array of realizations of c_op
        """
        self.G = G
        self.c_op = c_op
        self.sigma = sigma
        self.J = J
        self.rate = rate
        self.n_samp = n_samp
        self.rng = rng
        self.spectrum = spectrum
        self.dynamics = dynamics
        self.g_func = g_func
        self.disorder_type = disorder_type
        self.save_L = save_L
        self.L_root = L_root

        self.h_eigvals = h_eigvals
        self.l_eigvals = l_eigvals
        self.l_singvals = l_singvals
        self.l_null = l_null
        self.d_null = d_null
        self.cond = cond
        self.depF = depF
        self.times = times
        self.expects = expects
        self.gaps = gaps

        self.h_instances = h_instances
        self.c_instances = c_instances

    def random_sample(self, H, fixed_c_op, sig, coup, gamma, i, j, k, m):
        """
        Samples H and c_ops from distribution determined by disorder_type or
        g_func
        """
        n_H = H.shape[0]

        if self.g_func is None:
            match self.disorder_type:
                case 'diagonal':
                    # Disorder from diagonal perturbation
                    perturb = graphdyn.diagonal_disorder(n_H-1, sig*coup,
                            rng=self.rng)
                    return H + perturb, fixed_c_op

                case 'coherent-gue':
                    # Add to the Hamiltonian a matrix drawn from
                    # the GUE in the 1-exc subspace
                    perturb = qt.Qobj(graphdyn.make_vac(
                            graphdyn.gue_matrix(n_H-1, sig*coup, rng=self.rng)))
                    return H + perturb, fixed_c_op

                case 'coherent-goe':
                    # Add to the Hamiltonian a matrix drawn from
                    # the GOE in the 1-exc subspace
                    perturb = qt.Qobj(graphdyn.make_vac(
                            graphdyn.goe_matrix(n_H-1, sig*coup, rng=self.rng)))
                    return H + perturb, fixed_c_op

                case 'dephase-gue':
                    # Add to the c_op list a matrix drawn from the GUE in the
                    # 1-exc subspace
                    s_rate = np.sqrt(sig*coup*gamma)
                    c_op = qt.Qobj(graphdyn.make_vac(
                            graphdyn.gue_matrix(n_H-1, s_rate, rng=self.rng)))
                    return H, [c_op] + fixed_c_op

                case 'dephase-goe':
                    # Add to the c_op list a matrix drawn from the GOE in the
                    # 1-exc subspace
                    s_rate = np.sqrt(sig*coup*gamma)
                    c_op = qt.Qobj(graphdyn.make_vac(
                            graphdyn.goe_matrix(n_H-1, s_rate, rng=self.rng)))
                    return H, [c_op] + fixed_c_op

                case 'loss-gue':
                    # Add to the c_op list a matrix drawn from the GUE in the
                    # 1-exc subspace
                    s_rate = np.sqrt(sig*coup*gamma)
                    c_op = qt.Qobj(
                            graphdyn.gue_matrix(n_H, s_rate, rng=self.rng))
                    return H, [c_op] + fixed_c_op

                case 'loss-goe':
                    # Add to the c_op list a matrix drawn from the GOE in the
                    # 1-exc subspace
                    s_rate = np.sqrt(sig*coup*gamma)
                    c_op = qt.Qobj(
                            graphdyn.goe_matrix(n_H, s_rate, rng=self.rng))
                    return H, [c_op] + fixed_c_op

                case 'read_H' | 'read_L':
                    # Read the Hamiltonians out of h_instances and the c_ops out
                    # of c_instances
                    H_samp = qt.Qobj(self.h_instances[i, j, k, m, :, :])
                    c_op_samp = [qt.Qobj(c_op) for c_op in self.c_instances[i,
                            j, k, m, :, :, :]]
                    return H_samp, c_op_samp

                case _:
                    raise Exception('Invalid disorder_type')

        else:
            # Disorder from random graph ensemble
            rand_g = self.g_func(self.rng)
            rand_c = graphdyn.make_coupling_H(rand_g)
            H_samp = graphdyn.diagonal_H(n_H-1) - coup*rand_c
            return H_samp, fixed_c_op


    def run(self):
        """
        Will sweep through all parameters for disorder, coherent coupling
        constant, and incoherent rate
        """
        # Make coupling matrix
        C = graphdyn.make_coupling_H(self.G)

        # Need to know how many eigenvalues we're getting to preallocate
        n_H = C.shape[0]
        n_L = n_H**2
        n_sigma = self.sigma.size
        n_J = self.J.size
        n_rate = self.rate.size
        n_samp = self.n_samp
        match self.disorder_type:
            case 'dephase-gue' | 'dephase-goe' | 'loss-gue' | 'loss-goe':
                n_c_op = len(self.c_op) + 1
            case _:
                n_c_op = len(self.c_op)

        # for a general overview of array/list strategies see:
        # https://stackoverflow.com/questions/7133885
        # /fastest-way-to-grow-a-numpy-numeric-array
        # https://stackoverflow.com/questions/24439137
        # /efficient-way-for-appending-numpy-array
        # Our case is dead simple though, since we can just preallocate
        self.h_eigvals = np.zeros((n_sigma, n_J, n_rate, n_samp*n_H),
                dtype=np.float64)
        self.l_eigvals = np.zeros((n_sigma, n_J, n_rate, n_samp*n_L),
                dtype=np.complex128)
        self.l_singvals = np.zeros((n_sigma, n_J, n_rate, n_samp*n_L),
                dtype=np.float64)
        self.l_null = []
        self.d_null = np.zeros((n_sigma, n_J, n_rate, n_samp), dtype=int)
        self.cond = np.zeros((n_sigma, n_J, n_rate, n_samp), dtype=np.float64)
        self.depF = np.zeros((n_sigma, n_J, n_rate, n_samp), dtype=np.complex128)

        # Make some room to store realizations of H and c_ops
        match self.save_L:
            case 'partial' | 'full':
                self.h_instances = np.zeros(
                        (n_sigma, n_J, n_rate, n_samp, n_H, n_H),
                        dtype=np.complex128)
                self.c_instances = np.zeros(
                        (n_sigma, n_J, n_rate, n_samp, n_c_op, n_H, n_H),
                        dtype=np.complex128)

        # Prep times array if we are running dynamics
        t_steps = self.dynamics[0]
        t_cut = self.dynamics[1]
        if t_steps > 0:
            self.times = np.linspace(0.0, t_cut, t_steps)
            self.expects = np.zeros((n_sigma, n_J, n_rate, n_samp, 4, t_steps),
                    dtype=np.complex128)
            self.gaps = np.zeros((n_sigma, n_J, n_rate, n_samp),
                    dtype=np.float64)

        for i, sig in enumerate(self.sigma):
            for j, coup in enumerate(self.J):
                H = graphdyn.diagonal_H(n_H-1) - coup*C

                for k, gamma in enumerate(self.rate):
                    print('sigma=' + str(sig)
                            + ', J=' + str(coup)
                            + ', rate=' + str(gamma))
                    # Scale the c_ops
                    scaled_c_op = [np.sqrt(gamma*coup)*c for c in self.c_op]
                    # Scale the time variable
                    scaled_times = self.times/gamma
                    l_null_acc = []

                    for m in range(self.n_samp):
                        gc.collect() # Ensure memory usage stays tight

                        # Add randomness
                        H_samp, c_op_samp = self.random_sample(H, scaled_c_op,
                                sig, coup, gamma, i, j, k, m)

                        # Always compute Hamiltonian spectrum, because its cheap
                        # and also needed for a dynamic analysis
                        h_evals, h_evecs = H_samp.eigenstates()
                        self.h_eigvals[i, j, k, n_H*m:n_H*(m+1)] = \
                                h_evals

                        # Construct L
                        if self.disorder_type == 'read_L':
                            l_matrix = np.load(self.L_root + '_' + str(i) + '_'
                                    + str(j) + '_' + str(k) + '_' + str(m) +
                                    '.npy')
                            L = qt.Qobj(l_matrix, dims=[[[n_H], [n_H]], [[n_H],
                                [n_H]]], superrep='super')
                        else:
                            L = qt.liouvillian(H_samp, c_op_samp)
                            l_matrix = L.full()

                        # Compute eigendecompositions and SVD
                        if self.spectrum:        
                            # Do an eigendecomposition on the Liouvillian defined by
                            # H and c_op
                            l_evals, l_evecs = sp.linalg.eig(l_matrix)
                            self.l_eigvals[i, j, k, n_L*m:n_L*(m+1)] = \
                                l_evals

                            # Compute SVD, giving steady states and measures of
                            # non-normality
                            try:
                                U, s, Vh = sp.linalg.svd(l_matrix)
                            except sp.linalg.LinAlgError:
                                print('gessd failed, trying gesvd')
                                try:
                                    U, s, Vh = sp.linalg.svd(l_matrix,
                                            lapack_driver='gesvd')
                                except sp.linalg.LinAlgError:
                                    print('gesvd failed, dumping matrix')
                                    np.save('svd_dump.npy', l_matrix)
                                    raise Exception(
                                            'Failed both svd algorithms')

                            self.l_singvals[i, j, k, n_L*m:n_L*(m+1)] = s

                            # Grab null_space through same method used by
                            # sp.linalg.null_space
                            rcond = np.finfo(s.dtype).eps * l_matrix.shape[0]
                            tol = s[0] * rcond
                            num = np.sum(s > tol, dtype=int)
                            NSpace = Vh[num:,:].T.conj()
                            print('null space: ' + str(NSpace.shape))
                            l_null_acc += [NSpace]
                            self.d_null[i, j, k, m] = NSpace.shape[1]

                            # Compute measures of non-normality
                            # See Trefethen and Embree, Spectra and
                            # Pseudospectra, Ch. 48

                            # First, the induced 2-norm condition number of the
                            # eigenvector matrix
                            cond = np.linalg.cond(l_evecs, p=2)
                            print('cond = ' + str(cond))
                            self.cond[i, j, k, m] = cond

                            # Now we compute the departure from normality
                            # measure in the Frobenius norm, which can be
                            # expressed entirely in terms of eigenvalues and
                            # singular values
                            # Force complex to get estimate of negativity,
                            # rather than just a NaN
                            # https://stackoverflow.com/questions/2598734/
                            # numpy-creating-a-complex-array-from-2-real-ones
                            depF = np.sqrt((np.sum(np.square(s)) -
                                    np.sum(np.square(np.abs(l_evals))))
                                    + 0.0J)
                            print('depF = ' + str(depF))
                            self.depF[i, j, k, m] = depF

                        # Compute dynamics
                        if t_steps > 0:
                            # Check assumptions on Hamiltonian spectral gap
                            gap = h_evals[1] - h_evals[0]
                            print('gap = ' + str(gap))
                            if gap <= 1.0e-12:
                                print('Found very small Hamiltonian gap!')
                            self.gaps[i, j, k, m] = gap

                            # Prep observables
                            l_coh = h_evecs[0] * h_evecs[1].dag()
                            r_coh = h_evecs[1] * h_evecs[0].dag()
                            p0 = h_evecs[0] * h_evecs[0].dag()
                            p1 = h_evecs[1] * h_evecs[1].dag()

                            # Run a 1st excited state decay sim
                            result = qt.mesolve(L, h_evecs[1], scaled_times,
                                    e_ops=[p0,p1,l_coh,r_coh])
                            for n, e in enumerate(result.expect):
                                self.expects[i, j, k, m, n, :] = e

                        # Save realizations of H, c_ops, and L
                        match self.save_L:
                            case 'partial' | 'full':
                                self.h_instances[i, j, k, m, :, :] = H_samp.full()
                                for n, c in enumerate(c_op_samp):
                                    self.c_instances[i, j, k, m, n, :, :] = c.full()
                                if self.save_L == 'full':
                                    np.save(self.L_root + '_' + str(i) + '_' +
                                            str(j) + '_' + str(k) + '_' +
                                            str(m) + '.npy', l_matrix)

                    self.l_null += [l_null_acc]

    def make_dynamics_fig(self, fname_root=None):
        """Plots the decay of the first excited state"""
        plt.rcParams['text.usetex'] = True
        plt.rcParams.update({'font.size': 16})

        fig_list = []

        for i, sig in enumerate(self.sigma):
            for j, coup in enumerate(self.J):
                for k, gamma in enumerate(self.rate):
                    fig, ax = plt.subplots()
                    scaled_times = self.times/gamma

                    for m in range(self.n_samp):
                        ax.plot(scaled_times,
                                np.real(self.expects[i, j, k, m, 1, :]))

                    ax = fig.axes[0]
                    ax.set_title('Population Dynamics')
                    ax.set_xlabel(r'$t$')
                    ax.set_ylabel('Excited State Population')

                    if fname_root is not None:
                        fig.savefig(fname_root
                                + '_dynamics_sigma' + '{:.3g}'.format(sig)
                                + '_J' + '{:.3g}'.format(coup)
                                + '_rate' + '{:.3g}'.format(gamma)
                                + '.png')
                        plt.close(fig)
                    else:
                        fig_list += [fig]

        return fig_list

    def make_disorder_fig(self, fname_root=None):
        """Makes eigenvalue plots that sweep over disorder strength"""
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
                label=['$\sigma/J =$ ' + '{:.3g}'.format(x) for x in self.sigma]
                # If this errors, see here
                # https://github.com/numpy/numpy/issues/11879
                # https://github.com/numpy/numpy/issues/10297
                # The easiest fix is to slightly increase sigma
                ax1.hist(self.h_eigvals[:,i,j,:].transpose(), 'auto',
                        density=True, histtype='step', label=label)

                # https://stackoverflow.com/questions/10984085/
                # automatically-rescale-ylim-and-xlim
                ax1.relim()
                ax1.autoscale_view()

                ax1.set_title('Hamiltonian Spectrum')
                ax1.set_xlabel(r'$E$')
                ax1.set_ylabel(r'$\sigma(E)$')
                ax1.legend(loc = 'upper left')

                for l in self.l_eigvals[:,i,j,:]:
                    ax2.scatter(np.real(l), np.imag(l), s=2)
                    ax2.set_title('Liouvillian Spectrum')
                    ax2.set_xlabel(r'Re$(\lambda)$')
                    ax2.set_ylabel(r'Im$(\lambda)$')

                fig.tight_layout()

                if fname_root is not None:
                    fig.savefig(fname_root
                            + '_sweepSigma'
                            + '_J' + '{:.3g}'.format(coup)
                            + '_rate' + '{:.3g}'.format(gamma)
                            + '.png')
                    plt.close(fig)
                else:
                    fig_list += [fig]

        return fig_list


    def make_J_fig(self, fname_root=None):
        """Makes eigenvalue plots that sweep over coupling strengths"""
        plt.rcParams['text.usetex'] = True
        plt.rcParams.update({'font.size': 16})

        fig_list = []

        for i, sig in enumerate(self.sigma):
            for j, gamma in enumerate(self.rate):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize= [6, 9])
                label=['$J=$ ' + '{:.3g}'.format(x) for x in self.J]
                ax1.hist(self.h_eigvals[i,:,j,:].transpose(), 'auto',
                        density=True, histtype='step', label=label)

                ax1.relim()
                ax1.autoscale_view()

                ax1.set_title('Hamiltonian Spectrum')
                ax1.set_xlabel(r'$E$')
                ax1.set_ylabel(r'$\sigma(E)$')
                ax1.legend(loc = 'upper left')

                for l in self.l_eigvals[i,:,j,:]:
                    ax2.scatter(np.real(l), np.imag(l), s=2)
                    ax2.set_title('Liouvillian Spectrum')
                    ax2.set_xlabel(r'Re$(\lambda)$')
                    ax2.set_ylabel(r'Im$(\lambda)$')

                fig.tight_layout()

                if fname_root is not None:
                    fig.savefig(fname_root
                            + '_sweepJ'
                            + '_sigma' + '{:.3g}'.format(sig)
                            + '_rate' + '{:.3g}'.format(gamma)
                            + '.png')
                    plt.close(fig)
                else:
                    fig_list += [fig]

        return fig_list


    def make_rate_fig(self, fname_root=None):
        """Makes eigenvalue plots that sweep over dissipation rates"""
        plt.rcParams['text.usetex'] = True
        plt.rcParams.update({'font.size': 16})

        fig_list = []

        for i, sig in enumerate(self.sigma):
            for j, coup in enumerate(self.J):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize= [6, 9])
                label=['$\gamma/J =$ ' + '{:.3g}'.format(x) for x in self.rate]
                ax1.hist(self.h_eigvals[i,j,:,:].transpose(), 'auto',
                        density=True, histtype='step', label=label)

                ax1.relim()
                ax1.autoscale_view()

                ax1.set_title('Hamiltonian Spectrum')
                ax1.set_xlabel(r'$E$')
                ax1.set_ylabel(r'$\sigma(E)$')
                ax1.legend(loc = 'upper left')

                for l in self.l_eigvals[i,j,:,:]:
                    ax2.scatter(np.real(l), np.imag(l), s=2)
                    ax2.set_title('Liouvillian Spectrum')
                    ax2.set_xlabel(r'Re$(\lambda)$')
                    ax2.set_ylabel(r'Im$(\lambda)$')

                fig.tight_layout()

                if fname_root is not None:
                    fig.savefig(fname_root
                            + '_sweepRate'
                            + '_sigma' + '{:.3g}'.format(sig)
                            + '_J' + '{:.3g}'.format(coup)
                            + '.png')
                    plt.close(fig)
                else:
                    fig_list += [fig]

        return fig_list


    def save_file(self, fname):
        """
        Saves a binary representation of the instance to a .pyz file. Keeps all
        information except rng, spectrum, g_func, disorder_type, save_L, L_root,
        and l_null, which either cannot be saved in array format or are not
        worth saving.
        """
        G_arr = nx.to_numpy_array(self.G)
        c_op_arr = np.array([c.full() for c in np.atleast_1d(self.c_op)])

        np.savez(fname,
                 G=G_arr,
                 c_op=c_op_arr,
                 sigma=self.sigma,
                 J=self.J,
                 rate=self.rate,
                 n_samp=self.n_samp,
                 dynamics=self.dynamics,
                 h_eigvals=self.h_eigvals,
                 l_eigvals=self.l_eigvals,
                 l_singvals=self.l_singvals,
                 d_null=self.d_null,
                 cond=self.cond,
                 depF=self.depF,
                 times=self.times,
                 expects=self.expects,
                 gaps=self.gaps,
                 h_instances=self.h_instances,
                 c_instances=self.c_instances)


def param_sweep_from_file(fname, rng=np.random.default_rng()):
    """
    Load binary representation made by ParameterSweep.save_file(fname). However,
    will need a new rng and will have an empty list of l_null, due to the
    inability to save those out as np.arrays.
    """
    raw = np.load(fname)
    G = nx.from_numpy_array(raw['G'])
    c_op = np.array([qt.Qobj(c) for c in raw['c_op']])
    dr = raw['dynamics']

    return ParameterSweep(G,
                          c_op,
                          sigma=raw['sigma'],
                          J=raw['J'],
                          rate=raw['rate'],
                          n_samp=raw['n_samp'].item(),
                          rng=rng,
                          dynamics=(int(dr[0]),dr[1]),
                          disorder_type='read_H',
                          save_L='none',
                          h_eigvals=raw['h_eigvals'],
                          l_eigvals=raw['l_eigvals'],
                          l_singvals=raw['l_singvals'],
                          d_null=raw['d_null'],
                          cond=raw['cond'],
                          depF=raw['depF'],
                          times=raw['times'],
                          expects=raw['expects'],
                          gaps=raw['gaps'],
                          h_instances=raw['h_instances'],
                          c_instances=raw['c_instances'])
