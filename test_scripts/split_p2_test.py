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
    
    base_path = 'img/split'

    # reload the simulation and delete the old results
    reloaded = param_sweep_from_file(base_path + '.npz')
    reloaded.disorder_type = 'read_L'
    reloaded.L_root = 'L_data/split'
    reloaded.h_eigvals = np.array([])
    reloaded.l_eigvals = np.array([])
    reloaded.expects = np.array([])
    # Double the dynamical simulation length
    dynamics2 = (2000,800.0)
    reloaded.dynamics = dynamics2
    # Rerun the simulation, using the old samples of H and c_ops
    reloaded.run()
    reloaded.make_disorder_fig(fname_root=base_path+'2')
    reloaded.make_dynamics_fig(fname_root=base_path+'2')
    reloaded.save_file(base_path + '2' + '.npz')
    del reloaded

    print(datetime.datetime.now())
