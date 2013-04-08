"""
This script analyses the 'abstract' input / the rate envolope files for a given blur
"""
import numpy as np
import utils
import pylab
import sys
import os


import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

output_string = '%.4e\t%.4e\t' % (params['blur_X'], params['blur_V'])

blur_x_start = 0.01
blur_x_stop = 0.5
blur_x_step = 0.01
blur_x_range = np.arange(blur_x_start, blur_x_stop, blur_x_step)

blur_v = 0.15
blur_v_range = blur_v * np.ones(blur_x_range.size)

dt = params['dt_rate'] # [ms] time step for the non-homogenous Poisson process 
time = np.arange(0, params['t_stimulus'], dt)

n_cells = params['n_exc']
L_net = np.zeros((blur_x_range.size, n_cells+2))
L_net[:, 0] = blur_x_range
L_net[:, 1] = blur_v_range


try:
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size
    print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
except:
    USE_MPI = False
    pc_id, n_proc, comm = 0, 1, None
    print "MPI not used"

#my_units = utils.distribute_n(params['n_exc'], n_proc, pc_id)

for j_, blur_x in enumerate(blur_x_range):
    L_input = np.zeros((n_cells, time.shape[0]))
    params['blur_X'], params['blur_V'] = blur_x, blur_v
    print 'Blur', params['blur_X'], params['blur_V']
    tuning_prop = utils.set_tuning_prop(params, mode='hexgrid', v_max=params['v_max'])        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
    for i_time, time_ in enumerate(time):
        if (i_time % 100 == 0):
            print "t:", time_
        L_input[:, i_time] = utils.get_input(tuning_prop, params, time_/params['t_sim'])
#        L_input[:, i_time] = utils.get_input(tuning_prop[my_units, :], params, time_/params['t_sim'])
#        L_input[:, i_time] *= params['f_max_stim']
    for cell in xrange(n_cells):
        L_net[j_, cell+2] = L_input[cell, :].sum()

L_net_output_fn = 'L_net.dat'
np.savetxt(L_net_output_fn, L_net)

