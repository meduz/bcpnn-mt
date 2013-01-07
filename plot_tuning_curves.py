import os
import pylab
import numpy as np
import utils

# load simulation parameters
import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

input_params = np.loadtxt(params['input_params_fn'])


# these values must be the same as in measure_tuning_curves
n_theta = 3
n_trials = 2  # per angle

n_sim = input_params[:, 0].size * n_trials
assert (n_sim == n_theta * n_trials), "Wrong number of simulations given in file and in %s\nCheck!\n" % (params['input_params_fn'])

v_theta = np.linspace(-.5 * np.pi, .5 * np.pi, n_theta, endpoint=False)

nspikes = np.zeros((n_sim, params['n_exc']))
sim_cnt = 0 
for i_theta, theta in enumerate(v_theta):
    for trial in xrange(n_trials):
        fn_spikes = params['exc_spiketimes_fn_merged'] + '%d.ras' % sim_cnt
        spikes = np.loadtxt(fn_spikes)
        for gid in xrange(params['n_exc']):
            ns = (spikes[:, 1] == gid).nonzero()[0].size
            nspikes[sim_cnt, gid] = ns
        sim_cnt = i_theta * n_trials + trial + 1

fn_out = params['tmp_folder'] + 'tuning_curve.dat' 
print 'Saving to', fn_out
np.savetxt(fn_out, nspikes.transpose())
#    nspikes = utils.get_nspikes

