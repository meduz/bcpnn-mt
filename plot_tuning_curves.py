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
n_theta = 36
n_trials = 5  # per angle
#n_theta = 10
#n_trials = 3  # per angle

n_gids_to_plot = 3

n_sim = input_params[:, 0].size * n_trials
assert (n_sim == n_theta * n_trials), "Wrong number of simulations given in file and in %s\nCheck!\n" % (params['input_params_fn'])

v_theta = np.linspace(-.5 * np.pi, .5 * np.pi, n_theta, endpoint=False)
#v_theta = np.linspace(-.25 * np.pi, .25 * np.pi, n_theta, endpoint=False)

nspikes = np.zeros((n_sim, params['n_exc']))
sim_cnt = 0 
for i_theta, theta in enumerate(v_theta):
    for trial in xrange(n_trials):
        fn_spikes = params['exc_spiketimes_fn_merged'] + '%d.ras' % sim_cnt
        spikes = np.loadtxt(fn_spikes)
        if spikes.ndim == 1:
            nspikes[sim_cnt, spikes[1]] = 1
        else:
            for gid in xrange(params['n_exc']):
#                print 'debug', spikes
                ns = (spikes[:, 1] == gid).nonzero()[0].size
                nspikes[sim_cnt, gid] = ns
        sim_cnt = i_theta * n_trials + trial + 1

fn_out = params['tmp_folder'] + 'tuning_curve.dat' 
print 'Saving to', fn_out
np.savetxt(fn_out, nspikes.transpose())


nspikes_mean = np.zeros((n_theta, params['n_exc']))
nspikes_std = np.zeros((n_theta, params['n_exc']))

gids_to_plot = []
# take n_gids_to_plot cells with highest mean output rates for a specific angle
for i_theta, theta in enumerate(v_theta):
    sim_0 = i_theta * n_trials
    sim_1 = (i_theta + 1) * n_trials
    for cell in xrange(params['n_exc']):
        nspikes_mean[i_theta, cell] = nspikes[sim_0:sim_1, cell].mean()
        nspikes_std[i_theta, cell] = nspikes[sim_0:sim_1, cell].std()
    macs = nspikes_mean[i_theta, :].argsort()[-n_gids_to_plot:]
    print theta, 'mac, n', macs, nspikes_mean[i_theta, macs]


nspikes_mean /= params['t_sim'] / 1000.
nspikes_std /= params['t_sim'] / 1000.
fig = pylab.figure()
x_axis = np.degrees(v_theta + np.pi)
ax = fig.add_subplot(111)
#gids_to_plot = [118, 119, 127, 208, 120, 161]
gids_to_plot = [119, 127, 208, 120]
for gid in gids_to_plot:
    ax.errorbar(x_axis, nspikes_mean[:, gid], yerr=nspikes_std[:, gid])

ax.set_title('Tuning curves')
ax.set_xlabel('Angle of motion stimulus')
ax.set_ylabel('Mean output rate [Hz]')

output_fn = params['figures_folder'] + 'tuning_curves.png'
pylab.savefig(output_fn)
output_fn = params['figures_folder'] + 'tuning_curves.eps'
pylab.savefig(output_fn)
pylab.show()
#gids_to_plot = nspikes_mean.argsort()[-n_gids_to_plot:]
#print 'gids with highest mean output rates:', gids_to_plot
#print nspikes_mean[gids_to_plot]


