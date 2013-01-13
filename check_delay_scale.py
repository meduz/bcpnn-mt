"""
This script requires data from an example run to be present in the standard output folder.
Thus, do a run_all.sh with a small network before.
"""


import numpy as np
import pylab
import utils
import sys
import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary

tp_exc = np.loadtxt(params['tuning_prop_means_fn'])
mp = params['motion_params']
conn_list_fn = 'SmallSpikingModel_CC_delayScale20_blurX1.00e-01_blurV1.00e-01_wsigmax1.00e-01_wsigmav1.00e-01/Connections/merged_conn_list_ee.dat'
conn_list = np.loadtxt(conn_list_fn)  #params['merged_conn_list_ee'])

# choose this numbers from the file gids_to_record_fn created by prepare_tuning_prop.py
gid_pre = 51
gid_post = 85
dx = np.sqrt((tp_exc[gid_pre, 0] - tp_exc[gid_post, 0])**2 + (tp_exc[gid_pre, 1] - tp_exc[gid_post, 1])**2)

n_src = 1 # from how many cells should the post-synaptic cell receive input from?
src_gids = utils.get_sources(conn_list, gid_post)


# load file pre
rate_pre = np.load(params['input_rate_fn_base'] + str(gid_pre) + '.npy')
rate_pre /= rate_pre.max()
input_spikes_pre = np.load(params['input_st_fn_base'] + '%d.npy' % gid_pre)

# load file post
rate_post = np.load(params['input_rate_fn_base'] + str(gid_post) + '.npy')
rate_post /= rate_post.max()
input_spikes_post = np.load(params['input_st_fn_base'] + '%d.npy' % gid_post)

# load output spikes from the network simulation (only response to stimulus)
spikes_fn = params['exc_spiketimes_fn_merged'] + '0.ras'
nspikes, spiketrains = utils.get_nspikes(spikes_fn, params['n_exc'], get_spiketrains=True)

# only response to stimulus
output_st_pre = spiketrains[gid_pre]
output_st_post = spiketrains[gid_post] 
t_max_input_pre = utils.get_time_of_max_stim(tp_exc[gid_pre, :], mp) * params['t_stimulus']
t_max_input_post = utils.get_time_of_max_stim(tp_exc[gid_post, :], mp) * params['t_stimulus']

print 't_max_input_pre', t_max_input_pre
print 't_max_input_post', t_max_input_post
print ' diff = ', t_max_input_post - t_max_input_pre


t_max_response_pre, t_max_response_pre_std = utils.get_time_of_max_response(output_st_pre, range=(0, params['t_sim']), n_binsizes=20)
t_max_response_post, t_max_response_post_std = utils.get_time_of_max_response(output_st_post, range=(0, params['t_sim']), n_binsizes=20)
print 't_max_response_pre', t_max_response_pre
print 't_max_response_post', t_max_response_post



# SETUP etc
(delay_min, delay_max) = params['delay_range']
w_ij = 0.015 # weight pre --> post
#conn_delay = 40
conn_delay = min((t_max_input_post - t_max_input_pre), delay_max)
print 'conn_delay = ', conn_delay, ' dx(cells) = ', dx

from pyNN.nest import *
setup(timestep=0.1, min_delay=delay_min, max_delay=delay_max)

# Simulate two post-synaptic cells with different input:
# 1) post synaptic cell with input only from the pre-synaptic cell (+noise)
# 2) post synaptic cell with input from the stimulus + input from pre-synaptic cell (+noise)
n_exc = 5
exc_pop = Population(n_exc, IF_cond_exp, params['cell_params_exc'], label='exc_cells')


# 0) simulate post syn cell with input only
tgt = 0
ssa = create(SpikeSourceArray, {'spike_times': input_spikes_post})
connect(ssa, exc_pop[tgt], params['w_input_exc'], synapse_type='excitatory')

# 1)
#     post synaptic cell with input only from the pre-synaptic cell (+noise)
# connect pre-synaptic cell --> cell 2
tgt = 1
#for src in xrange(n_src):
#    src_gid = sources[src]
#    output_st_pre = spiketrains[gid_pre]
ssa = create(SpikeSourceArray, {'spike_times': output_st_pre})
connect(ssa, exc_pop[tgt], w_ij, synapse_type='excitatory', delay=conn_delay)

# 1) + noise
tgt = 2
ssa = create(SpikeSourceArray, {'spike_times': output_st_pre})
connect(ssa, exc_pop[tgt], w_ij, synapse_type='excitatory', delay=conn_delay)
# + noise
noise_exc = create(native_cell_type('poisson_generator'), {'rate' : params['f_exc_noise']})
noise_inh = create(SpikeSourcePoisson, {'rate' : params['f_inh_noise']})
connect(noise_exc, exc_pop[tgt], weight=params['w_exc_noise'], synapse_type='excitatory', delay=1.)
connect(noise_inh, exc_pop[tgt], weight=params['w_inh_noise'], synapse_type='inhibitory', delay=1.)

# 2)
#     post synaptic cell with input from the stimulus + input from pre-synaptic cell (+noise)
tgt = 3
# connect pre-synaptic cell --> cell 2
ssa = create(SpikeSourceArray, {'spike_times': output_st_pre})
connect(ssa, exc_pop[tgt], w_ij, synapse_type='excitatory', delay=conn_delay)
# connect stimulus --> cell 2
ssa = create(SpikeSourceArray, {'spike_times': input_spikes_post})
connect(ssa, exc_pop[tgt], params['w_input_exc'], synapse_type='excitatory')

# 2) + noise
tgt = 4
ssa = create(SpikeSourceArray, {'spike_times': output_st_pre})
connect(ssa, exc_pop[tgt], w_ij, synapse_type='excitatory')
connect(ssa, exc_pop[tgt], w_ij, synapse_type='excitatory', delay=conn_delay)
# connect stimulus --> cell 2 + noise
ssa = create(SpikeSourceArray, {'spike_times': input_spikes_post})
noise_exc = create(native_cell_type('poisson_generator'), {'rate' : params['f_exc_noise']})
noise_inh = create(SpikeSourcePoisson, {'rate' : params['f_inh_noise']})
connect(noise_exc, exc_pop[tgt], weight=params['w_exc_noise'], synapse_type='excitatory', delay=1.)
connect(noise_inh, exc_pop[tgt], weight=params['w_inh_noise'], synapse_type='inhibitory', delay=1.)


exc_pop.record()
exc_pop.record_v()
run(params['t_sim'])

folder_name = 'CheckDelayScale/'
spikes_sim_fn = folder_name + 'spikes_wij%.1e.ras' % w_ij
volt_sim_fn = folder_name + 'volt_wij%.1e.v' % w_ij
print 'Printing spikes to', spikes_sim_fn
print 'Printing volt to', volt_sim_fn
exc_pop.printSpikes(spikes_sim_fn)
exc_pop.print_v(volt_sim_fn, compatible_output=False)


#pre_trace = utils.convert_spiketrain_to_trace(output_st_pre, params['t_sim'] + 1) # + 1 is to handle spikes in the last time step
#lp_trace_pre = utils.low_pass_filter(pre_trace, tau=25, spike_height=3.)
#print 't_max output_trace_pre', lp_trace_pre.argmax()



#"""
y_min, y_max = 0., 1.
# input spike train and L_i(t)
fig = pylab.figure(figsize=(16, 11))
pylab.subplots_adjust(hspace=0.65)
pylab.rcParams['legend.fontsize'] = 10


ax = fig.add_subplot(331)
for s in input_spikes_pre:
    ax.plot((s, s), (0.2 * (y_max-y_min) + y_min, 0.8 * (y_max-y_min) + y_min), c='k')
n_steps = int(round(1. / params['dt_rate']))
rate_pre = rate_pre[::n_steps] # ::10 because dt for rate creation was 0.1 ms
ax.plot(np.arange(rate_pre.size), rate_pre, lw=2, c='b')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Normalized motion energy')
ax.set_ylim((y_min, y_max))
ax.set_title('Input spikes into pre-synaptic cell')

# output spike train pre
ax = fig.add_subplot(334) 
for s in output_st_pre:
    ax.plot((s, s), (0.2 * (y_max-y_min) + y_min, 0.8 * (y_max-y_min) + y_min), c='k')
ax.set_ylim((y_min, y_max))
ax.set_xlim((0, params['t_sim']))
ax.set_title('Output spikes of pre_synaptic cell')

# output spike histogram pre
ax = fig.add_subplot(337)
n, bins = np.histogram(output_st_pre, bins=20, range=(0, params['t_sim']))
ax.bar(bins[:-1], n, width = bins[1] - bins[0], label='t_max_response=%d ms' % t_max_response_pre)
ax.legend()
ax.set_xlim((0, params['t_sim']))
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Binned output spike train')


# input spikes (stimulus) into post synaptic cell
ax = fig.add_subplot(332)
for s in input_spikes_post:
    ax.plot((s, s), (0.2 * (y_max-y_min) + y_min, 0.8 * (y_max-y_min) + y_min), c='k')
n_steps = int(round(1. / params['dt_rate']))
rate_post = rate_post[::n_steps] # ::10 because dt for rate creation was 0.1 ms
ax.plot(np.arange(rate_post.size), rate_post, lw=2, c='b')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Normalized motion energy')
ax.set_ylim((y_min, y_max))
ax.set_title('WITHOUT pre->post connection\nInput spikes into post-synaptic cell')


# voltage traces
volt = np.loadtxt(volt_sim_fn)
t_axis, v1= utils.extract_trace(volt, 0)
t_axis, v2 = utils.extract_trace(volt, 3)
ax = fig.add_subplot(335)
ax.plot(t_axis, v1, lw=2, label='input=only stimulus')
ax.plot(t_axis, v2, lw=2, label='input=stim + rec')
ax.legend()

ax = fig.add_subplot(338)
ax.plot(t_axis, v2 - v1, lw=2, label='diff: stim+rec - stim_only')
ax.legend()


# output spike train post
#ax = fig.add_subplot(335) 
#for s in output_st_post:
#    ax.plot((s, s), (0.2 * (y_max-y_min) + y_min, 0.8 * (y_max-y_min) + y_min), c='k')
#ax.set_ylim((y_min, y_max))
#ax.set_xlim((0, params['t_sim']))
#ax.set_title('WITHOUT pre->post connection\nOutput spikes of post synaptic cell')

#"""


#n_post, bins_post = np.histogram(output_st_post, bins=n_bins, range=(0, params['t_sim']))

#t_axis = np.arange(0, params['t_sim'] + 1, 1)
#print 't_axis.size', t_axis.size, 'lp_trace_pre', lp_trace_pre.size


#ax = fig.add_subplot(325)
#ax.plot(t_axis, lp_trace_pre)

#ax.bar(bins_post[:-1], n_post, width = bins_post[1] - bins_post[0])
#ax.set_xlim((0, params['t_sim']))
pylab.show()
