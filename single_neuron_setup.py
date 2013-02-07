import os
import sys
import simulation_parameters
import numpy as np
from pyNN.utility import get_script_args
from pyNN.errors import RecordingError

network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.params

#simulator_name = get_script_args(1)[0]  
simulator_name  = 'nest'
exec("from pyNN.%s import *" % simulator_name)


params['t_sim'] = 1000
setup(timestep=0.1,min_delay=0.1,max_delay=4.0, rng_seeds_seed=0)

ifcell = create(IF_cond_exp, {  'cm' : 1., 'tau_m' : 10, 
                                'i_offset' : 0.0,    'tau_refrac' : 1.0,
                                'v_thresh' : -51.0,  'tau_syn_E'  : 2.0,
                                'v_rest' : -70, 
                                'tau_syn_I': 5.0,    'v_reset'    : -70.0,
                                'e_rev_E'  : 0.,     'e_rev_I'    : -80.})

ifcell2 = create(EIF_cond_exp_isfa_ista, {  
    'cm' : 1., 'tau_m' :      10.,
    'v_rest' : -70.0, 
    'i_offset' : 0.0,    'tau_refrac' : 1.0,
    'v_thresh' : -51.0,  'tau_syn_E'  : 2.0,
    'a' : 4.0, 'b' : 0.1, 'delta_T' : 2., 
    'tau_syn_I': 5.0,    'v_reset'    : -70.0,
    'e_rev_E'  : 0.,     'e_rev_I'    : -80.})

"""
cm          0.281   nF  Capacity of the membrane
tau_refrac  0.0     ms  Duration of refractory period
v_spike     0.0     mV  Spike detection threshold
v_reset     -70.6   mV  Reset value for membrane potential after a spike
v_rest      -70.6   mV  Resting membrane potential (Leak reversal potential)
tau_m       9.3667  ms  Membrane time constant
i_offset    0.0     nA  Offset current
a           4.0     nS  Subthreshold adaptation conductance
b           0.0805  nA  Spike-triggered adaptation
delta_T     2.0     mV  Slope factor
tau_w       144.0   ms  Adaptation time constant
v_thresh    -50.4   mV  Spike initiation threshold
e_rev_E     0.0     mV  Excitatory reversal potential
tau_syn_E   5.0     ms  Decay time constant of excitatory synaptic conductance
e_rev_I     -80.0   mV  Inhibitory reversal potential
tau_syn_I   5.0     ms  Decay time constant of the inhibitory synaptic conductance
"""

def create_spikes(gid = None, random=False):
    if gid != None:
        spike_times = np.load(params['input_st_fn_base'] + str(gid) + '.npy')
    else:
        t_start = 50
        t_stop = .8 * params['t_sim']
        if random:
            n_spikes = (t_stop - t_start) / 1000.  * .3 * params['f_max_stim']
            spike_times = (t_stop - t_start) * np.random.random(n_spikes)
        else:
            spike_times = np.linspace(t_start, t_stop, 10)
    spike_times.sort()
    spike_times = spike_times.tolist()
#    print 'spike_times', spike_times
    return spike_times




def connect_spikes(spike_times, w=None):
    spike_sourceE = create(SpikeSourceArray, {'spike_times': spike_times})
    if w == None:
        w = params['w_input_exc']
    connE = connect(spike_sourceE, ifcell, weight=w, synapse_type='excitatory', delay=1.0)
    connE = connect(spike_sourceE, ifcell2, weight=w, synapse_type='excitatory', delay=1.0)

def connect_noise():
    spike_sourceE = create(SpikeSourcePoisson, {'rate' : params['f_exc_noise']})
    spike_sourceI = create(SpikeSourcePoisson, {'rate' : params['f_inh_noise']})
    connE = connect(spike_sourceE, ifcell, weight=params['w_exc_noise'], synapse_type='excitatory',delay=1.0)
    connI = connect(spike_sourceI, ifcell, weight=params['w_inh_noise'], synapse_type ='inhibitory',delay=1.0)
    connE = connect(spike_sourceE, ifcell2, weight=params['w_exc_noise'], synapse_type='excitatory',delay=1.0)
    connI = connect(spike_sourceI, ifcell2, weight=params['w_inh_noise'], synapse_type ='inhibitory',delay=1.0)


def connect_current():
    amp = 2.
    src = DCSource(amp, 50, .8 * params['t_sim'])
    src.inject_into(ifcell)
    src.inject_into(ifcell2)



try:
    gid = int(sys.argv[1])
except:
#    if os.path.exists(params['gids_to_record_fn']):
#        gid = int(np.loadtxt(params['gids_to_record_fn'])[0])
#    else:
    gid = None

print 'gid', gid

spike_times = create_spikes(gid, random=True)
#connect_spikes(spike_times, w=.014)
connect_current()

    
volt_fn_2 = "SingleNeuronResults/aEIF_cond_exp_isfa_ista.v"
volt_fn_1 = "SingleNeuronResults/IF_cond_exp.v"
gsyn_fn_2 = "SingleNeuronResults/aEIF_cond_exp_isfa_ista.gsyn"
gsyn_fn_1 = "SingleNeuronResults/IF_cond_exp.gsyn"
spikes_fn_2 = "SingleNeuronResults/aEIF_cond_exp_isfa_ista.ras"
spikes_fn_1 = "SingleNeuronResults/IF_cond_exp.ras"

record_v(ifcell, volt_fn_1)
record_gsyn(ifcell, gsyn_fn_1)

record_v(ifcell2, volt_fn_2)
record_gsyn(ifcell2, gsyn_fn_2)

record(ifcell, spikes_fn_1)
record(ifcell2, spikes_fn_2)

run(params['t_sim'])

end()

spikes_1 = np.loadtxt(spikes_fn_1)
n_spikes_1 = spikes_1[:, 0].size
spikes_2 = np.loadtxt(spikes_fn_2)
n_spikes_2 = spikes_2[:, 0].size
print 'nspikes IF_cond_exp:', n_spikes_1
print 'nspikes EIF_cond_exp_isfa_ista:', n_spikes_2

import pylab
fig = pylab.figure()
ax = fig.add_subplot(111)

v1 = np.loadtxt(volt_fn_1)
t_axis = .1 * np.arange(0, v1[:, 0].size)
v2 = np.loadtxt(volt_fn_2)

ax.plot(t_axis, v1[:, 0], label='IF_cond_exp')
ax.plot(t_axis, v2[:, 0], label='AdEx_cond')
ax.legend()
pylab.show()
