"""
This script runs one simulation:
    cells only receive input from stimulus, no recurrent connectivity
"""

import sys
import numpy as np
import utils
import NeuroTools.parameters as ntp
from pyNN.nest import *
simulator_name = 'nest'
#from pyNN.brian import *
#simulator_name = 'brian'
import simulation_parameters
ps = simulation_parameters.parameter_storage()
params = ps.params


# ===================================
#    G E T   P A R A M E T E R S 
# ===================================
x0, y0 = params['motion_params'][0:2]
sim_cnt = int(sys.argv[1])
mp = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])

from pyNN.utility import Timer
timer = Timer()
timer.start()
times = {} # stores time stamps
tuning_prop = utils.set_tuning_prop(params, mode='hexgrid')
time = np.arange(0, params['t_stimulus'], params['dt_rate'])

#print 'Prepare spike trains'
#L_input = np.zeros((params['n_exc'], time.shape[0]))
#for i_time, time_ in enumerate(time):
#    if (i_time % 100 == 0):
#        print "t:", time_
#    L_input[:, i_time] = utils.get_input(tuning_prop, params, time_/params['t_sim'])
#    L_input[:, i_time] *= params['f_max_stim']

cells_closest_to_stim_pos, x_dist_to_stim = utils.sort_gids_by_distance_to_stimulus(tuning_prop, mp)
print 'debug cells_closest_to_stim_pos', cells_closest_to_stim_pos


# ===============
#    S E T U P 
# ===============
(delay_min, delay_max) = params['delay_range']
setup(timestep=0.1, min_delay=delay_min, max_delay=delay_max, rng_seeds_seed=sim_cnt)
times['t_setup'] = timer.diff()
exc_pop = Population(params['n_exc'], IF_cond_exp, params['cell_params_exc'], label='exc_cells')
times['t_create'] = timer.diff()

rng_v = NumpyRNG(seed = sim_cnt*3147 + params['seed'])
v_init_dist = RandomDistribution('normal',
        (params['v_init'], params['v_init_sigma']),
        rng=rng_v,
        constrain='redraw',
        boundaries=(-80, -60))
exc_pop.initialize('v', v_init_dist)

# ==================================
#    C O N N E C T    I N P U T 
# ==================================
for tgt in xrange(params['n_exc']):
    try:
        fn = params['input_st_fn_base'] + str(tgt) + '.npy'
        spike_times = np.load(fn)
    except: # this cell does not get any input
        print "Missing file: ", fn
        spike_times = []
    ssa = create(SpikeSourceArray, {'spike_times': spike_times})
    connect(ssa, exc_pop[tgt], params['w_input_exc'], synapse_type='excitatory')

    # connect noise
    if (simulator_name == 'nest'): # for nest one can use the optimized Poisson generator
        noise_exc = create(native_cell_type('poisson_generator'), {'rate' : params['f_exc_noise']})
    else:
        noise_exc = create(SpikeSourcePoisson, {'rate' : params['f_exc_noise']})
    connect(noise_exc, exc_pop[tgt], weight=params['w_exc_noise'], synapse_type='excitatory', delay=1.)

# ==================
#    R E C O R D 
# ==================
gids_to_record = cells_closest_to_stim_pos[:5]
exc_pop_view = PopulationView(exc_pop, gids_to_record, label='good_exc_neurons')
exc_pop_view.record_v()
exc_pop.record()

# ==========
#    R U N 
# ==========
run(params['t_sim'])
times['t_sim'] = timer.diff()

fn_volt = params['exc_volt_fn_base'] + '%d.v' % sim_cnt
print 'print_v to file: %s' % (fn_volt)
exc_pop_view.print_v("%s" % (fn_volt), compatible_output=False)
times['t_print_v'] = timer.diff()

fn_spikes = params['exc_spiketimes_fn_merged'] + '%d.ras' % sim_cnt
print "Printing excitatory spikes", fn_spikes
exc_pop.printSpikes(fn_spikes)
times['t_print_spikes'] = timer.diff()

# ==========
#    E N D 
# ==========
end()

times['t_end'] = timer.diff()
for k in times.keys():
    print '%s\ttook\t%f\tseconds' % (k, times[k])
times['t_all'] = 0.
for k in times.keys():
    times['t_all'] += times[k]

times = ntp.ParameterSet(times)
times.save('times_dict.py')

print 'Total time: %d sec' % (times['t_all'])
