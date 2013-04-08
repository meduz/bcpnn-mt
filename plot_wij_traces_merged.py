import sys
import numpy as np
import pylab
import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.params

pre_id = 85
post_id = 161
#cells_to_record = [85, 161, 71, 339]

n_speeds = 1
n_cycles = 5
n_stim = 8
n_iterations = n_stim * n_cycles * n_speeds
n_iterations = 5

def merge_traces(fn_base):
    iteration = 0
    fn_trace = fn_base + '%d_%d_%d.dat' % (iteration, pre_id, post_id)
    trace = np.loadtxt(fn_trace)
    n_steps = trace.size
    t_axis = np.arange(n_steps * n_iterations) * params['dt_rate']
    merged_traces = np.zeros(n_iterations * n_steps)
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.plot(t_axis, merged_traces)

    for iteration in xrange(n_iterations):
        idx_0 = iteration * n_steps
        idx_1 = (iteration + 1) * n_steps
        fn_trace = fn_base + '%d_%d_%d.dat' % (iteration, pre_id, post_id)
        print 'Loading', fn_trace
        trace = np.loadtxt(fn_trace)
        print 'trace', trace.max()
        merged_traces[idx_0:idx_1] = trace
    y_max = merged_traces.max()
    y_min = merged_traces.min()

#    scale = 100.
#    for iteration in xrange(n_iterations):
#        idx_0 = iteration * n_steps
#        idx_1 = (iteration + 1) * n_steps
#        training_input_folder = "%sTrainingInput_%d/" % (params['folder_name'], iteration)
#        mp = np.loadtxt(training_input_folder + 'input_params.txt')
#        ax.quiver((idx_0+idx_1)/2., y_max*1.1, mp[2]*scale, mp[3]*scale, color='k', \
#                units='x', angles='xy', scale_units='xy', headwidth=3.)

    stim_dur = params['t_sim'] 
    cycle_dur = params['t_sim'] * n_stim
    full_dur = cycle_dur * n_speeds
    t0 = 0.
    for cycle in xrange(n_cycles):
        for stim in xrange(n_stim):
            t1 = t0 + params['t_sim']
            t0 += params['t_sim']
            ax.plot((t0, t1), (y_min, y_max), ls='--', c='k')
        ax.plot((t0, t1), (y_min, y_max), ls='-', lw=2, c='k')
    return ax
#    ax.plot(((cycle+1) * cycle_dur , (cycle+1) * cycle_dur), (y_min, y_max), ls='-', c='k')

ax = merge_traces(params['bcpnntrace_folder'] + 'wij_')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Abstract weight')
ax.set_title('Weight %d - %d' % (pre_id, post_id))

ax = merge_traces(params['bcpnntrace_folder'] + 'pij_')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('pij')
ax.set_title('pij %d - %d' % (pre_id, post_id))


pylab.show()
