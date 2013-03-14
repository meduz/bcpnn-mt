"""
This script plots the input spike trains in the top panel
and the output rasterplots in the middle and lower panel
"""
import sys
import pylab
import numpy as np
import re
import utils


if len(sys.argv) > 1:
    param_fn = sys.argv[1]
    if os.path.isdir(param_fn):
        param_fn += '/Parameters/simulation_parameters.info'
    import NeuroTools.parameters as NTP
    fn_as_url = utils.convert_to_url(param_fn)
    print 'Loading parameters from', param_fn
    params = NTP.ParameterSet(fn_as_url)

else:
    print '\nPlotting the default parameters give in simulation_parameters.py\n'
    import simulation_parameters
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary


def plot_input_spikes(ax, shift=0, m='o', c='k'):
    """
    Shift could be used when plotting in the same axis as the output spikes
    """
    n_cells = params['n_exc']
    for cell in xrange(n_cells):
        fn = params['input_st_fn_base'] + str(cell) + '.npy'
        spiketimes = np.load(fn)
        nspikes = len(spiketimes)
        ax.plot(spiketimes, cell * np.ones(nspikes) + shift, m, color=c, markersize=2)


def plot_spikes(ax, fn, n_cells):
    nspikes, spiketimes = utils.get_nspikes(fn, n_cells, get_spiketrains=True)
    for cell in xrange(int(len(spiketimes))):
        ax.plot(spiketimes[cell], cell * np.ones(nspikes[cell]), 'o', color='k', markersize=2)



fn_exc = params['exc_spiketimes_fn_merged'] + '.ras'
fn_inh = params['inh_spiketimes_fn_merged'] + '.ras'

# ax1 is if input spikes shall be plotted in a seperate axis  (from the output spikes)
fig = pylab.figure()
#ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(211)
ax3 = fig.add_subplot(212)
    
#ax1.set_xlabel('Time [ms]')
#ax1.set_ylabel('Target GID')
#ax1.set_title('Input spikes')
ax2.set_xlabel('Time [ms]')
ax3.set_xlabel('Time [ms]')


ax2.set_ylabel('Exc ID')
ax3.set_ylabel('Inh ID')

pylab.rcParams['lines.markeredgewidth'] = 0

plot_input_spikes(ax2, shift=.5, m='o', c='r')
plot_spikes(ax2, fn_exc, params['n_exc'])
plot_spikes(ax3, fn_inh, params['n_inh'])
ax2.set_ylim((0, params['n_exc'] + 1))
ax3.set_ylim((0, params['n_inh'] + 1))

#output_fn = params['rasterplot_%s_fig' % cell_type] 
#print "Saving to", output_fn
#pylab.savefig(output_fn)

pylab.show()

