import sys
import os
import utils
import numpy as np

if len(sys.argv) > 1:
    param_fn = sys.argv[1]
    if os.path.isdir(param_fn):
        param_fn += '/Parameters/simulation_parameters.info'
    import NeuroTools.parameters as NTP
    fn_as_url = utils.convert_to_url(param_fn)
    print 'Loading parameters from', param_fn
    params = NTP.ParameterSet(fn_as_url)

else:
    print '\nPlotting the default parameters given in simulation_parameters.py\n'
    import simulation_parameters
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

script_names = ['prepare_tuning_prop.py', 'merge_connlists.py', 'analyse_simple.py', 'merge_connlists.py', \
'get_conductance_matrix.py', 'plot_spike_histogram.py exc', 'plot_spike_histogram.py inh', 'plot_connectivity_profile.py']

for conn_type in params['conn_types']:
    os.system('python plot_weight_and_delay_histogram.py %s' % conn_type)

#for gid in np.loadtxt(params['gids_to_record_fn'])[:3]:
#    os.system('python plot_input.py %d' % gid)

for sn in script_names:
    os.system('python %s' % sn)



#plot_connlist_as_colormap.py 'ee'
#plot_connlist_as_colormap.py 'ei'
#plot_connlist_as_colormap.py 'ie'
#plot_connlist_as_colormap.py 'ii'
