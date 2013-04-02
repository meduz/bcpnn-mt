import sys
import os
import pylab
import numpy as np
import plot_connectivity_profile as pc
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
    print '\nPlotting the default parameters given in simulation_parameters.py\n'
    import simulation_parameters
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

CP = pc.ConnectionPlotter(params)
gid = CP.find_cell_closest_to_vector((.5, .5), (.2, .0))
CP.plot_connection_histogram(gid, 'ee')


pylab.show()
