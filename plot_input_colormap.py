import matplotlib
matplotlib.use('Agg')
import pylab
import PlotPrediction as P
import sys
import NeuroTools.parameters as ntp
import simulation_parameters
import os
import utils
import numpy as np


def merge_input_spiketrains(params):

    all_spikes = np.array([])
    all_gids = np.array([])
    for i in xrange(params['n_exc']):
        fn = params['input_st_fn_base'] + str(i) + '.npy'
        spike_times = np.load(fn)
        all_spikes = np.concatenate((all_spikes, spike_times))
        all_gids = np.concatenate((all_gids, i * np.ones(spike_times.size)))
    
    output_data = np.array((all_spikes, all_gids)).transpose()
    output_fn = params['merged_input_spiketrains_fn']
    print 'output_data', output_data
    print 'Saving merged spike trains to:', output_fn

    np.savetxt(output_fn, output_data)
    return output_data


def plot_input_colormap(params=None, data_fn=None, inh_spikes = None):

    if params== None:
        network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
#        P = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
        params = network_params.params

    if data_fn == None:
        if params.has_key('merged_input_spiketrains_fn'):
            output_fn = params['merged_input_spiketrains_fn']
        else:
            params['merged_input_spiketrains_fn'] = "%sinput_spiketrain_merged.dat" % (params['input_folder'])
            data_fn = params['merged_input_spiketrains_fn']
    if not os.path.exists(data_fn):
        merge_input_spiketrains(params)
    

    plotter = P.PlotPrediction(params, data_fn)
    pylab.rcParams['axes.labelsize'] = 14
    pylab.rcParams['axes.titlesize'] = 16
    if plotter.no_spikes:
        return

    plotter.compute_v_estimates()
#    plotter.compute_position_estimates()
#    plotter.compute_theta_estimates()

    # fig 1
    # neuronal level
    output_fn_base = params['figures_folder'] + 'input_colormap.png'

    plotter.create_fig()  # create an empty figure
    pylab.subplots_adjust(left=0.07, bottom=0.07, right=0.97, top=0.93, wspace=0.3, hspace=.2)
    plotter.n_fig_x = 2
    plotter.n_fig_y = 2
#    plotter.plot_rasterplot('exc', 1)               # 1 
#    plotter.plot_rasterplot('inh', 2)               # 2 
    plotter.plot_vx_grid_vs_time(1)              # 3 
    plotter.plot_vy_grid_vs_time(2)              # 4 
    plotter.plot_x_grid_vs_time(3, ylabel='x-position of stimulus')
    plotter.plot_y_grid_vs_time(4, ylabel='y-position of stimulus')
    output_fn = output_fn_base + '_0.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn)

#    pylab.show()

if __name__ == '__main__':


    if len(sys.argv) > 1:
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.info'
        import NeuroTools.parameters as NTP
        fn_as_url = utils.convert_to_url(param_fn)
        params = NTP.ParameterSet(fn_as_url)
        print 'Loading parameters from', param_fn
        plot_input_colormap(params=params)

    else:
        print '\nPlotting the default parameters give in simulation_parameters.py\n'
        plot_input_colormap()

#folder = 'Data_inputstrength_swepng/NoColumns_winit_random_wsigmaX2.50e-01_wsigmaV2.50e-01_winput2.00e-03_finput2.00e+03pthresh1.0e-01_ptow1.0e-02/' 
#params_fn = folder + 'simulation_parameters.info'
#data_fn = folder + 'Spikes/exc_spikes_merged_.ras'
#inh_spikes = folder + 'Spikes/inh_spikes_.ras'
#tuning_prop_means_fn = folder + 'Parameters/tuning_prop_means.prm'
#output_fn = folder + 'Figures/prediction_0.png'

#params = ntp.ParameterSet(params_fn)
#params['exc_spiketimes_fn_merged'] = data_fn
#params['tuning_prop_means_fn'] = tuning_prop_means_fn

#new_params = { 'folder_name' : folder}
#PS = simulation_parameters.parameter_storage() # load the current parameters
#params = PS.params
#PS.update_values(new_params)
#print 'debug', PS.params['folder_name']
#data_fn = params['exc_spiketimes_fn_merged'] + '.ras'
#print 'data_fn: ', data_fn
#exit(1)

#plot_prediction()#params, data_fn, inh_spikes, output_fn)

