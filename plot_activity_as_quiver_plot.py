import sys
import pylab
import numpy as np
import simulation_parameters
import matplotlib
from matplotlib import cm
import time
import utils
import rcParams
rcP= rcParams.rcParams
pylab.rcParams.update(rcP)

class ActivityQuiverPlot(object):

    def __init__(self, params):

        self.params = params


    def set_network_activity(self, activity_array):

        self.network_activity = activity_array
        self.gids_to_plot = self.network_activity.nonzero()[0]



    def plot_activity_as_quiver(self, tuning_prop=None):

        if tuning_prop == None:
            tp = np.loadtxt(self.params['tuning_prop_means_fn'])
        else:
            tp = tuning_prop

        scale = 1

        fig = pylab.figure()
        pylab.subplots_adjust(bottom=.15, left=.15)
        ax = fig.add_subplot(111, aspect='equal')

        o_min = self.network_activity.min()
        o_max = self.network_activity.max()
        norm = matplotlib.mpl.colors.Normalize(vmin=o_min, vmax=o_max)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.Greys)#jet)
        m.set_array(np.arange(o_min, o_max, 0.01))
        cb = fig.colorbar(m, ax=ax, shrink=.825)
        cb.set_label('Number of input spikes')
        rgba_colors = []

#        idx_sorted = range(self.gids_to_plot.size)
        idx = self.network_activity[self.gids_to_plot].argsort()
        idx_sorted = self.gids_to_plot[idx]
        idx_sorted = idx_sorted.tolist()
#        idx_sorted.reverse()
#        for i_ in self.gids_to_plot:
#            gid = i_
        for i_ in idx_sorted:
            gid = i_
            activity = self.network_activity[gid]
    #        o_max = max(o_max, activity)
    #        o_min = min(o_min, activity)
    #        print 'max activity cell %d' % cell, activity.max()
            rgba_colors.append(m.to_rgba(activity))
#            print activity, gid

        print 'debug global max', self.network_activity.max(), self.network_activity[self.gids_to_plot].argmax()
        n_cells = self.gids_to_plot.size
        data = np.zeros((n_cells+1, 4), dtype=np.double)
#        data[:n_cells,:] = tp[self.gids_to_plot, :]
        data[:n_cells,:] = tp[idx_sorted, :]
        mp = self.params['motion_params']
        data[-1,:] = mp

        rgba_colors.append('b')
        ax.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], \
                  angles='xy', scale_units='xy', scale=scale, color=rgba_colors, headwidth=4, pivot='middle', width=0.007)
#        ax.annotate('Stimulus', (mp[0]+.1*mp[2], mp[1]+0.1*mp[1]), fontsize=12, color='b')

        ax.set_xlim((-.1, 1.2))
        ax.set_ylim((.1, .9))
#        ax.set_ylim((-.1, 1.2))
        ax.set_xlabel('x-position')
        ax.set_ylabel('y-position')
        ax.set_xticks(np.arange(0, 1.2, .2))
        ax.set_yticks(np.arange(0, 1.2, .2))

        output_fn = self.params['figures_folder'] + 'activity_quiverplot.png'
        print 'Saving to', output_fn
        pylab.savefig(output_fn, dpi=200)
        output_fn = self.params['figures_folder'] + 'activity_quiverplot.pdf'
        print 'Saving to', output_fn
        pylab.savefig(output_fn)
        pylab.show()


if __name__ == '__main__':


    if len(sys.argv) > 1:
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.info'
        import NeuroTools.parameters as NTP
        fn_as_url = utils.convert_to_url(param_fn)
        params = NTP.ParameterSet(fn_as_url)
        print 'Loading parameters from', param_fn
        plot_prediction(params=params)

    else:
        PS = simulation_parameters.parameter_storage()
        params = PS.load_params()

    tp = np.loadtxt(params['tuning_prop_means_fn'])

    AQP = ActivityQuiverPlot(params)
    sim_cnt = 0

    # you can also plot the input but do this before:
    # os.system('python merge_input_spikefiles.py')
    spikes_fn = params['input_folder'] + 'merged_input.dat'

#    spikes_fn = params['exc_spiketimes_fn_merged'] + '.ras'
    print 'Loading ', spikes_fn
    # the spikes_fn should store the raw spike trains with spike times in column 0, and gids in column 1
    nspikes = utils.get_nspikes(spikes_fn, n_cells = params['n_exc'])
    print 'nspikes', nspikes.shape
    AQP.set_network_activity(nspikes)
    AQP.plot_activity_as_quiver(tuning_prop=tp)

