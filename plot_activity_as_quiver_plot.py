import sys
import pylab
import numpy as np
import simulation_parameters
import matplotlib
from matplotlib import cm
import time
import utils


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
        ax = fig.add_subplot(111)

        o_min = self.network_activity.min()
        o_max = self.network_activity.max()
        norm = matplotlib.mpl.colors.Normalize(vmin=o_min, vmax=o_max)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.Greys)#jet)
        m.set_array(np.arange(o_min, o_max, 0.01))
        fig.colorbar(m)
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
            print activity, gid

        print 'debug global max', self.network_activity.max(), self.network_activity[self.gids_to_plot].argmax()
        n_cells = self.gids_to_plot.size
        data = np.zeros((n_cells+1, 4), dtype=np.double)
#        data[:n_cells,:] = tp[self.gids_to_plot, :]
        data[:n_cells,:] = tp[idx_sorted, :]
        mp = self.params['motion_params']
        data[-1,:] = mp

        rgba_colors.append('r')
        ax.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], \
                  angles='xy', scale_units='xy', scale=scale, color=rgba_colors, headwidth=4, pivot='middle')
        ax.annotate('Stimulus', (mp[0]+.1*mp[2], mp[1]+0.1*mp[1]), fontsize=12, color='r')

        ax.set_xlim((-0.2, 1.2))
        ax.set_ylim((-0.2, 1.2))
        output_fn = self.params['figures_folder'] + 'activity_quiverplot.png'
        pylab.savefig(output_fn)
        pylab.show()


if __name__ == '__main__':


    if len(sys.argv) > 1:
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.info'
        import NeuroTools.parameters as NTP
        fn_as_url = utils.convert_to_url(param_fn)
        print 'debug ', fn_as_url
        params = NTP.ParameterSet(fn_as_url)
        print 'Loading parameters from', param_fn
        plot_prediction(params=params)

    else:
        PS = simulation_parameters.parameter_storage()
        params = PS.load_params()

    tp = np.loadtxt(params['tuning_prop_means_fn'])

    AQP = ActivityQuiverPlot(params)
    sim_cnt = 0
    spikes_fn = params['exc_spiketimes_fn_merged'] + '.ras'
    print 'Loading ', spikes_fn
    nspikes = utils.get_nspikes(spikes_fn, n_cells = params['n_exc'])
    AQP.set_network_activity(nspikes)
    AQP.plot_activity_as_quiver(tuning_prop=tp)

