"""
Simple network with a Poisson spike source projecting to populations of of IF_cond_exp neurons
"""

import time
import numpy as np
import numpy.random as nprnd
import sys
import os
import CreateConnections as CC
import utils
import pylab



class CompareSimulations(object):

    def __init__(self, parameter_storage):

        self.parameter_storage = parameter_storage
        self.params = parameter_storage.params

        assert os.path.exists(self.params['tuning_prop_means_fn']), \
                '\nFile not found: %s\n\nPlease run NetworkSimModuleNoColumns.py or run_all.sh before with NO connectivity!\n' % (self.params['tuning_prop_means_fn'])
        self.tuning_prop_exc = np.loadtxt(self.params['tuning_prop_means_fn'])
        self.tuning_prop_inh = np.loadtxt(self.params['tuning_prop_inh_fn'])

        self.color_dict = {}
        self.input_rate = {}
        self.input_spike_trains = {}

        self.fig = pylab.figure(figsize=(10, 12))
        self.n_fig_y, self.n_fig_x = 5, 1
        self.fig_cnt = 1
        pylab.subplots_adjust(hspace=.45)


    def set_gids_to_plots(self, n_cells=2, cell_type='exc'):
        """
        Choose n_cells which shall be plotted.
        Look at the tuning properties and choose cells that get sequentially activated.
        Assign colors to cells. 
        """
        self.n_cells = n_cells
        if cell_type == 'exc':
            tp = self.tuning_prop_exc
        else:
            tp = self.tuning_prop_inh

        good_gids = np.loadtxt(self.params['gids_to_record_fn'], dtype='int')
        idx_sorted_by_xpos = tp[good_gids, 0].argsort()
        idx_ = np.linspace(0, good_gids.size, n_cells, endpoint=False)
        idx_to_plot = [int(round(i)) for i in idx_]
        print 'idx_to_plot', idx_to_plot
        self.gids_to_plot = good_gids[idx_sorted_by_xpos[idx_to_plot]]
        print 'Plotting GIDS\tx\t\ty\t\tu\tv\t\tdx'
        print 'tp[%d, :] =\t' % self.gids_to_plot[0], tp[self.gids_to_plot[0], :]
        for i_ in xrange(1, len(self.gids_to_plot)):
            gid = self.gids_to_plot[i_]
            print 'tp[%d, :] =\t' % gid, tp[gid, :], tp[gid, 0] - tp[self.gids_to_plot[i_-1], 0]

        # assign a certain color to each cell
        color_list = ['b', 'g', 'r', 'y', 'c', 'm', '#00f80f', '#deff00', '#ff00e4', '#00ffe6', 'k']
        for i_, gid in enumerate(self.gids_to_plot):
            self.color_dict[gid] = color_list[i_]
            self.input_spike_trains[gid] = {}
            self.input_rate[gid] = {}



    def create_input(self):

        nprnd.seed(self.params['input_spikes_seed'])
        dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process 
        self.time = np.arange(0, self.params['t_sim'], dt)
        blank_idx = np.arange(1./dt * self.params['t_stimulus'], 1. / dt * (self.params['t_stimulus'] + self.params['t_blank']))

        L_input = np.zeros((self.n_cells, self.time.shape[0]))
        for i_time, time_ in enumerate(self.time):
            if (i_time % 1000 == 0):
                print "t:", time_
            L_input[:, i_time] = utils.get_input(self.tuning_prop_exc[self.gids_to_plot, :], self.params, time_/self.params['t_stimulus'])
            L_input[:, i_time] *= self.params['f_max_stim']

        L_input_noblank = L_input.copy()
        for i_time in blank_idx:
            L_input[:, i_time] = 0.

        # create input with blank
        for i_, gid in enumerate(self.gids_to_plot):
            rate_of_t = L_input[i_, :]
            n_steps = rate_of_t.size
            spike_times= []
            for i in xrange(n_steps):
                r = nprnd.rand()
                if (r <= ((rate_of_t[i]/1000.) * dt)): # rate is given in Hz -> 1/1000.
                    spike_times.append(i * dt) 
            self.input_spike_trains[gid]['with_blank'] = spike_times
            self.input_rate[gid]['with_blank'] = L_input[i_, :]

        # create input without blank
        for i_, gid in enumerate(self.gids_to_plot):
            rate_of_t = L_input_noblank[i_, :]
            n_steps = rate_of_t.size
            spike_times= []
            for i in xrange(n_steps):
                r = nprnd.rand()
                if (r <= ((rate_of_t[i]/1000.) * dt)): # rate is given in Hz -> 1/1000.
                    spike_times.append(i * dt) 
            self.input_spike_trains[gid]['no_blank'] = spike_times
            self.input_rate[gid]['no_blank'] = L_input_noblank[i_, :]



    def plot_input(self, fill_blank=True):
        """
        This function iterates through self.gids_to_plot,
        and computes how the L_i (rate envelope) and the spike train would
        look like for each cell with the blank and if fill_blank is set, also 
        without the blank.
        """
        self.create_input()

        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, self.fig_cnt)
        self.fig_cnt += 1
        for gid in self.gids_to_plot:
            ax.plot(self.time, self.input_rate[gid]['with_blank'], c=self.color_dict[gid], lw=2)
            ax.plot(self.time, self.input_rate[gid]['no_blank'], c=self.color_dict[gid], lw=2, ls='--')

        ax.set_title('Input with and without blank', fontsize=14)
#        ax.set_xlabel('Time [ms]', fontsize=14)
        ax.set_ylabel('Input rate [Hz]', fontsize=14)
        self.plot_blank(ax, txt='Blank')

#        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, self.fig_cnt)
#        self.fig_cnt += 1
#        ax.set_ylim((0, len(self.gids_to_plot)))
        y_min, y_max = ax.get_ylim()
        ypos = np.linspace(y_max, y_min, len(self.gids_to_plot)+1, endpoint = False)
        dy = ypos[1] - ypos[0]
        for i_, gid in enumerate(self.gids_to_plot):
            spiketimes = self.input_spike_trains[gid]['with_blank']
#            (y_min, y_max) = (i_, i_+ 1)
            self.plot_spikes(ax, spiketimes, gid, ypos[i_]+.5*dy, ypos[i_+1])
        ax.set_title('Input spike trains', fontsize=14)
#        ax.set_xlabel('Time [ms]', fontsize=14)
#        ax.set_ylabel('Cell index', fontsize=14)
        self.plot_blank(ax, txt='Blank')


    def plot_response(self, title):
        """
        Plots the cell's membrane potential and output spikes in response to the
        stimulus alone (that's why you need to simulate without connectivity before),
        and the response with connectivity.
        """
        volt_fn = self.params['exc_volt_fn_base'] + '.v'
        print 'Loading membrane potentials from:', volt_fn
        volt_data = np.loadtxt(volt_fn)
        spike_fn = self.params['exc_spiketimes_fn_merged'] + '%d.ras' % 0
        nspikes, spiketimes = utils.get_nspikes(spike_fn, self.params['n_exc'], get_spiketrains=True)

        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, self.fig_cnt)
        self.fig_cnt += 1
        ax2 = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, self.fig_cnt)
        self.fig_cnt += 1
        y_min, y_max = -70, -45
        for i_, gid in enumerate(self.gids_to_plot):
            time_axis, volt = utils.extract_trace(volt_data, gid)
            ax.plot(time_axis, volt, lw=1, label='nspikes[%d]=%d' % (gid, nspikes[gid]), color=self.color_dict[gid])
            self.plot_spikes(ax, spiketimes[gid], gid, y_min, y_max, lw=2)
            self.plot_spike_histogram(ax2, gid, i_, spiketimes[gid])

        ax.set_ylim((y_min, y_max))
        ax.set_ylabel('Membrane voltage [mV]', fontsize=14)
        ax.set_title(title, fontsize=14)
        self.plot_blank(ax, txt='Blank')


    def plot_spike_histogram(self, ax, gid, shift, spiketimes):

        n_bins = 20

        c = self.color_dict[gid]
        n, bins = np.histogram(spiketimes, bins=n_bins, range=(0, self.params['t_sim']))
        print 'debug, n bins', n_bins
        bin_width = (bins[1] - bins[0]) / len(self.gids_to_plot)
        bins += shift * bin_width
        ax.bar(bins[:-1], n, width=bin_width, facecolor=c)

#        ax.set_xlabel('Time [ms]', fontsize=14)




    def plot_spikes(self, ax, spiketimes, gid, y_min=0, y_max=1, lw=1):
        """
        Plots spiketrain for the given gid into the ax with a certain row idx
        """
        for s in spiketimes:
            ax.plot((s, s), (y_min+0.1, y_max-.1), c=self.color_dict[gid], lw=lw)
#            ax.plot((s, s), (0.2 * (y_max-y_min) + y_min, 0.8 * (y_max-y_min) + y_min), c=self.color_dict[gid])


    def plot_blank(self, ax, c='k', txt=''):

        ylim = ax.get_ylim()
        ax.plot((self.params['t_stimulus'], self.params['t_stimulus']), (ylim[0], ylim[1]), ls='--', c=c, lw=2)
        ax.plot((self.params['t_stimulus'] + self.params['t_blank'], self.params['t_stimulus'] + self.params['t_blank']), (ylim[0], ylim[1]), ls='--', c=c, lw=2)
        if txt != '':
            txt_pos_x = (self.params['t_stimulus'] + .25 * self.params['t_blank'])
            ylim = ax.get_ylim()
            txt_pos_y = ylim[0] + .85 * (ylim[1] - ylim[0])
            print 'DEBUG txt_pos', txt_pos_x, txt_pos_y
            ax.annotate(txt, (txt_pos_x, txt_pos_y), fontsize=14, color='k')



if __name__ == '__main__':

    import simulation_parameters
    ps = simulation_parameters.parameter_storage()
    # set parameters

    ps.params['connectivity_ee'] = False
    ps.params['connectivity_ei'] = False
    ps.params['connectivity_ie'] = False
    ps.params['connectivity_ii'] = False
    ps.set_filenames()

    CS = CompareSimulations(ps)
    CS.set_gids_to_plots(n_cells = 5)
    CS.plot_input(fill_blank=True)
    CS.plot_response('Response to stimulus only (without connectivity)')

    ps.params['connectivity_ee'] = 'anisotropic'
    ps.params['connectivity_ei'] = 'anisotropic'
    ps.params['connectivity_ie'] = 'anisotropic'
    ps.params['connectivity_ii'] = 'anisotropic'
    ps.set_filenames()

    CS.ps = ps
    CS.plot_response('Response to stimulus with connectivity')

    axes = CS.fig.get_axes()
    axes[-1].set_xlabel('Time [ms]', fontsize=14)

#    ps.set_filenames()
#    CS.plot_response()
#    
#    
    pylab.show()
