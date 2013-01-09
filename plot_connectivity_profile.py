import numpy as np
import utils
import pylab
import sys
import os
import simulation_parameters

class ConnectionPlotter(object):

    def __init__(self, params):
        self.params = params
        self.load_connection_matrices()

        self.tp_exc = np.loadtxt(params['tuning_prop_means_fn'])
        self.tp_inh = np.loadtxt(params['inh_cell_pos_fn'])

#        self.lw_max = 10 # maximum line width for connection strengths
        self.markersize = 3
        self.markersize_max = 10
        fig = pylab.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel('x position')
        self.ax.set_ylabel('y position')
        self.ax.set_xlim((0, 1))
        self.ax.set_ylim((0, 1))


    def load_connection_matrices(self):
        self.conn_mat_ee_fn = self.params['conn_mat_fn_base'] + 'ee.dat'
        self.conn_mat_ei_fn = self.params['conn_mat_fn_base'] + 'ei.dat'
        self.conn_mat_ie_fn = self.params['conn_mat_fn_base'] + 'ie.dat'
        self.conn_mat_ii_fn = self.params['conn_mat_fn_base'] + 'ii.dat'
        self.delay_mat_ee_fn = self.params['delay_mat_fn_base'] + 'ee.dat'
        self.delay_mat_ei_fn = self.params['delay_mat_fn_base'] + 'ei.dat'
        self.delay_mat_ie_fn = self.params['delay_mat_fn_base'] + 'ie.dat'
        self.delay_mat_ii_fn = self.params['delay_mat_fn_base'] + 'ii.dat'

        # E - E 
        if os.path.exists(self.conn_mat_ee_fn):
            print 'Loading', self.conn_mat_ee_fn
            self.conn_mat_ee = np.loadtxt(self.conn_mat_ee_fn)
        #    delays_ee = np.loadtxt(delay_mat_ee_fn)
        else:
            self.conn_mat_ee, delays_ee = utils.convert_connlist_to_matrix(params['merged_conn_list_ee'], params['n_exc'])
            np.savetxt(self.conn_mat_ee_fn, self.conn_mat_ee)
        #    np.savetxt(delay_mat_ee_fn, delay_mat_ee)

        # E - I 
        if os.path.exists(self.conn_mat_ei_fn):
            print 'Loading', self.conn_mat_ei_fn
            self.conn_mat_ei = np.loadtxt(self.conn_mat_ei_fn)
        #    delays_ei = np.loadtxt(delay_mat_ei_fn)
        else:
            self.conn_mat_ei, delays_ei = utils.convert_connlist_to_matrix(params['merged_conn_list_ei'], params['n_exc'])
            np.savetxt(self.conn_mat_ei_fn, self.conn_mat_ei)
        #    np.savetxt(delay_mat_ei_fn, delay_mat_ei)

        # I - E
        if os.path.exists(self.conn_mat_ie_fn):
            print 'Loading', self.conn_mat_ie_fn
            self.conn_mat_ie = np.loadtxt(self.conn_mat_ie_fn)
        #    delays_ie = np.loadtxt(delay_mat_ie_fn)
        else:
            self.conn_mat_ie, delays_ie = utils.convert_connlist_to_matrix(params['merged_conn_list_ie'], params['n_exc'])
            np.savetxt(self.conn_mat_ie_fn, self.conn_mat_ie)
        #    np.savetxt(delay_mat_ie_fn, delay_mat_ie)

        # I - I
        if os.path.exists(self.conn_mat_ii_fn):
            print 'Loading', self.conn_mat_ii_fn
            self.conn_mat_ii = np.loadtxt(self.conn_mat_ii_fn)
        #    delays_ii = np.loadtxt(delay_mat_ii_fn)
        else:
            self.conn_mat_ii, delays_ii = utils.convert_connlist_to_matrix(params['merged_conn_list_ii'], params['n_exc'])
            np.savetxt(self.conn_mat_ii_fn, self.conn_mat_ii)
        #    np.savetxt(delay_mat_ie_fn, delay_mat_ie)


    def plot_cell(self, cell_id, exc=True):
        """
        markers = {0: 'tickleft', 1: 'tickright', 2: 'tickup', 3: 'tickdown', 4: 'caretleft', 'D': 'diamond', 
                    6: 'caretup', 7: 'caretdown', 's': 'square', '|': 'vline', '': 'nothing', 'None': 'nothing', 'x': 'x', 
                   5: 'caretright', '_': 'hline', '^': 'triangle_up', ' ': 'nothing', 'd': 'thin_diamond', None: 'nothing', 
                   'h': 'hexagon1', '+': 'plus', '*': 'star', ',': 'pixel', 'o': 'circle', '.': 'point', '1': 'tri_down', 
                   'p': 'pentagon', '3': 'tri_left', '2': 'tri_up', '4': 'tri_right', 'H': 'hexagon2', 'v': 'triangle_down', 
                   '8': 'octagon', '<': 'triangle_left', '>': 'triangle_right'}
        """
        marker = '^'
        if exc:
            color = 'r'
            tp = self.tp_exc
        else:
            color = 'b'
            tp = self.tp_inh

        x0, y0, u0, v0 = tp[cell_id, 0], tp[cell_id, 1], tp[cell_id, 2], tp[cell_id, 3]
        self.ax.plot(x0, y0, marker, c=color, markersize=self.markersize)

#        direction = self.ax.plot((x0, x0+u0), (y0, (y0+v0)), 'yD-.', lw=1)
#        self.ax.legend((direction[0]), ('predicted direction'))
        self.ax.quiver(x0, y0, u0, v0, angles='xy', scale_units='xy', scale=1, color='y', headwidth=6)

#        self.ax.legend((target_cell_exc[0], source_plot_ee[0], source_cell_exc[0], direction[0], target_plot_ei[0], source_plot_ie[0]), \
#                ('exc target cell', 'incoming connections from exc', 'exc source cell', 'predicted direction', 'outgoing connections to inh', 'incoming connections from inh'))


    def plot_connections(self, tgt_ids, tgt_tp, weights, marker, color, with_annotations=False):
        """
        """
        markersizes = utils.linear_transformation(weights, 1, self.markersize_max)
        for i_, tgt in enumerate(tgt_ids):
            x_tgt = tgt_tp[tgt, 0] 
            y_tgt = tgt_tp[tgt, 1] 
            w = weights[i_]
            self.ax.plot(x_tgt, y_tgt, marker, c=color, markersize=markersizes[i_])
#            line_width = lws[i_]
#            target_cell_exc = self.ax.plot(x_tgt, y_tgt, 'bo', lw=line_width)
        #    target_plot_ee = self.ax.plot((x0, x_tgt), (y0, y_tgt), 'b--', lw=line_width)

            if with_annotations:
                self.ax.annotate('(%d, %.2e, %.2e)' % (tgt, w, d), (x_tgt, y_tgt), fontsize=8)


    def plot_ee(self, src_gid):
        marker = 'x'
        color = 'r'
        tgts_ee = self.conn_mat_ee[src_gid, :].nonzero()[0]
        weights = self.conn_mat_ee[src_gid, tgts_ee]
        tgt_tp = self.tp_exc
        self.plot_connections(tgts_ee, tgt_tp, weights, marker, color)

    def plot_ei(self, src_gid):
        marker = 'o'
        color = 'r'
        tgts_ee = self.conn_mat_ei[src_gid, :].nonzero()[0]
        weights = self.conn_mat_ei[src_gid, tgts_ee]
        tgt_tp = self.tp_inh
        self.plot_connections(tgts_ee, tgt_tp, weights, marker, color)



if __name__ == '__main__':
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

    P = ConnectionPlotter(params)

    try:
        cell_id = int(sys.argv[1])
    except:
#        cell_id = int(.5 * params['n_exc'])
        cell_id = 650
    P.plot_cell(cell_id, exc=True)
    P.plot_ee(cell_id)
    P.plot_ei(cell_id)

    cell_id = 651
    P.plot_cell(cell_id, exc=True)
    P.plot_ee(cell_id)
    P.plot_ei(cell_id)


    pylab.show()


"""
srcs_ee = conn_mat_ee[:, exc_cell].nonzero()[0]
weights_ee = conn_mat_ee[exc_cell, tgts_ee]

print "Plotting exc_cell -> E"
print "Plotting E -> exc_cell"
lws = utils.linear_transformation(conn_mat_ee[srcs_ee, exc_cell], 1, lw_max)
for i_, src in enumerate(srcs_ee):
    x_src = tp_exc[src, 0] 
    y_src = tp_exc[src, 1] 
    w = conn_mat_ee[src, exc_cell]
#    d = delays_ee[src, exc_cell]
    line_width = lws[i_]
    source_cell_exc = self.ax.plot(x_src, y_src, 'b^', lw=line_width)
    source_plot_ee = self.ax.plot((x_src, x0), (y_src, y0), 'b:', lw=line_width)

print "Plotting exc_cell -> I"
tgts_ei = conn_mat_ei[exc_cell, :].nonzero()[0]
#lws = utils.linear_transformation(conn_mat_ei[tgts_ei, exc_cell], 1, lw_max)
lws = [2 for i in xrange(len(tgts_ei))]
for i_, tgt in enumerate(tgts_ei):
    x_tgt = tp_inh[tgt, 0] 
    y_tgt = tp_inh[tgt, 1] 
    w = conn_mat_ei[tgt, exc_cell]
#    d = delays_ei[tgt, exc_cell]
    line_width = lws[i_]
    target_plot_ei = self.ax.plot(x_tgt, y_tgt, 'ro', lw=line_width)

print "Plotting I -> exc_cell"
srcs_ie = conn_mat_ie[:, exc_cell].nonzero()[0]
#lws = utils.linear_transformation(conn_mat_ie[srcs_ie, exc_cell], 1, lw_max)
lws = [2 for i in xrange(len(srcs_ie))]
for i_, src in enumerate(srcs_ie):
    x_src = tp_inh[src, 0] 
    y_src = tp_inh[src, 1] 
    self.ax.plot(x_src, y_src, 'o', c='r', markersize=1)
    w = conn_mat_ie[src, exc_cell]
#    d = delays_ie[src, exc_cell]
    line_width = lws[i_]
    source_plot_ie = self.ax.plot(x_src, y_src, 'r^', lw=line_width)
    source_plot_ie = self.ax.plot((x_src, x0), (y_src, y0), 'r:', lw=line_width)


title = 'Connectivity profile of cell %d\ntp:' % (exc_cell) + str(tp_exc[exc_cell, :])
title += '\nw_sigma_x=%.2f w_sigma_v=%.2f' % (params['w_sigma_x'], params['w_sigma_v'])
self.ax.set_title(title)

fig_fn = params['figures_folder'] + 'precomp_conn_profile_%d.png' % exc_cell
print "Saving fig to", fig_fn
pylab.savefig(fig_fn)
pylab.show()

"""
