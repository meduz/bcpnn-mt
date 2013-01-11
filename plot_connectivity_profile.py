import numpy as np
import utils
import pylab
import sys
import os
import simulation_parameters

class ConnectionPlotter(object):

    def __init__(self, params):
        self.params = params

        self.tp_exc = np.loadtxt(params['tuning_prop_means_fn'])
        self.tp_inh = np.loadtxt(params['tuning_prop_inh_fn'])
        self.connection_matrices = {}
        self.delays = {}

#        self.lw_max = 10 # maximum line width for connection strengths
        self.markersize = 10
        self.markersize_max = 10
        fig = pylab.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel('x position')
        self.ax.set_ylabel('y position')
        self.ax.set_xlim((-0.2, 1.2))
        self.ax.set_ylim((-0.2, 1.4))
        self.legends = {}
        self.conn_mat_loaded = [False, False, False, False]

    def load_connection_matrices(self, conn_type):

        if conn_type == 'ee':
            n_src, n_tgt = self.params['n_exc'], self.params['n_exc']
            loaded = self.conn_mat_loaded[0]
        elif conn_type == 'ei':
            n_src, n_tgt = self.params['n_exc'], self.params['n_inh']
            loaded = self.conn_mat_loaded[1]
        elif conn_type == 'ie':
            n_src, n_tgt = self.params['n_inh'], self.params['n_exc']
            loaded = self.conn_mat_loaded[2]
        elif conn_type == 'ii':
            n_src, n_tgt = self.params['n_inh'], self.params['n_inh']
            loaded = self.conn_mat_loaded[3]

        if loaded:
            return

        conn_mat_fn = self.params['conn_mat_fn_base'] + '%s.dat' % (conn_type)
        delay_mat_fn = self.params['delay_mat_fn_base'] + '%s.dat' % (conn_type)
        if os.path.exists(conn_mat_fn):
            print 'Loading', conn_mat_fn
            self.connection_matrices[conn_type] = np.loadtxt(conn_mat_fn)
        #    delays_ee = np.loadtxt(delay_mat_ee_fn)
        else:
            self.connection_matrices[conn_type], self.delays[conn_type] = utils.convert_connlist_to_matrix(params['merged_conn_list_%s' % conn_type], n_src, n_tgt)
            np.savetxt(conn_mat_fn, self.connection_matrices[conn_type])
#            np.savetxt(delay_mat_fn, self.delays[conn_type])
            
        if conn_type == 'ee':
            self.conn_mat_loaded[0] = True
        elif conn_type == 'ei':
            self.conn_mat_loaded[1] = True
        elif conn_type == 'ie':
            self.conn_mat_loaded[2] = True
        elif conn_type == 'ii':
            self.conn_mat_loaded[3] = True


    def load_connection_matrices_delme(self):
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
            self.conn_mat_ee, delays_ee = utils.convert_connlist_to_matrix(params['merged_conn_list_ee'], params['n_exc'], params['n_exc'])
            np.savetxt(self.conn_mat_ee_fn, self.conn_mat_ee)
        #    np.savetxt(delay_mat_ee_fn, delay_mat_ee)

        # E - I 
        if os.path.exists(self.conn_mat_ei_fn):
            print 'Loading', self.conn_mat_ei_fn
            self.conn_mat_ei = np.loadtxt(self.conn_mat_ei_fn)
        #    delays_ei = np.loadtxt(delay_mat_ei_fn)
        else:
            self.conn_mat_ei, delays_ei = utils.convert_connlist_to_matrix(params['merged_conn_list_ei'], params['n_exc'], params['n_inh'])
            np.savetxt(self.conn_mat_ei_fn, self.conn_mat_ei)
        #    np.savetxt(delay_mat_ei_fn, delay_mat_ei)

        # I - E
        if os.path.exists(self.conn_mat_ie_fn):
            print 'Loading', self.conn_mat_ie_fn
            self.conn_mat_ie = np.loadtxt(self.conn_mat_ie_fn)
        #    delays_ie = np.loadtxt(delay_mat_ie_fn)
        else:
            self.conn_mat_ie, delays_ie = utils.convert_connlist_to_matrix(params['merged_conn_list_ie'], params['n_inh'], params['n_exc'])
            np.savetxt(self.conn_mat_ie_fn, self.conn_mat_ie)
        #    np.savetxt(delay_mat_ie_fn, delay_mat_ie)

        # I - I
        if os.path.exists(self.conn_mat_ii_fn):
            print 'Loading', self.conn_mat_ii_fn
            self.conn_mat_ii = np.loadtxt(self.conn_mat_ii_fn)
        #    delays_ii = np.loadtxt(delay_mat_ii_fn)
        else:
            self.conn_mat_ii, delays_ii = utils.convert_connlist_to_matrix(params['merged_conn_list_ii'], params['n_inh'], params['n_inh'])
            np.savetxt(self.conn_mat_ii_fn, self.conn_mat_ii)
        #    np.savetxt(delay_mat_ie_fn, delay_mat_ie)


    def plot_cell(self, cell_id, exc=True, color='r'):
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
#            color = 'r'
            tp = self.tp_exc
        else:
#            color = 'b'
            tp = self.tp_inh

        x0, y0, u0, v0 = tp[cell_id, 0], tp[cell_id, 1], tp[cell_id, 2], tp[cell_id, 3]
        self.ax.plot(x0, y0, marker, c=color, markersize=self.markersize)

#        direction = self.ax.plot((x0, x0+u0), (y0, (y0+v0)), 'yD-.', lw=1)
#        self.ax.legend((direction[0]), ('predicted direction'))
        self.ax.quiver(x0, y0, u0, v0, angles='xy', scale_units='xy', scale=1, color='y', headwidth=4)

#        self.ax.legend((target_cell_exc[0], source_plot_ee[0], source_cell_exc[0], direction[0], target_plot_ei[0], source_plot_ie[0]), \
#                ('exc target cell', 'incoming connections from exc', 'exc source cell', 'predicted direction', 'outgoing connections to inh', 'incoming connections from inh'))


    def plot_connections(self, tgt_ids, tgt_tp, weights, marker, color, with_annotations=False):
        """
        """
        markersizes = utils.linear_transformation(weights, 3, self.markersize_max)
        for i_, tgt in enumerate(tgt_ids):
            x_tgt = tgt_tp[tgt, 0] 
            y_tgt = tgt_tp[tgt, 1] 
            w = weights[i_]
            plot = self.ax.plot(x_tgt, y_tgt, marker, c=color, markersize=markersizes[i_])
#            line_width = lws[i_]
#            target_cell_exc = self.ax.plot(x_tgt, y_tgt, 'bo', lw=line_width)
        #    target_plot_ee = self.ax.plot((x0, x_tgt), (y0, y_tgt), 'b--', lw=line_width)

            if with_annotations:
                self.ax.annotate('(%d, %.2e, %.2e)' % (tgt, w, d), (x_tgt, y_tgt), fontsize=8)
        
        return plot


    def plot_ee(self, src_gid, marker='x', color='r'):
        self.load_connection_matrices('ee')
        tgts_ee = self.connection_matrices['ee'][src_gid, :].nonzero()[0]
        weights = self.connection_matrices['ee'][src_gid, tgts_ee]
        tgt_tp = self.tp_exc
        plot = self.plot_connections(tgts_ee, tgt_tp, weights, marker, color)
        self.legends[src_gid] = (plot[0], 'exc_src=%d' % src_gid)
        print 'src_gid %d has %d outgoing E->E connection' % (src_gid, len(weights))

    def plot_ei(self, src_gid, marker = 'o', color = 'r'):
        self.load_connection_matrices('ei')
        tgts_ei = self.connection_matrices['ei'][src_gid, :].nonzero()[0]
        weights = self.connection_matrices['ei'][src_gid, tgts_ei]
        tgt_tp = self.tp_inh
        self.plot_connections(tgts_ei, tgt_tp, weights, marker, color)


    def plot_cells_as_dots(self, gids, tp):

        marker = 'o'
        ms = 1
        color = 'k'
        for i in xrange(len(gids)):
            gid = gids[i]
            x, y = tp[gid, 0], tp[gid, 1]
            self.ax.plot(x, y, marker, markersize=ms, c=color)


    def make_legend(self):

        plots = []
        labels = []
        for gid in self.legends.keys():
            plots.append(self.legends[gid][0])
            labels.append(self.legends[gid][1])
        self.ax.legend(plots, labels)


if __name__ == '__main__':
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

    P = ConnectionPlotter(params)

    try:
        cell_id = int(sys.argv[1])
    except:
#        cell_id = int(.5 * params['n_exc'])
        cell_id = 630


    os.system("python merge_connlists.py")

    color_list = ['k', 'b', 'g', 'r', 'y', 'c', 'm', '#00f80f', '#deff00', '#ff00e4', '#00ffe6']

    np.random.seed(0)
    exc_gids_to_plot = np.random.randint(0, params['n_exc'], 4)
#    exc_gids_to_plot = [20, 279, 684, 500, 100, 800, 1000]
    for i_, gid in enumerate(exc_gids_to_plot):
        c = color_list[i_ % len(color_list)]
        P.plot_cell(gid, exc=True, color=c)
        P.plot_ee(gid, marker='o', color=c)

#    cell_id = 379
#    P.plot_cell(cell_id, exc=True)
#    P.plot_ee(cell_id, color='r')

#    cell_id = 279
#    P.plot_cell(cell_id, exc=True)
#    P.plot_ee(cell_id, color='b')

    P.make_legend()
#    P.plot_cells_as_dots(range(params['n_exc']), P.tp_exc)

#    P.plot_ei(cell_id)

#    cell_id = 651
#    P.plot_cell(cell_id, exc=True)
#    P.plot_ee(cell_id)
#    P.plot_ei(cell_id)


    pylab.show()

