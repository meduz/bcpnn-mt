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
        self.shaft_width = 0.003
        fig = pylab.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel('x position')
        self.ax.set_ylabel('y position')
#        self.ax.set_xlim((-0.2, 1.2))
#        self.ax.set_ylim((-0.2, 1.4))
        self.legends = []
        self.quivers = {}
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
        self.quivers[cell_id] = (x0, y0, u0, v0, 'y', self.shaft_width*3)


    def plot_connections(self, tgt_ids, tgt_tp, weights, marker, color, quiver=False):
        """
        """
        markersizes = utils.linear_transformation(weights, 3, self.markersize_max)
        debug = False
        for i_, tgt in enumerate(tgt_ids):
            x_tgt = tgt_tp[tgt, 0] 
            y_tgt = tgt_tp[tgt, 1] 
            w = weights[i_]
            plot = self.ax.plot(x_tgt, y_tgt, marker, c=color, markersize=markersizes[i_])
            if quiver:
                self.quivers[tgt] = (x_tgt, y_tgt, tgt_tp[tgt, 2], tgt_tp[tgt, 3], color, self.shaft_width)
            if debug:
                self.ax.annotate('%d' % tgt, (x_tgt + 0.05, y_tgt + 0.05), fontsize=12)
        return plot


    def plot_ee(self, src_gid, marker='x', color='r'):
        self.load_connection_matrices('ee')
        tgts_ee = self.connection_matrices['ee'][src_gid, :].nonzero()[0]
        weights = self.connection_matrices['ee'][src_gid, tgts_ee]
        tgt_tp = self.tp_exc
        plot = self.plot_connections(tgts_ee, tgt_tp, weights, marker, color, quiver=True)
        self.legends.append((plot[0], 'exc src %d, exc tgts' % src_gid))
        print 'src_gid %d has %d outgoing E->E connection' % (src_gid, len(weights))
        return tgts_ee

    def plot_ei(self, src_gid, marker = 'o', color = 'r'):
        self.load_connection_matrices('ei')
        tgts_ei = self.connection_matrices['ei'][src_gid, :].nonzero()[0]
        weights = self.connection_matrices['ei'][src_gid, tgts_ei]
        tgt_tp = self.tp_inh
        plot = self.plot_connections(tgts_ei, tgt_tp, weights, marker, color, quiver=True)
        self.legends.append((plot[0], 'exc src %d, inh tgts' % src_gid))
        print 'src_gid %d has %d outgoing E->I connection' % (src_gid, len(weights))
        return tgts_ei


    def plot_ie(self, src_gid, marker='x', color='b'):
        self.load_connection_matrices('ie')
        tgts_ie = self.connection_matrices['ie'][src_gid, :].nonzero()[0]
        weights = self.connection_matrices['ie'][src_gid, tgts_ie]
        tgt_tp = self.tp_exc
        plot = self.plot_connections(tgts_ie, tgt_tp, weights, marker, color, quiver=True)
        self.legends.append((plot[0], 'inh src %d, exc tgts' % src_gid))
        print 'src_gid %d has %d outgoing I->E connection' % (src_gid, len(weights))
        return tgts_ie

    def plot_ii(self, src_gid, marker='o', color='b'):
        self.load_connection_matrices('ii')
        tgts_ii = self.connection_matrices['ii'][src_gid, :].nonzero()[0]
        weights = self.connection_matrices['ii'][src_gid, tgts_ii]
        tgt_tp = self.tp_exc
        plot = self.plot_connections(tgts_ii, tgt_tp, weights, marker, color, quiver=True)
        self.legends.append((plot[0], 'inh src %d, inh tgts' % src_gid))
        print 'src_gid %d has %d outgoing I->I connection' % (src_gid, len(weights))
        return tgts_ii



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
        for i in xrange(len(self.legends)):
            plots.append(self.legends[i][0])
            labels.append(self.legends[i][1])
        self.ax.legend(plots, labels)


    def plot_quivers(self):

        data = np.zeros((len(self.quivers.keys()), 4))
        for i_, key in enumerate(self.quivers.keys()):
            (x, y, u, v, c, shaft_width) = self.quivers[key]
            data[i_, :] = np.array([x, y, u, v])
            self.ax.quiver(data[i_, 0], data[i_, 1], data[i_, 2], data[i_, 3], angles='xy', scale_units='xy', scale=1, color=c, headwidth=3, width=shaft_width)
#        self.ax.quiver(data[:, 0], data[:, 1], data[:, 2], data[:, 3], angles='xy', scale_units='xy', scale=1, color=c, headwidth=3)

if __name__ == '__main__':
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

    P = ConnectionPlotter(params)

    np.random.seed(0)
    try:
        gid = int(sys.argv[1])
    except:
#        cell_id = int(.5 * params['n_exc'])
        gid = np.random.randint(0, params['n_exc'], 1)[0]
        print 'plotting GID', gid
        
    os.system("python merge_connlists.py")

    color_list = ['k', 'b', 'g', 'r', 'y', 'c', 'm', '#00f80f', '#deff00', '#ff00e4', '#00ffe6']
    P.plot_ee(gid)
    tgts = P.plot_ei(gid)
    P.plot_cell(gid, exc=True, color='r')

#    exc_gids_to_plot = np.random.randint(0, params['n_exc'], 1)
#    for i_, gid in enumerate(exc_gids_to_plot):
#        c = color_list[i_ % len(color_list)]
#        P.plot_ee(gid, marker='o', color=c)

#    P.plot_cells_as_dots(range(params['n_exc']), P.tp_exc)


    # debug #find tgt inh cells which have
#    import CreateConnections as CC 
#    tp_src = np.loadtxt(params['tuning_prop_means_fn'])
#    tp_tgt = np.loadtxt(params['tuning_prop_inh_fn'])
#    src = gid
#    n_tgt = params['n_inh']
#    p, latency = np.zeros(n_tgt), np.zeros(n_tgt)
#    for tgt in xrange(n_tgt):
#        p[tgt], latency[tgt] = CC.get_p_conn(tp_src[src, :], tp_tgt[tgt, :], params['w_sigma_x'], params['w_sigma_v'])
#    sorted_indices = np.argsort(p)
#    n_tgt_cells_per_neuron = int(round(params['p_ei'] * n_tgt))
#    targets = sorted_indices[-n_tgt_cells_per_neuron:] 
#    for i in xrange(len(targets)):
#        tgt = targets[i]
#        print 'gid, tp_tgt, p', tgt, tp_tgt[tgt, :], p[tgt]
#    print 'tp_src', tp_src[src, :]
#    gid = targets[0]
    gid = tgts[0]
    P.plot_cell(gid, exc=False, color='b')
    P.plot_ie(gid)
    P.plot_ii(gid)


    quiver = True
    if quiver:
        P.plot_quivers()

    P.make_legend()


    pylab.show()

