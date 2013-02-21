import numpy as np
import utils
import pylab
import sys
import re
import os
import simulation_parameters

class ConnectivityAnalyser(object):

    def __init__(self, params=None):
        if params == None:
            network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
            params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
        else:
            self.params = params
        print 'Merging connlists ...'
        os.system('python merge_connlists.py')

    def plot_tuning_vs_conn_cg(self, conn_type, show=False):
        """
        For each source cell, loop through all target connections and compute the 
        scalar (dot) product between the preferred direction of the source cell and the center of gravity of the connection vector
        (both in the spatial domain and the direction domain)
        c_x_i = sum_j w_ij * (x_i - x_j) # x_ are position vectors of the cell
        c_v_i = sum_j w_ij * (v_i - v_j) # v_ are preferred directions
        """
        (n_src, n_tgt, tp_src, tp_tgt) = utils.resolve_src_tgt_with_tp(conn_type, self.params)
        conn_list = np.loadtxt(self.params['merged_conn_list_%s' % conn_type])

        conn_mat_fn = self.params['conn_mat_fn_base'] + '%s.dat' % (conn_type)
        if os.path.exists(conn_mat_fn):
            print 'Loading', conn_mat_fn
            w = np.loadtxt(conn_mat_fn)
        else:
            w, delays = utils.convert_connlist_to_matrix(params['merged_conn_list_%s' % conn_type], n_src, n_tgt)
            print 'Saving:', conn_mat_fn
            np.savetxt(conn_mat_fn, w)

        cx_ = np.zeros(n_src)
        cv_ = np.zeros(n_src)
        for i in xrange(n_src):
            src_gid = i
            targets = utils.get_targets(conn_list, src_gid)
            targets = np.array(targets[:, 1], dtype=np.int)
            weights = w[src_gid, targets]
            c_x, c_v = self.get_cg_vec(tp_src[src_gid, :], tp_tgt[targets, :], weights)

            (x_src, y_src, vx_src, vy_src) = tp_src[src_gid, :]
            cx_[i] = np.abs(np.dot(c_x, (vx_src, vy_src)))
            cv_[i] = np.abs(np.dot(c_v, (vx_src, vy_src)))

        cx_mean = cx_.mean()
        cx_sem = cx_.std() / np.sqrt(cx_.size)
        cv_mean = cv_.mean()
        cv_sem = cv_.std() / np.sqrt(cv_.size)

        output_fn = self.params['data_folder'] + 'scalar_products_between_tuning_prop_and_cgxv.dat'
        output_data = np.array((cx_, cv_)).transpose()
        print 'Saving to:', output_fn
        np.savetxt(output_fn, output_data)

        fig = pylab.figure(figsize=(12, 10))
        pylab.subplots_adjust(hspace=0.35)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        x = range(n_src)
        ax1.set_xlabel('source cell')
        ax1.set_ylabel('$|\\vec{v}_i \cdot \\vec{c}_i^X|$')
        title = '$\langle|\\vec{v}_i \cdot \\vec{c}_i^X| \\rangle = %.2e \pm %.1e$' % (cx_mean, cx_sem)
        ax1.bar(x, cx_)
        ax1.set_title('Scalar product between preferred direction $\\vec{v}_i$ and CG $\\vec{c}_i^x$\n%s' % title)
#        ax1.legend()
               
        ax2.bar(x, cv_)
        ax2.set_xlabel('source cell')
        ax1.set_ylabel('$|\\vec{v}_i \cdot \\vec{c}_i^V|$')
        title = '$\langle|\\vec{v}_i \cdot \\vec{c}_i^V| \\rangle = %.2e \pm %.1e$' % (cv_mean, cv_sem)
        ax2.set_title(title)
#        ax2.legend()
        output_fig = self.params['figures_folder'] + 'scalar_products_between_tuning_prop_and_cgxv.png'
        print 'Saving to:', output_fig
        pylab.savefig(output_fig)
        if show:
            pylab.show()



    def get_cg_vec(self, tp_src, tp_tgt, weights):
        """
        Computes the center of gravity connection vector in the spatial and direction domain
        c_x_i = sum_j w_ij * (x_i - x_j) # x_ are position vectors of the cell
        c_v_i = sum_j w_ij * (v_i - v_j) # v_ are preferred directions

        tp_src = 4-tuple of the source's tuning properties
        tp_tgt = 4 x n_tgt array with all the target's tuning properties
        """

        c_x = np.zeros(2)
        c_v = np.zeros(2)
        (x_src, y_src, vx_src, vy_src) = tp_src

        n_tgt = tp_tgt[:, 0].size
        for tgt in xrange(n_tgt):
            (x_tgt, y_tgt, vx_tgt, vy_tgt) = tp_tgt[tgt, :]
            c_x += weights[tgt] * np.array(x_src - x_tgt, y_src - y_tgt)
            c_v += weights[tgt] * np.array(vx_src - vx_tgt, vy_src - vy_tgt)

        return c_x, c_v
#        n_tgt = 


"""
if (len(sys.argv) < 2):
    conn_type = 'ee'
else:
    conn_type = sys.argv[1]

fn = params['merged_conn_list_%s' % conn_type] 
if not os.path.exists(fn):
    os.system('python merge_connlists.py')

conn_list = np.loadtxt(fn)
w = conn_list[:, 2]

#M, delays = utils.convert_connlist_to_matrix(fn, params['n_exc'])
#w_in = np.zeros(params['n_exc'])
#w_out = np.zeros(params['n_exc'])
#for i in xrange(params['n_exc']):
#    w_in[i] = M[:, i].sum()
#    idx = M[i, :].nonzero()[0]
#    w_out[i] =  M[i, idx].mean()
#print 'w_out_mean_all', w_out.mean()

n_bins = 100
n, bins = np.histogram(w, bins=n_bins)

w_min, w_max, w_mean, w_std, w_median = w.min(), w.max(), w.mean(), w.std(), np.median(w)
label_txt = 'w_ee min, max = %.2e %.2e\nmean, std = %.2e, %.2e\nmedian=%.2e   w_sum=%.2e\nmax count=%d for w[%d]=(%.1e-%.1e)' % (w_min, w_max, w_mean, w_std, w_median, w.sum(), np.max(n), np.argmax(n), bins[np.argmax(n)], bins[np.argmax(n)+1])
print 'Info:', label_txt

fig = pylab.figure()
ax1 = fig.add_subplot(111)
bar = ax1.bar(bins[:-1], n, width=bins[1]-bins[0])


#x = np.arange(n)

#ax1.set_xlabel('Cells sorted by num input spikes')
#ax1.set_ylabel('Number of input spikes')
#ax1.set_xlim((params['w_min'], params['w_max']* 1.02))
#ax1.set_xlim((w_min, w_max * 1.02))
#ax1.set_ylim((0, 17000))

title = 'Distribution of all weights\nInput parameters:\n w_sigma_x(v)=%.1e (%.1e)\nn_exc=%d n_inh=%d' % (params['w_sigma_x'], params['w_sigma_v'], params['n_exc'], params['n_inh'])
ax1.set_title(title)
pylab.subplots_adjust(top=0.8)

(text_pos_x, text_pos_y) = ax1.get_xlim()[1] * 0.45, ax1.get_ylim()[1] * 0.75
pylab.text(text_pos_x, text_pos_y, label_txt, bbox=dict(pad=5.0, ec="k", fc="none"))


#output_fig = 'Figures_WsigmaSweepV/' + 'fig_wsigmaXV%.1e_%.1e.png' % (params['w_sigma_x'], params['w_sigma_v'])
#print 'Saving to:', output_fig
#pylab.savefig(output_fig)

#output_fig = 'Figures_WsigmaSweep_TransformedBound/%d.png' % (file_count)
output_fig = '%sconnection_hist.png' % (params['figures_folder'])
print 'Saving to:', output_fig
pylab.savefig(output_fig)
#pylab.show()
"""


if __name__ == '__main__':

    conn_types = ['ee', 'ei', 'ie', 'ii']
    print len(sys.argv)
        
    try:
        param_fn = sys.argv[1]
        print 'Trying to load parameters from', param_fn, 
        import NeuroTools.parameters as NTP
        params = NTP.ParameterSet(param_fn)
        print '\n succesfull!\n'
    except:
        print '\n NOT successfull\nLoading the parameters currently in simulation_parameters.py\n'
        network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
        params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

    # get the connection type either from sys.argv[1] or [2]
    try:
        conn_type = sys.argv[1]
        assert (conn_type in conn_types), 'Non-existant conn_type %s' % conn_type
    except:
        try:
            conn_type = sys.argv[2]
            assert (conn_type in conn_types), 'Non-existant conn_type %s' % conn_type
        except:
            conn_type = 'ee'

    print 'Processing conn_type', conn_type
    CA = ConnectivityAnalyser(params)
    CA.plot_tuning_vs_conn_cg(conn_type, show=False)



