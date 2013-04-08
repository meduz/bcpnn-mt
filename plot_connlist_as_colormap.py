import matplotlib
matplotlib.use('Agg')
import pylab
import numpy as np
import sys
import os
import utils
import simulation_parameters

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
sim_cnt = 0

conn_type = sys.argv[1]
if conn_type == 'ee':
    n_src, n_tgt = params['n_exc'], params['n_exc']
elif conn_type == 'ei':
    n_src, n_tgt = params['n_exc'], params['n_inh']
elif conn_type == 'ie':
    n_src, n_tgt = params['n_inh'], params['n_exc']
elif conn_type == 'ii':
    n_src, n_tgt = params['n_inh'], params['n_inh']

conn_list_fn = params['merged_conn_list_%s' % conn_type]
print 'Loading: ', conn_list_fn

conn_mat_fn = params['conn_mat_fn_base'] + '%s.dat' % (conn_type)
#delay_mat_fn = params['delay_mat_fn_base'] + '%s.dat' % (conn_type)
if os.path.exists(conn_mat_fn):
    print 'Loading', conn_mat_fn
    w = np.loadtxt(conn_mat_fn)
#    delays_ee = np.loadtxt(delay_mat_ee_fn)
else:
    w, delays = utils.convert_connlist_to_matrix(params['merged_conn_list_%s' % conn_type], n_src, n_tgt)
    print 'Saving:', conn_mat_fn
    np.savetxt(conn_mat_fn, w)

print 'Weights: min %.2e median %.2e max %.2e mean %.2e st %.2e' % (w.min(), w.max(), np.median(w), w.mean(), w.std())
fig = pylab.figure()
ax = fig.add_subplot(111)
print "plotting ...."
title = 'Connection matrix %s \nw_sigma_x(v): %.1f (%.1f)' % (conn_type, params['w_sigma_x'], params['w_sigma_v'])
ax.set_title(title)
cax = ax.pcolormesh(w)#, edgecolor='k', linewidths='1')
ax.set_ylim(0, w.shape[0])
ax.set_xlim(0, w.shape[1])
ax.set_ylabel('Target')
ax.set_xlabel('Source')
pylab.colorbar(cax)


#max_incoming_weights = np.zeros(params['n_exc'])
#for i in xrange(params['n_exc']):
#    sorted_idx = w[:, i].argsort()
#    print 'max weights', w[:, i].max(), w[sorted_idx[-6:], i]
#    print 'sorted idx', w[:, i].argmax(), sorted_idx[-3:]
#    max_incoming_weights[i] = w[:, i].max()
#    print w[:, i].max()

#count, bins = np.histogram(max_incoming_weights, bins=20)
#bin_width = bins[1] - bins[0]
#ax = fig.add_subplot(212)
#ax.bar(bins[:-1], count, width=bin_width)

output_fig = params['figures_folder'] + conn_list_fn.rsplit('/')[-1].rsplit('.dat')[0] + '.png'
print 'Saving fig to', output_fig
pylab.savefig(output_fig)
#pylab.show()


