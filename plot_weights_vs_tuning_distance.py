import numpy as np
import pylab
import os

def get_distance_matrices():
    print 'Computing distance matrices ...'
    # distance matrices based on different vectors
    x_distance_matrix = np.zeros((n_cells, n_cells)) # only spatial tuning
    v_distance_matrix = np.zeros((n_cells, n_cells)) # only directional tuning
    tp_distance_matrix = np.zeros((n_cells, n_cells)) # 4-dim tuning vector

    print 'cell: ', 
    for i in xrange(n_cells):
        print 'i ', i
        for j in xrange(n_cells-i):
            x_distance_matrix[i, j] = np.sqrt((tp[i, 0] - tp[j, 0])**2 + (tp[i, 1] - tp[j, 1])**2)
            v_distance_matrix[i, j] = np.sqrt((tp[i, 2] - tp[j, 2])**2 + (tp[i, 3] - tp[j, 3])**2)
            tp_distance_matrix[i, j] = np.sqrt((tp[i, 0] - tp[j, 0])**2 + (tp[i, 1] - tp[j, 1])**2 + (tp[i, 2] - tp[j, 2])**2 + (tp[i, 3] - tp[j, 3])**2)

            x_distance_matrix[j, i] = x_distance_matrix[i, j]
            v_distance_matrix[j, i] = v_distance_matrix[i, j]
            tp_distance_matrix[j, i] = tp_distance_matrix[i, j]
    np.savetxt(params['x_distance_matrix_fn'], x_distance_matrix)
    np.savetxt(params['v_distance_matrix_fn'], v_distance_matrix)
    np.savetxt(params['tp_distance_matrix_fn'], tp_distance_matrix)
    return x_distance_matrix, v_distance_matrix, tp_distance_matrix


import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary
tp = np.loadtxt(params['tuning_prop_means_fn'])

n_bins_x = 20
n_cells = params['n_exc']

if not os.path.exists(params['x_distance_matrix_fn']):
    x_distance_matrix, v_distance_matrix, tp_distance_matrix = get_distance_matrices()
else:
    print 'Loading distance matrices ...'
    x_distance_matrix = np.loadtxt(params['x_distance_matrix_fn'])
    v_distance_matrix = np.loadtxt(params['v_distance_matrix_fn'])
    tp_distance_matrix = np.loadtxt(params['tp_distance_matrix_fn'])

weight_matrix_fn = 'Anders_data/wijs_72.txt'
weight_matrix = np.loadtxt(weight_matrix_fn)

fig = pylab.figure()
pylab.subplots_adjust(hspace=0.35)
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
print 'Plotting ...'
#for i in xrange(n_cells):
#src_cells = np.argsort(weight_matrix)[:-10]
#src_cells = np.random.randint(0, n_cells, 50)
for src in xrange(n_cells):
    idx,  = weight_matrix[src, :].nonzero()
    if (idx.size > 0):
        for tgt in idx:
#    for j in xrange(n_cells-i):
            y = weight_matrix[src, tgt]
#            if y != 0:
            x1 = x_distance_matrix[src, tgt]
            ax1.plot(x1, y, '.', c='k')

            x2 = v_distance_matrix[src, tgt]
            ax2.plot(x2, y, '.', c='k')

            x3 = tp_distance_matrix[src, tgt]
            ax3.plot(x3, y, '.', c='k')

ax1.set_ylabel('wij')
ax2.set_ylabel('wij')
ax3.set_ylabel('wij')
ax1.set_xlabel('x-dist')
ax2.set_xlabel('v-dist')
ax3.set_xlabel('tuning prop-dist (4-dim)')


#x_dist_min, x_dist_max, x_dist_mean, x_dist_std = x_distance_matrix.min(), x_distance_matrix.max(), x_distance_matrix.mean(), x_distance_matrix.std()
#print 'x_dist: min %.3e\tmax %.3e\tmean %.3e\t+- %.2e' % (x_dist_min, x_dist_max, x_dist_mean, x_dist_std)
#v_dist_min, v_dist_max, v_dist_mean, v_dist_std = v_distance_matrix.min(), v_distance_matrix.max(), v_distance_matrix.mean(), v_distance_matrix.std()
#print 'v_dist: min %.3e\tmax %.3e\tmean %.3e\t+- %.2e' % (v_dist_min, v_dist_max, v_dist_mean, v_dist_std)
#tp_dist_min, tp_dist_max, tp_dist_mean, tp_dist_std = tp_distance_matrix.min(), tp_distance_matrix.max(), tp_distance_matrix.mean(), tp_distance_matrix.std()
#print 'tp_dist: min %.3e\tmax %.3e\tmean %.3e\t+- %.2e' % (tp_dist_min, tp_dist_max, tp_dist_mean, tp_dist_std)

#x_dist_hist = np.zeros(n_bins_x)
#v_dist_hist = np.zeros(n_bins_x)
#tp_dist_hist = np.zeros(n_bins_x)
#for src in xrange(n_cells):
#    cnt, bins_x = np.histogram(x_distance_matrix[src, :], bins=n_bins_x, range=(x_dist_min, x_dist_max))
#    x_dist_hist += cnt
#    cnt, bins_v = np.histogram(v_distance_matrix[src, :], bins=n_bins_x, range=(v_dist_min, v_dist_max))
#    v_dist_hist += cnt
#    cnt, bins_tp = np.histogram(tp_distance_matrix[src, :], bins=n_bins_x, range=(tp_dist_min, tp_dist_max))
#    tp_dist_hist += cnt

#fig = pylab.figure()
#ax = fig.add_subplot(3, 1, 1)
#ax.bar(bins_x[:-1], x_dist_hist, width=bins_x[1] - bins_x[0])
#ax.set_xlabel('x_dist')
#ax.set_ylabel('count')
output_fn = 'weights_vs_dist.png'
print 'output_fn', output_fn
pylab.savefig(output_fn)
#pylab.show()
