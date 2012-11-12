import numpy as np

import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary
tp = np.loadtxt(params['tuning_prop_means_fn'])

weights_fn = 'Anders_data/wij.dat'

n_bins_x = 20

n_cells = params['n_exc']
# distance matrices based on different vectors
x_distance_matrix = np.zeros((n_cells, n_cells)) # only spatial tuning
v_distance_matrix = np.zeros((n_cells, n_cells)) # only directional tuning
tp_distance_matrix = np.zeros((n_cells, n_cells)) # 4-dim tuning vector

for i in xrange(n_cells):
    print 'i', i
    for j in xrange(n_cells-i):
        x_distance_matrix[i, j] = np.sqrt((tp[i, 0] - tp[j, 0])**2 + (tp[i, 1] - tp[j, 1])**2)
        v_distance_matrix[i, j] = np.sqrt((tp[i, 2] - tp[j, 2])**2 + (tp[i, 3] - tp[j, 3])**2)
        tp_distance_matrix[i, j] = np.sqrt((tp[i, 0] - tp[j, 0])**2 + (tp[i, 1] - tp[j, 1])**2 + (tp[i, 2] - tp[j, 2])**2 + (tp[i, 3] - tp[j, 3])**2)

        x_distance_matrix[j, i] = x_distance_matrix[i, j]
        v_distance_matrix[j, i] = v_distance_matrix[i, j]
        tp_distance_matrix[j, i] = tp_distance_matrix[i, j]

print 'x_dist: min %.3e\tmax %.3e\tmean %.3e\t+- %.2e' % (x_distance_matrix.min(), x_distance_matrix.max(), x_distance_matrix.mean(), x_distance_matrix.std())
print 'v_dist: min %.3e\tmax %.3e\tmean %.3e\t+- %.2e' % (v_distance_matrix.min(), v_distance_matrix.max(), v_distance_matrix.mean(), v_distance_matrix.std())
print 'tp_dist: min %.3e\tmax %.3e\tmean %.3e\t+- %.2e' % (tp_distance_matrix.min(), tp_distance_matrix.max(), tp_distance_matrix.mean(), tp_distance_matrix.std())
