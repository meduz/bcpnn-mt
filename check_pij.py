import numpy as np
import utils
import sys

def get_distances(tp_src, tp_tgt):
    n_src = tp_src[:, 0].size
    n_tgt = tp_tgt[:, 0].size
    dist = np.zeros(n_src * n_tgt)
    for src in xrange(n_src):
        print '%d / %d' % (src, n_src)
        for tgt in xrange(n_tgt):
            idx = src * n_tgt + tgt
            dist[idx] = utils.torus_distance2D(tp_src[src, 0], tp_tgt[tgt, 0], tp_src[src, 1], tp_tgt[tgt, 1])
    return dist
    
if len(sys.argv) > 1:
    param_fn = sys.argv[1]
    if os.path.isdir(param_fn):
        param_fn += '/Parameters/simulation_parameters.info'
    # TODO json
#    import NeuroTools.parameters as NTP
    fn_as_url = utils.convert_to_url(param_fn)
    print 'Loading parameters from', param_fn
    params = NTP.ParameterSet(fn_as_url)

else:
    print '\nPlotting the default parameters given in simulation_parameters.py\n'
    import simulation_parameters
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

tp_exc = np.loadtxt(params['tuning_prop_means_fn'])
tp_inh = np.loadtxt(params['tuning_prop_inh_fn'])

def compute_dist():
    dist = {}
    print 'computing distances'
    print 'ee'
    dist['ee']= get_distances(tp_exc, tp_inh)
    print 'ei'
    dist['ei'] = get_distances(tp_exc, tp_inh)
    print 'ie'
    dist['ie'] = get_distances(tp_inh, tp_exc)
    print 'ii'
    dist['ii'] = get_distances(tp_inh, tp_inh)
    output_fn = params['parameters_folder'] + 'distances_ee.dat'
    np.savetxt(output_fn, dist_ee)
    output_fn = params['parameters_folder'] + 'distances_ie.dat'
    np.savetxt(output_fn, dist_ie)
    output_fn = params['parameters_folder'] + 'distances_ei.dat'
    np.savetxt(output_fn, dist_ei)
    output_fn = params['parameters_folder'] + 'distances_ii.dat'
    np.savetxt(output_fn, dist_ii)
    return dist

def load_distances():
    dist = {}
    for conn_type in ['ie', 'ii', 'ei', 'ee']:
        fn = params['parameters_folder'] + 'distances_%s.dat' % conn_type
        print 'Loading distances from:', fn
        dist[conn_type] = np.loadtxt(fn)

    return dist

all_distances = load_distances()
w_sigma = np.arange(0.01, 2.0, 0.05)
#w_sigma = np.arange(0.01, 0.3, 0.05)

n_tgt = 200
n_src = 200
n_trials = 10
#distances = np.loadtxt("distances_in_large_network.dat")
#distances = np.loadtxt("distances_between_cells.dat")
for conn_type in ['ie', 'ii', 'ei', 'ee']:
    distances = all_distances[conn_type]
    n = distances.size
    for w_sigma_x in w_sigma:
        output_fn = 'p_effective/peff_wsigma%.3f_%s.dat' % (w_sigma_x, conn_type)
        output = '# p_max, p_eff, p_eff_std\n'
        output_file = file(output_fn, 'w')
        print 'Writing to:', output_fn
        print output
#        for p_max in np.arange(0.005, .9, .01):
        for p_max in np.arange(0.005, .9, .005):
            n_conn = np.zeros(n_trials)
            for trial in xrange(n_trials):
                for j in xrange(n_tgt):
                    for i in xrange(n_src):
                        d_ij = distances[np.random.randint(0, n)]
                        p_ij = p_max * np.exp(-d_ij**2 / (2 * w_sigma_x**2))
                        if np.random.rand() <= p_ij:
                            n_conn[trial] += 1
            p_eff = n_conn.mean() / (n_src * float(n_tgt))
            p_eff_std = n_conn.std() / np.sqrt(n_src * float(n_tgt))
            #    print 'n_conn', n_conn
            output = '%.3e\t%.3e\t%.3e\t%.2e\n' % (p_max, p_eff, p_eff_std, w_sigma_x)
            output_file.write(output)
            output_file.flush()
            print p_max, p_eff, p_eff_std, w_sigma_x

#n_tgt = 200
#n_src = 200
#n_trials = 20
#for w_sigma_x in w_sigma:
#    output_file.close()
#    output_fn = 'p_effective/peff_wsigma%.3f.dat' % w_sigma_x
#    output = '# p_max, p_eff, p_eff_std\n'
#    output_file = file(output_fn, 'w')
#    print 'Writing to:', output_fn
#    print output
#    for p_max in np.arange(0.005, .9, .005):
#        n_conn = np.zeros(n_trials)
#        for trial in xrange(n_trials):
#            for j in xrange(n_tgt):
#                for i in xrange(n_src):
#                    d_ij = np.random.triangular(0., 0.5, 0.7)
#                    p_ij = p_max * np.exp(-d_ij**2 / (2 * w_sigma_x**2))
#                    if np.random.rand() <= p_ij:
#                        n_conn[trial] += 1
#        p_eff = n_conn.mean() / (n_src * float(n_tgt))
#        p_eff_std = n_conn.std() / np.sqrt(n_src * float(n_tgt))
#        output = '%.3e\t%.3e\t%.3e\t%.2e\n' % (p_max, p_eff, p_eff_std, w_sigma_x)
#        output_file.write(output)
#        output_file.flush()
#        print p_max, p_eff, p_eff_std, w_sigma_x

#    output_file.close()
