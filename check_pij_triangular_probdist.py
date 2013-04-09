import numpy as np
import utils
import sys

w_sigma = np.arange(0.01, 2.0, 0.01)
#w_sigma = np.arange(0.01, 0.3, 0.05)

n_tgt = 2000
n_src = 2000

n = n_tgt * n_src
for w_sigma_x in w_sigma:
    output_fn = 'p_effective/peff_triangular_wsigma%.3f.dat' % (w_sigma_x)
    output = '# p_max, p_eff\n'
    output_file = file(output_fn, 'w')
    print 'Writing to:', output_fn
    print output
    for p_max in np.arange(0.005, .9, .005):
        d_ij_samples = np.random.triangular(0., .5, .7, n)
        p_ij = p_max * np.exp(-d_ij_samples**2 / (2 * w_sigma_x**2))
        rnd_nums = np.random.rand(n)
        n_conn = (rnd_nums < p_ij).nonzero()[0].size
        p_eff = n_conn / (n_src * float(n_tgt))
        output = '%.3e\t%.3e\t%.2e\n' % (p_max, p_eff, w_sigma_x)
        output_file.write(output)
        output_file.flush()
        print p_max, p_eff, w_sigma_x

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
