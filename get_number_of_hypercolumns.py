
import numpy as np


n_theta = 16
n_v = 3
for n_rf in xrange(28, 280):
    n_rf_x = np.int(np.sqrt(n_rf*np.sqrt(3)))
    n_rf_y = np.int(np.sqrt(n_rf)#/np.sqrt(3))) # np.sqrt(np.sqrt(3)) comes from resolving the problem "how to quantize the square with a hex grid of a total of N_RF dots?"
    n_hc = n_rf_x * n_rf_y
    n_mc_per_hc = n_theta * n_v
    n_mc = n_hc * n_mc_per_hc
    if n_hc % 24 == 0:
        print n_rf, n_hc, n_mc
