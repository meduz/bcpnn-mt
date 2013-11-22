import sys
import numpy as np

try:
    fn = sys.argv[1]
    t_min = int(sys.argv[2])
    t_max = int(sys.argv[3])
except:
    print '\n\nUsage:\n\tpython cut_spike_file.py [FILENAME] [T_MIN] [T_MAX]'
    exit(1)

d = np.loadtxt(fn)
idx_1 = (d[:, 0] > t_min).nonzero()[0]
idx_2 = (d[:, 0] < t_max).nonzero()[0]

valid_idx = list(set(idx_1).intersection(set(idx_2)))

last_dot_idx = fn.rfind('.')
output_fn = fn[:last_dot_idx] + '_%d-%d' % (t_min, t_max) + fn[last_dot_idx:]

print 'Saving cut file to:', output_fn
np.savetxt(output_fn, d[valid_idx, :])
