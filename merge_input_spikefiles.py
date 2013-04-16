import os
import numpy as np
import json
import sys

if len(sys.argv) > 1:
    if sys.argv[1].isdigit():
        gid = int(sys.argv[1])
    else:
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.json'
        print 'Loading parameters from', param_fn
        f = file(param_fn, 'r')
        params = json.load(f)
else:
    import simulation_parameters
    ps = simulation_parameters.parameter_storage()
    params = ps.params


input_spike_trains = [[] for i in xrange(params['n_exc'])]
n_input_spikes = np.zeros(params['n_exc'])
for i in xrange(params['n_exc']):
    fn = params['input_st_fn_base'] + '%d.npy' % i
    if os.path.exists(fn):
        d = np.load(fn)
        input_spike_trains[i] = d
        n_input_spikes[i] = d.size
    else:
        print 'Missing input file:', fn

print 'Building output array'
output_array = np.zeros((n_input_spikes.sum(), 2))
idx_0 = 0
for i in xrange(params['n_exc']):
    idx_1 = n_input_spikes[i] + idx_0
    if n_input_spikes[i] > 0:
        output_array[idx_0:idx_1, 1] = np.ones(idx_1 - idx_0) * i
        output_array[idx_0:idx_1, 0] = input_spike_trains[i]
        idx_0 = idx_1

output_fn = params['input_folder'] + 'merged_input.dat'
print 'Saving to', output_fn
np.savetxt(output_fn, output_array)
