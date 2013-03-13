import pylab
import numpy as np
import utils
import sys

import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary

try:
    cell_type = sys.argv[1]
except:
    cell_type = 'exc'
spike_fn = params['%s_spiketimes_fn_merged' % cell_type] + '.ras'
print 'Loading ', spike_fn
#np.loadtxt(spike_fn)

nspikes = utils.get_nspikes(spike_fn, params['n_%s' % cell_type])

n_cells = nspikes.nonzero()[0].size

idx = np.argsort(nspikes)
print 'GID\tnspikes'
print '----------------'
#for i in xrange(1, int(round(.2 *(n_cells + 1)))):
for i in xrange(1, n_cells + 1):
    print '%d\t%d' % (idx[-i], nspikes[idx[-i]])

fig = pylab.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
x_axis = range(params['n_%s' % cell_type])
ax.bar(x_axis, nspikes)
ax.set_xlabel('Cell idx %s' % cell_type)
ax.set_ylabel('nspikes over full run')

with_annotations = True
if with_annotations:
    for i in xrange(1, n_cells + 1):
        gid = idx[-i] 
        ax.annotate('%s' % gid, (gid, nspikes[gid] + 1))

ax.set_ylim((0, nspikes.max() + 2))
ax.set_xlim((0, params['n_%s' % cell_type]))

output_fig = params['figures_folder'] + 'nspike_histogram_%s.png' % cell_type
print 'Saving to:', output_fig
pylab.savefig(output_fig)

pylab.show()


