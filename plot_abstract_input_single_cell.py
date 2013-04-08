import pylab
import numpy as np
import sys

if (len(sys.argv) < 2):
    fn = raw_input("Please enter data file to be plotted\n")
else:
    fn = sys.argv[1]

rcParams = { 'axes.labelsize' : 18,
            'label.fontsize': 20,
            'xtick.labelsize' : 16, 
            'ytick.labelsize' : 16, 
            'legend.fontsize': 9}
pylab.rcParams.update(rcParams)
import figure_sizes as fs
fig = pylab.figure(figsize=fs.get_figsize(800))
ax = fig.add_subplot(111)
data = pylab.loadtxt(fn)
if (data.ndim == 1):
    x_axis = np.arange(data.size)
    ax.plot(x_axis, data, lw=3)
else:
    ax.plot(data[:,0], data[:, 1], '-')

#pylab.plot((0.12, 0.12), (0, 7000), lw=2, c='k')


ax.set_xlabel('Time [ms]')
ax.set_ylabel('Response strength [a.u.]')

pylab.subplots_adjust(right=.90)
pylab.subplots_adjust(top=.90)
pylab.subplots_adjust(bottom=0.10)
pylab.subplots_adjust(left=.10)

output_fn = 'input.png'
pylab.savefig(output_fn)
pylab.show()
