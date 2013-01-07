import pylab
import numpy as np
import sys

if (len(sys.argv) < 2):
    fn = raw_input("Please enter data file to be plotted\n")
else:
    fn = sys.argv[1]

data = np.load(fn)
#data = np.loadtxt(fn)
if (data.ndim == 1):
    x_axis = np.arange(data.size)
    pylab.plot(x_axis, data)
else:
    pylab.plot(data[:,0], data[:, 1], '-')

#pylab.plot((0.12, 0.12), (0, 7000), lw=2, c='k')
pylab.show()
