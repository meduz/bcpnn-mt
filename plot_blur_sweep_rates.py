import pylab
import numpy as np
import sys

#if (len(sys.argv) < 2):
#    fn = raw_input("Please enter data file to be plotted\n")
#else:
#    fn = sys.argv[1]

fn  = 'L_net.dat'
data = pylab.loadtxt(fn)

x_axis_idx = 0
x_axis = data[:, x_axis_idx]
n = x_axis.size
y_axis = np.zeros(n)
for i in xrange(n):
    y_axis[i] = data[i, 2:].sum()


fig = pylab.figure()
ax = fig.add_subplot(111)

ax.plot(x_axis, y_axis)

xlabel = 'blur_x'
ylabel = 'Sum over Integral over L_i(t)'
#title = 'After scaling f_max in dependence of blur_x'
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
#ax.set_title(title)


pylab.show()
