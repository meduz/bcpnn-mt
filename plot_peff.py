import pylab
import numpy as np
import sys
import utils
from scipy.optimize import leastsq

def residuals_function(p, x, y):
    """
    x: normal x coordinate
    p: parameters of the function to fit, e.g. a and b in y = a * x + b
    """
    y1 = peval_function(x, p)
    err = y - y1
    return err 


def peval_function(x, p):
    y = p[0] * x + p[1]
    return y



#pylab.rcParams.update({'path.simplify' : False})
pylab.figure()

fn = sys.argv[1]
data = np.loadtxt(fn)

x_axis = data[:, 0]
y = data[:, 1]
yerr = data[:, 2]
pylab.errorbar(x_axis, y, yerr=yerr, lw=2, label=fn)


guess_params = [.1, 0.]
opt_params = leastsq(residuals_function, guess_params, args=(x_axis, y), maxfev=10000)
opt_func = peval_function(x_axis, opt_params[0])
pylab.plot(x_axis, opt_func, lw=2, label='fitted function')

print 'optimal parameters:', opt_params[0]

pylab.legend()
pylab.show()
