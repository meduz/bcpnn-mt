import os
import numpy as np
import pylab


# ------- for 2   dimension 
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
#    y = p[0] * x + p[1]
    y = p[0] * x# + p[1]
    return y

def residuals_function_exp_decay(p, x, y):
    """
    x: normal x coordinate
    p: parameters of the function to fit, e.g. a and b in y = a * x + b
    """
    y1 = peval_function_exp_decay(x, p)
    err = y - y1
    return err 

def peval_function_exp_decay(x, p):
#    y = p[0] * np.exp(-x / p[1]) + p[2]
    y = p[0] * np.exp(-x**p[3] / p[1]) + p[2]
#    y = p[0] * np.exp(-np.sqrt(x) / p[1]) + p[2]
#    y = p[0] * np.exp(-x**2 / (2 * p[1]**2))# + p[2]
    return y



fig = pylab.figure()
ax = fig.add_subplot(111)

w_sigma_range = np.arange(0.01, 2.00, 0.01)
#w_sigma_range = np.arange(0.01, 0.3, 0.02)

opt_params_array = np.zeros((w_sigma_range.size, 3))
conn_type = 'ei'
for i_, w_sigma_x in enumerate(w_sigma_range):
#    fn = 'p_effective/peff_wsigma%.3f_%s.dat' % (w_sigma_x, conn_type)
    fn = 'p_effective/peff_triangular_wsigma%.3f.dat' % (w_sigma_x)
    d = np.loadtxt(fn)
    p_max = d[:, 0]
    p_eff = d[:, 1]
    p_eff_sem = d[:, 2]
#    ax.errorbar(p_eff, p_max, xerr=p_eff_sem, label='w_sigma_x=%.2f' % w_sigma_x)
    ax.plot(p_eff, p_max, label='w_sigma_x=%.2f' % w_sigma_x)

    # fit
    guess_params = [.1, 0.]
    opt_params = leastsq(residuals_function, guess_params, args=(p_eff, p_max), maxfev=10000)
    opt_func = peval_function(p_eff, opt_params[0])
    pylab.plot(p_eff, opt_func, lw=2, c='k', ls='--')#, label='fit w_sigma_x=%.2f' % w_sigma_x)
    pylab.xlabel('p_eff')
    pylab.ylabel('p_max')
    
    opt_params_array[i_, 0] = w_sigma_x
    opt_params_array[i_, 1] = opt_params[0][0]
    opt_params_array[i_, 2] = opt_params[0][1]

pylab.legend()
print 'opt_params:'
print opt_params_array

#guess_params = [20., 0.05, 0.01]
#guess_params = [20., 0.05, 0.01]
guess_params = [20., 0.05, 0.01, .5]
opt_params_gradient = leastsq(residuals_function_exp_decay, guess_params, args=(opt_params_array[:, 0], opt_params_array[:, 1]), maxfev=10000)
opt_func_gradient = peval_function_exp_decay(opt_params_array[:, 0], opt_params_gradient[0])
print 'fit wsigma', opt_params_gradient[0]

fig = pylab.figure()

ax = fig.add_subplot(111)

print 'difference:', (opt_params_array[:, 1] - opt_func_gradient)**2
print 'average difference:', ((opt_params_array[:, 1] - opt_func_gradient)**2).mean()
ax.plot(opt_params_array[:, 0], opt_params_array[:, 1], label='gradient')
ax.plot(opt_params_array[:, 0], opt_func_gradient, lw=2, c='k', ls='--', label='fit')
ax.set_ylabel('gradient')
ax.set_xlabel('w_sigma_x')

#ax2 = fig.add_subplot(122)
#ax2.plot(opt_params_array[:, 0], opt_params_array[:, 2], label='offset')
#ax2.set_xlabel('w_sigma_x')
#ax2.set_ylabel('offset')
pylab.legend()

pylab.show()
