import pylab
import numpy as np
import sys
import utils
from scipy.optimize import leastsq

# ------- for 2   dimension 
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


# ------- for 3   dimensions
def residuals_function_3d(p, x, y, z):
    """
    x: normal x coordinate
    p: parameters of the function to fit, e.g. a and b in y = a * x + b
    """
    z1 = peval_function_3d(x, y, p)
    err = z - z1
    return err 

def peval_function_3d(x, y, p):
#    z = p[0] * x + p[1] * y + p[2] * np.ones(x.size)
#    z = p[0] * x + p[1] * x**2 + p[2] * y + p[3] * y**2 + p[4] * np.ones(x.size)
#    z = p[4] * np.exp( -(x - p[0])**2 / p[1]**2 - (y - p[2])**2 / p[3]**2)
#    z = p[0] * x * y
    z = p[0] * x * np.exp(- y / p[1])

    return z



# for 2 dimensions:
#pylab.figure()

#fn = sys.argv[1]
#data = np.loadtxt(fn)

#p_max = data[:, 0]
#p_eff = data[:, 1]
#p_eff_std = data[:, 2]
#w_sigma_x = data[:, 3]
#y = data[:, 1]
#yerr = data[:, 2]

#pylab.errorbar(p_eff, p_max, xerr=p_eff_std, lw=2, label=fn)
#guess_params = [.1, 0.]
#opt_params = leastsq(residuals_function, guess_params, args=(p_eff, p_max), maxfev=10000)
#opt_func = peval_function(p_eff, opt_params[0])
#pylab.plot(p_eff, opt_func, lw=2, label='fitted function')
#pylab.xlabel('p_eff')
#pylab.ylabel('p_max')


# for 3 dimensions:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm


fn = sys.argv[1]
data = np.loadtxt(fn)

p_max = data[:, 0]
p_eff = data[:, 1]
w_sigma_x = data[:, 3]

# get array dimensions
n_p_max = np.unique(p_max).size
n_sigma = np.unique(w_sigma_x).size

X = np.reshape(p_eff, (n_sigma, n_p_max))
Z = np.reshape(p_max, (n_sigma, n_p_max))

fig = plt.figure()#figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 1, 1, projection='3d')
#surf = ax.plot_surface(X, w_sigma_x, Z, label='data', rstride=1, cstride=1, cmap=cm.jet,
#                linewidth=0, antialiased=True)

scatter = ax.scatter(X, w_sigma_x, Z, color='r', marker='o', linewidth='3', label='data')
#fig.colorbar(surf, shrink=0.5, aspect=10)
#fig.colorbar(scatter, shrink=0.5, aspect=10)
#ax.set_zlim3d(-1.01, 1.01)

ax.set_xlabel('p_eff')
ax.set_ylabel('w_sigma_x')
ax.set_zlabel('p_max')

#pylab.errorbar(p_eff, p_max, xerr=p_eff_std, lw=2, label=fn)
#guess_params_3d = [.1, .1, .1]
#guess_params_3d = [.1, .1, .1, .1, .1]
#guess_params_3d = [.1]
guess_params_3d = [50., 0.02, 2.]
w_sigma_x = data[:, 3]
opt_params = leastsq(residuals_function_3d, guess_params_3d, args=(p_eff, w_sigma_x, p_max), maxfev=100000)
opt_func = peval_function_3d(p_eff, w_sigma_x, opt_params[0])

opt_func = opt_func.reshape((n_sigma, n_p_max))
w_sigma_x = w_sigma_x.reshape((n_sigma, n_p_max))
#print 'opt_func  - data ', opt_func - Z
print 'opt_func  - data sum', (opt_func - Z).sum()
print 'opt_func  - data mean', (opt_func - Z).mean()

#fig = plt.figure()
#ax2 = fig.add_subplot(1, 1, 1, projection='3d')
#surf2 = ax2.plot_surface(X, w_sigma_x, opt_func, label='fit', rstride=1, cstride=1, cmap=cm.jet,
#                linewidth=0, antialiased=True)
scatter = ax.scatter(X, w_sigma_x, opt_func, color='b', marker='o', linewidth='3', label='fit')
#fig.colorbar(surf2, shrink=0.5, aspect=10)
#pylab.plot(p_eff, opt_func, lw=2, label='fitted function')
#pylab.xlabel('p_eff')
#pylab.ylabel('p_max')



#p_eff = data[:, 1]
#w_sigma_x = data[:, 3]
#guessed_func = peval_function_3d(p_eff, w_sigma_x, [np.sqrt(2)])
#guessed_func = guessed_func.reshape((n_sigma, n_p_max))
#scatter = ax.scatter(X, w_sigma_x, guessed_func, color='g', marker='o', linewidth='2', label='guess')

#print 'optimal parameters:', opt_params[0]

pylab.legend()
pylab.show()
