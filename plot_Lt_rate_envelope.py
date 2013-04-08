import pylab
import numpy as np
import sys
import simulation_parameters
import utils

network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
tp = np.loadtxt(params['tuning_prop_means_fn'])
mp = params['motion_params']

#pylab.rcParams.update({'path.simplify' : False})
fig = pylab.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

for fn in sys.argv[1:]:
    gid = int(fn.rsplit('_')[-1].rsplit('.')[0])
    data = np.load(fn)
    x_axis = np.arange(data.size) * .1
    dist = np.zeros(data.size)
    for i in xrange(data.size):
        x_stim = mp[0] + x_axis[i] / params['t_sim'] * mp[2]
        y_stim = mp[1] + x_axis[i] / params['t_sim'] * mp[3]
        d_ij = utils.torus_distance2D(x_stim, tp[gid, 0], y_stim, tp[gid, 1])
        dist[i] = d_ij

    t_min = np.argmin(dist)
    print gid, t_min, data[t_min], np.min(dist), 'mp:', mp, 'tp:', tp[gid, :]

    ax1.plot(x_axis, data, lw=1, label=str(gid))
    ax2.plot(x_axis, dist, lw=1, label=str(gid))

#    ax1.set_title(fn)
#pylab.legend()
pylab.show()
