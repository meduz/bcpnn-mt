"""
This script assumes that you've run abstract_training.py before (the actual training is not necessary,
it's enough to have create_stimuli)

Two different tuning curves can be measured:
    1) stimulus orientation  vs  output rate
    2) stimulus distance  vs  output rate
"""
import os
import pylab
import time
import numpy as np
import utils
import simulation_parameters
import matplotlib.mlab as mlab
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

n_points = 20

def get_tuning_curve(blur_x, blur_v):
    params['blur_X'] = blur_x
    params['blur_V'] = blur_v
    location_tuning = np.zeros((n_points, 5))
    direction_tuning = np.zeros((n_points, 5))
    # row = cell gid
    # column0 : minimal distance between cell and stimulus during stimulus presentation
    #       1 : maximum input
    #       2 : summed input
    #       3 : |v_cell - v_stim|
    #       4 : angle(v_cell, v_stim)


    dt = params['dt_rate']
    time = np.arange(0, params['t_sim'], dt)
    tp = (0.5, 0.5, 0.2, 0.0)
    tp_ = np.array([[0.5, 0.5, 0.2, 0.0]])
    v_cell = np.sqrt(tp[2]**2 + tp[3]**2)

    # measure L_i vs location tuning curve
    x_stim_start = 0.2
    y_stim_start = np.linspace(0.2, 0.8, n_points)
    v_x_stim = .2
    v_y_stim = .0

    for stim in xrange(n_points):
        L_input = np.zeros(time.shape[0])
        mp = (x_stim_start, y_stim_start[stim], v_x_stim, v_y_stim)

        for i_time, time_ in enumerate(time):
            L_input[i_time] = utils.get_input(tp_, params, time_/params['t_stimulus'], motion_params=mp)

        debug_fn = 'delme_%d.dat' % stim
        np.savetxt(debug_fn, L_input)
        dist = utils.get_min_distance_to_stim(mp, tp, params)
        dist = dist[1]
        v_stim = np.sqrt(mp[2]**2 + mp[3]**2)
        location_tuning[stim, 0] = dist
        location_tuning[stim, 1] = L_input.max()
        location_tuning[stim, 2] = L_input.sum()
        location_tuning[stim, 3] = np.sqrt((mp[2] - tp[2])**2 + (mp[3] - tp[3])**2)
        location_tuning[stim, 4] = np.pi - np.arcsin(tp[2] / v_cell) - np.arcsin(mp[2] / v_stim)
        print 'debug dist', stim, dist, 'L_input.max()=', location_tuning[stim, 1], '\n', mp, '\t', tp

    # measure L_i vs direction tuning curve
    v_theta = np.linspace(0, np.pi, n_points)
    v_stim = .2
    v_x_stim = v_stim * np.cos(v_theta)
    v_y_stim = v_stim * np.sin(v_theta)
    for stim in xrange(n_points):
        L_input = np.zeros(time.shape[0])
        x_stim_start = tp[0] - v_x_stim[stim]
        y_stim_start = tp[0] - v_y_stim[stim]
        mp = (x_stim_start, y_stim_start, v_x_stim[stim], v_y_stim[stim])
        for i_time, time_ in enumerate(time):
            L_input[i_time] = utils.get_input(tp_, params, time_/params['t_stimulus'], motion_params=mp)
        dist = utils.get_min_distance_to_stim(mp, tp, params)[1]
        v_stim = np.sqrt(mp[2]**2 + mp[3]**2)

        direction_tuning[stim, 0] = dist
        direction_tuning[stim, 1] = L_input.max()
        direction_tuning[stim, 2] = L_input.sum()
        direction_tuning[stim, 3] = np.sqrt((mp[2] - tp[2])**2 + (mp[3] - tp[3])**2)
        direction_tuning[stim, 4] = np.pi - np.arcsin(tp[2] / v_cell) - np.arcsin(mp[2] / v_stim)

    return location_tuning, direction_tuning

ms = 3
fig = pylab.figure()
pylab.subplots_adjust(hspace=0.4)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.set_title('Tuning curves for position and direction')
#ax3 = fig.add_subplot(313)
blurs = np.arange(0.05, .3, 0.05)

n_blurs = blurs.size
for blur in xrange(n_blurs):
    blur_x = blurs[blur]
    blur_v = blurs[blur]
    location_tuning, direction_tuning = get_tuning_curve(blur_x, blur_v)

    ax1.plot(location_tuning[:, 0], location_tuning[:, 1], 'o', markersize=ms, color='k')
    x_axis = np.arange(0, 1, 0.01)
    gauss_x = location_tuning[:, 1].max() * np.exp( -x_axis**2 / (2 * params['blur_X']**2))
    ax1.plot(x_axis, gauss_x, label='blur_x=%.2f' % (blur_x), lw=2)

    ax1.set_xlabel('Minimal distance between stimulus and cells')
    ax1.set_ylabel('Maximum response\nto stimulus')

    gauss_v = location_tuning[:, 1].max() * np.exp( -x_axis**2 / (2 * params['blur_V']**2))
    ax2.plot(direction_tuning[:, 3], direction_tuning[:, 1], 'o', markersize=ms, color='k')
    ax2.plot(x_axis, gauss_v, label='blur_v=%.2f' % (blur_v), lw=2)
    ax2.set_xlabel('|v_cell - v_stim|')
    ax2.set_ylabel('Maximum response\nto stimulus')

#    ax3.plot(output_array[stim, :, 4], output_array[stim, :, 2], 'o', markersize=ms, color='k')
#    ax3.set_xlabel('angle(v_cell, v_stim)')
#    ax3.set_ylabel('Integrated response\n to stimulus')

ax1.legend()
ax2.legend()
output_fn = params['figures_folder'] + 'abstract_tuning_curve.png'
print 'Saving to', output_fn
pylab.savefig(output_fn)
pylab.show()


#t_start = time.time()
#t_stop = time.time()
#t_diff = t_stop - t_start
#print "Full time for %d runs: %d sec %.1f min" % (sim_cnt, t_diff, t_diff / 60.)


