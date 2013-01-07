import os
import simulation_parameters
import pylab
import numpy as np
import utils
import sys
import matplotlib
from matplotlib import cm

# load simulation parameters
def return_plot(iteration=None, fig=None):
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

    fn = params['tuning_prop_means_fn']
    tp = np.loadtxt(fn)
    if iteration == None:
        mp = params['motion_params']
    else:
        motion_params_fn = params['parameters_folder'] + 'input_params.txt'
        all_mp = np.loadtxt(motion_params_fn)
        mp = all_mp[iteration, :]

    n_cells = tp[:, 0].size
    #n_cells = 10
    ms = 10 # markersize for scatterplots
    bg_color = 'w'
    pylab.rcParams['lines.markeredgewidth'] = 0

    input_sum = np.zeros(n_cells)
    if iteration == None:
        input_fn_base = params['input_rate_fn_base'] 
        fn_ending = '.npy'
    else:
        input_fn_base = params['folder_name'] + 'TrainingInput_%d/abstract_input_' % iteration
        fn_ending = '.dat'

    for i in xrange(n_cells):
        input_fn = input_fn_base + str(i) + fn_ending
        if iteration == None:
            rate = np.load(input_fn)
        else:
            rate = np.loadtxt(input_fn)
        input_sum[i] = rate.sum()

    input_max = input_sum.max()
    print 'input_max', input_max
    idx = input_sum.argsort()
    n_mac = int(round(params['n_exc'] * 0.05))
    mac = idx[-n_mac:]
    print 'motion stimulus', mp
    print 'most activated cells:'
    for i in mac:
        dist = tp[i, :] - mp
#        print i, tp[i, :], np.sqrt(np.dot(dist, dist)), input_sum[i], input_max
    if fig == None:
        fig = pylab.figure(facecolor=bg_color)
    ax = fig.add_subplot(211)
#    h = 240.
#    s = 1. # saturation

    o_min = 0.
    o_max = input_max
    norm = matplotlib.mpl.colors.Normalize(vmin=o_min, vmax=o_max)
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.binary)#PuBu)#jet)
    m.set_array(np.arange(o_min, o_max, (o_max-o_min)/1000.))

    for i in idx:
        if i in mac:
            print 'debug', i, input_sum[i], input_max, input_sum[i] / input_max
        c = m.to_rgba(input_sum[i])
#        l = 1. - 0.5 * input_sum[i] / input_max
#        assert (0 <= h and h < 360)
#        assert (0 <= l and l <= 1)
#        assert (0 <= s and s <= 1)
#        (r, g, b) = utils.convert_hsl_to_rgb(h, s, l)
        x, y, u, v = tp[i, :]
        ax.plot(x, y, 'o', c=c, markersize=ms)
#        ax.plot(x, y, 'o', c=(r,g,b), markersize=ms)
        
#        if l < .7:
        if i in mac:
            ax.annotate('%d' % i, (x + np.random.rand() * 0.02, y + np.random.rand() * 0.02), fontsize=10)

    stim_color = 'k'
    ax.quiver(mp[0], mp[1], mp[2], mp[3], angles='xy', scale_units='xy', scale=1, color=stim_color, headwidth=4, pivot='tail')
    ax.annotate('Stimulus', (mp[0]+.5*mp[2], mp[1]+0.1), fontsize=12, color=stim_color)

    fig.colorbar(m)

    ax.set_xlim((-.05, 1.05))
    ax.set_ylim((-.05, 1.05))
    if iteration == None:
        iteration = 0
    ax.set_title('Abstract activtation iteration %d' % iteration)

    ax = fig.add_subplot(212)
    ax.bar(range(params['n_exc']), input_sum, width=1)

    output_fn = params['figures_folder'] + 'abstract_activation_%d.png' % (iteration)
    print "Saving figure: ", output_fn
    pylab.savefig(output_fn)#, facecolor=bg_color)
    return ax


if __name__ == '__main__':
#    utils.sort_cells_by_distance_to_stimulus(448)

    if len(sys.argv) > 1:
        iteration = int(sys.argv[1])
    else:
        iteration = None
    return_plot(iteration=iteration)


#    if len(sys.argv) > 2:
    pylab.show()


