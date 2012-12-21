import pylab
import numpy as np
import sys

def plot_all(params, pre_id, post_id, iteration_0, iteration_1, \
        L_i, L_j, d_zi, d_zj, d_ei, d_ej, d_eij, d_pi, d_pj, d_pij, \
        d_wij, d_bias, fig=None, text=None, show=True, output_fn=None, **kwargs):

    # --------------------------------------------------------------------------
    def get_figsize(fig_width_pt):
        inches_per_pt = 1.0/72.0                # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width*golden_mean      # height in inches
        fig_size =  [fig_width,fig_height]      # exact figsize
        return fig_size

    def get_figsize_landscape(fig_width_pt):
        inches_per_pt = 1.0/72.0                # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width/golden_mean      # height in inches
        fig_size =  [fig_width,fig_height]      # exact figsize
        return fig_size


    def get_figsize_A4():
        fig_width = 8.27
        fig_height = 11.69
        fig_size =  [fig_width,fig_height]      # exact figsize
        return fig_size

    params2 = {'backend': 'eps',
              'axes.labelsize': 12,
              'text.fontsize': 12,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'legend.pad': 0.2,     # empty space around the legend box
              'legend.fontsize': 12,
               'lines.markersize': 3,
               'lines.linewidth': 3,
              'font.size': 12,
              'path.simplify': False,
#              'figure.figsize': get_figsize_A4()}
#              'figure.figsize': get_figsize_landscape(600)}
              'figure.figsize': get_figsize(600)}

    pylab.rcParams.update(params2)
    # --------------------------------------------------------------------------
    
    input_params = np.loadtxt(params['parameters_folder'] + 'input_params.txt')
    t_axis = np.arange(0, d_zi.size * params['dt_rate'], params['dt_rate'])

    tp_fn = params['tuning_prop_means_fn']
    tp = np.loadtxt(tp_fn)
    if fig == None:
        fig = pylab.figure()
    pylab.subplots_adjust(hspace=.6, wspace=0.05, left=0.05, bottom=0.05, right=0.95, top=0.95)
#    pylab.subplots_adjust(wspace=.05)
#    fig.text(0.2, 0.95, text, fontsize=12)

    c1, c2, c3 = 'b', 'g', 'k' # line colors
    n_iterations = iteration_1 - iteration_0
    n_rows, n_cols = 7, n_iterations
    for i_, iteration in enumerate(range(iteration_0, iteration_1)):
        mp = input_params[iteration, :]
        ax = psac.return_plot([pre_id, post_id], '%d%d%d' % (n_rows, n_cols, i_+1), fig, motion_params=mp)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Stimulus, and predicted directions")

    n_rows, n_cols = 7, 1

    ax = fig.add_subplot(n_rows, n_cols, 2)
    ax.set_title('Pre_gid: %d - Post_gid: %d' % (pre_id, post_id))
    ax.plot(np.arange(0, L_i.size * params['dt_rate'], params['dt_rate']), L_i, c=c1, label='L_i')
    ax.plot(np.arange(0, L_i.size * params['dt_rate'], params['dt_rate']), L_j, c=c2, label='L_j')
    ax = fig.add_subplot(n_rows, n_cols, 2)
    ax.plot(t_axis, d_zi, c=c1, ls='--', label='z_i')
    ax.plot(t_axis, d_zj, c=c2, ls='--', label='z_j')
    ax.set_ylabel("L_i, L_j,\nz_i, z_j")
    ax.set_xlabel('Time [ms]')
    ax.legend()

    ax = fig.add_subplot(n_rows, n_cols, 3)
    ax.plot(t_axis, d_ei, c=c1)
    ax.plot(t_axis, d_ej, c=c2)
    ax.set_ylabel("e_i, e_j")
    ax.set_xlabel('Time [ms]')

    ax = fig.add_subplot(n_rows, n_cols, 4)
    ax.plot(t_axis, d_pi, c=c1)
    ax.plot(t_axis, d_pj, c=c2)
    ax.set_ylabel("p_i, p_j")
    ax.set_xlabel('Time [ms]')

    ax = fig.add_subplot(n_rows, n_cols, 5)
    ax.plot(t_axis, d_eij, c=c3, ls='--', label='eij')
    ax.set_xlabel('Time [ms]')
    ax.legend()
    ax = fig.add_subplot(n_rows, n_cols, 5)
    ax.plot(t_axis, d_pij, c=c3, ls=':', label='pij')
    ax.set_ylabel("e_ij and p_ij")
    ax.set_xlabel('Time [ms]')
    ax.legend()

    ax = fig.add_subplot(n_rows, n_cols, 6)
    ax.plot(t_axis, d_wij, c=c3)
    ax.set_ylabel("w_ij")
    ax.set_xlabel('Time [ms]')

    ax = fig.add_subplot(n_rows, n_cols, 7)
    ax.plot(t_axis, d_bias, c=c3)
    ax.set_ylabel("Bias")
    ax.set_xlabel('Time [ms]')

#    ax = fig.add_subplot(n_rows, n_cols, 8)
#    if text == None:
#        text = 'iteration: %d\n' % (iteration)
#        text += 'pre_id=%d  post_id=%d\n' % (pre_id, post_id)
#        text += '\nstim: ' + str(params['motion_params'])
#    ax.annotate(text, (.1, .1), fontsize=12)
#    if output_fn == None:
#        output_fig_fn = params['figures_folder'] + 'bcpnn_traces.png'
#    else:
#        output_fig_fn = output_fn

    if show == True:
        pylab.show()
    else:
        print 'Saving figure to:', output_fig_fn
        pylab.savefig(output_fig_fn)
        return fig

if __name__ == '__main__':
#    if (len(sys.argv) < 3):
#        print "Please give 2 gids to be plotted:\n"
#        pre_id = int(raw_input("GID 1:\n"))
#        post_id = int(raw_input("GID 2:\n"))
#    else:
    pre_id = int(sys.argv[1])
    post_id = int(sys.argv[2])
    cycle = int(sys.argv[3])
    print 'debug', pre_id, post_id, cycle

    import plot_stimulus_and_cell_tp as psac
    import simulation_parameters
    PS = simulation_parameters.parameter_storage()
    params = PS.params

    n_iterations = 8
    it_0 = cycle * n_iterations
    it_1 = it_0 + n_iterations
    n_time_steps = 300

    input_i = np.zeros(n_time_steps * n_iterations)
    input_j = np.zeros(n_time_steps * n_iterations)
    zi = np.zeros(n_time_steps * n_iterations)
    zj = np.zeros(n_time_steps * n_iterations)
    ei = np.zeros(n_time_steps * n_iterations)
    ej = np.zeros(n_time_steps * n_iterations)
    pi = np.zeros(n_time_steps * n_iterations)
    pj = np.zeros(n_time_steps * n_iterations)
    eij = np.zeros(n_time_steps * n_iterations)
    pij = np.zeros(n_time_steps * n_iterations)
    wij = np.zeros(n_time_steps * n_iterations)
    bias = np.zeros(n_time_steps * n_iterations)

    for iteration in xrange(it_0, it_1):
        t1 = (iteration - it_0) * n_time_steps
        t2 = (iteration - it_0 + 1) * n_time_steps
        input_fn_base = '%sTrainingInput_%d/%s' % (params['folder_name'], iteration, params['abstract_input_fn_base'])
        input_i[t1:t2] = np.loadtxt(input_fn_base + '%d.dat' % (pre_id))
        input_j[t1:t2] = np.loadtxt(input_fn_base + '%d.dat' % (post_id))
        fn_debug = "%szi_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, pre_id)
        print 'debug 2', pre_id, post_id, cycle, fn_debug
        zi[t1:t2] = np.loadtxt("%szi_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, pre_id))
        zj[t1:t2] = np.loadtxt("%szj_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, post_id))
        ei[t1:t2] = np.loadtxt("%sei_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, pre_id))
        ej[t1:t2] = np.loadtxt("%sej_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, post_id))
        pi[t1:t2] = np.loadtxt("%spi_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, pre_id))
        pj[t1:t2] = np.loadtxt("%spj_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, post_id))
        eij[t1:t2] = np.loadtxt('%seij_%d_%d_%d.dat' % (params['bcpnntrace_folder'], iteration, pre_id, post_id))
        pij[t1:t2] = np.loadtxt('%spij_%d_%d_%d.dat' % (params['bcpnntrace_folder'], iteration, pre_id, post_id))
        wij[t1:t2] = np.loadtxt("%swij_%d_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, pre_id, post_id))
        bias[t1:t2] = np.loadtxt("%sbias_%d_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, pre_id, post_id))

    plot_all(params, pre_id, post_id, it_0, it_1, \
            input_i, input_j, zi, zj, ei, ej, eij, pi, pj, pij, wij, bias)
