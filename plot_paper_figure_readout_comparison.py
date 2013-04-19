import sys
import os
import numpy as np
import pylab



def plot_vertical_lines(ax, c='k'):
    dotted_lines = [0, 600, 800]
    solid_lines = [200]
    ylim = ax.get_ylim()
    for x_pos in dotted_lines:
        ax.plot((x_pos, x_pos), (ylim[0], ylim[1]), ls='--', c=c, lw=2)
    for x_pos in solid_lines:
        ax.plot((x_pos, x_pos), (ylim[0], ylim[1]), ls='-', c=c, lw=2)

    xticks = [0, 500, 1000, 1500, 2500]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_xlim((0, 1500))

    # blank text
#    txt = 'blank'
#    txt_pos_y = .15 * ylim[1]
#    txt_pos_x = 30
#    ax.annotate(txt, (txt_pos_x, txt_pos_y), fontsize=14, color='k')
#    txt_pos_x = 630
#    ax.annotate(txt, (txt_pos_x, txt_pos_y), fontsize=14, color='k')
#    ax.set_ylim(ylim)

rcP = { 'axes.labelsize' : 24,
        'label.fontsize': 18,
        'xtick.labelsize' : 14, 
        'ytick.labelsize' : 14, 
        'axes.titlesize'  : 24,
        'legend.fontsize': 9}

pylab.rcParams.update(rcP)

fig_size_A4 =  [11.69, 8.27]   
fig = pylab.figure(figsize=fig_size_A4)
pylab.subplots_adjust(left=0.10, bottom=0.08, right=0.97, top=0.93, wspace=0.3, hspace=.25)

ax_vx = fig.add_subplot(231)
ax_vx.set_title('$v_x$-prediction')
#ax_vx.set_xlabel('Time [ms]')
ax_vx.set_ylabel('$v_x$')

ax_vy = fig.add_subplot(232)
ax_vy.set_title('$v_y$-prediction')
#ax_vy.set_xlabel('Time [ms]')
ax_vy.set_ylabel('$v_y$')

ax_vdiff = fig.add_subplot(233)
ax_vdiff.set_title('$|v_{diff}| = |\\vec{v}_{pred} - \\vec{v}_{stim}|$')
#ax_vdiff.set_xlabel('Time [ms]')
ax_vdiff.set_ylabel('$|v_{diff}|$')

ax_x = fig.add_subplot(234)
ax_x.set_title('$x$-prediction')
ax_x.set_xlabel('Time [ms]')
ax_x.set_ylabel('$x$-position')


ax_y = fig.add_subplot(235)
ax_y.set_title('$y$-prediction')
ax_y.set_xlabel('Time [ms]')
ax_y.set_ylabel('$y$-position')

ax_xdiff = fig.add_subplot(236)
ax_xdiff.set_title('$|x_{diff}| = |\\vec{x}_{pred} - \\vec{x}_{stim}|$')
ax_xdiff.set_xlabel('Time [ms]')
ax_xdiff.set_ylabel('$|x_{diff}|$')

all_subplots = [ax_vx, ax_vy, ax_vdiff, ax_x, ax_y, ax_xdiff]

folders = {}
conn_types = ['Motion-based','Direction-based', 'Isotropic']
for ct in conn_types:
    folders[ct] = []

#folder_names = sys.argv[1:]
folder_names = ['LS_xpred_wsi2.5e-01_AIII_pee5.0e-03_fstim5.0e+03_wstim5.0e-03_wsigmax3.00e-01_wsigmav3.00e-01_wee2.00e-01_wei1.80e+00_wie8.00e-01_wii1.50e-01_delay1000_connRadius1.00/', \
        'LS_cosConn_AIII_pee5.0e-03_fstim5.0e+03_wstim5.0e-03_wsigmax1.00e+00_wsigmav1.00e+00_wee2.50e-01_wei1.80e+00_wie8.00e-01_wii1.50e-01_delay1000_connRadius0.10/', \
        'LS_wsi1.0e-01IIII_pee5.0e-03_fstim5.0e+03_wstim5.0e-03_wsigmax1.00e+00_wsigmav1.00e+00_wee2.50e-01_wei1.80e+00_wie8.00e-01_wii1.50e-01_delay1000_connRadius0.10/']

colors = ['b', 'g', 'r']
line_styles = ['-', '--', '-.']
stim_color = 'k'
line_width = 3


for i_, folder_name in enumerate(folder_names):
    data_fn = os.path.abspath(folder_name) + '/Data/' + 'vx_linear_vs_time.dat'
    d = np.loadtxt(data_fn)
    c = colors[i_]
    ax_vx.plot(d[:, 0], d[:, 1], lw=line_width, c=c, ls=line_styles[i_])
    ax_vx.plot(d[:, 0], d[:, 2], lw=line_width, c=stim_color)

    data_fn = os.path.abspath(folder_name) + '/Data/' + 'vy_linear_vs_time.dat'
    d = np.loadtxt(data_fn)
    c = colors[i_]
    ax_vy.plot(d[:, 0], d[:, 1], lw=line_width, c=c, ls=line_styles[i_])
    ax_vy.plot(d[:, 0], d[:, 2], lw=line_width, c=stim_color)

    data_fn = os.path.abspath(folder_name) + '/Data/' + 'x_linear_vs_time.dat'
    d = np.loadtxt(data_fn)
    c = colors[i_]
    ax_x.plot(d[:, 0], d[:, 1], lw=line_width, c=c, ls=line_styles[i_])
    ax_x.plot(d[:, 0], d[:, 2], lw=line_width, c=stim_color)

    data_fn = os.path.abspath(folder_name) + '/Data/' + 'y_linear_vs_time.dat'
    d = np.loadtxt(data_fn)
    c = colors[i_]
    ax_y.plot(d[:, 0], d[:, 1], lw=line_width, c=c, ls=line_styles[i_])
    ax_y.plot(d[:, 0], d[:, 2], lw=line_width, c=stim_color)

    data_fn = os.path.abspath(folder_name) + '/Data/' + 'vdiff_vs_time.dat'
    d = np.loadtxt(data_fn)
    ax_vdiff.plot(d[:, 0], d[:, 1], lw=line_width, c=c, ls=line_styles[i_])

    data_fn = os.path.abspath(folder_name) + '/Data/' + 'xdiff_vs_time.dat'
    d = np.loadtxt(data_fn)
    ax_xdiff.plot(d[:, 0], d[:, 1], lw=line_width, c=c, ls=line_styles[i_])


for ax in all_subplots:
    plot_vertical_lines(ax)

output_fn = 'readout_figure.png'
pylab.savefig(output_fn, dpi=200)
print output_fn

output_fn = 'readout_figure.pdf'
pylab.savefig(output_fn, dpi=200)
print output_fn
pylab.show()

