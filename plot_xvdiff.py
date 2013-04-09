"""
To run this script run
    sweep_through_results.py 
first several times to get the data files to be plotted here.
"""
import numpy as np
import os
import simulation_parameters
import re
import pylab

conn_codes = ['AIII', 'IIII', 'AAAA', 'IAAA', 'RRRR']
#parameters = [0.02, 0.025, 0.03, 0.035]
parameters = np.arange(0.02, 0.042, 0.002)
#parameters = [200]
t_range = [0, 3000]
output_fn = 'xvdiff_vs_wee_t%d-%d.png' % (t_range[0], t_range[1])

#xlabel = '$w_{EE}: $ Strength of recurrent excitatory connections'
xlabel = 'Time of blank [ms]'

colors = ['b', 'g', 'r', 'm', 'c']
linestyles = ['-', '--', '-.', ':']

assert len(conn_codes) <= len(colors)
assert len(parameters) <= len(linestyles)


fig = pylab.figure()
pylab.subplots_adjust(hspace=.5)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

fmaxstim = 1000.
t0, t1 = '0', 't_{sim}'
#t0, t1 = '0 ms', '1000 ms'
for i_, conn_code in enumerate(conn_codes):
    for j_, param in enumerate(parameters):
#        fn = 'xvdiff_%s_tblank%d_weeSweep.dat' % (conn_code, param)
#        fn = 'xvdiff_%s_tblank%d_weeSweep_t%d-%d.dat' % (conn_code, param, t_range[0], t_range[1])
#        fn = 'xvdiff_%s_wee3.00e-02_tblankSweep_t%d-%d.dat' % (conn_code, t_range[0], t_range[1])
#        fn = 'xvdiff_%s_tbb400_fmaxstim%.2e_scaleLatency0.15_weeSweep_t%d-%d.dat' % (conn_code, fmaxstim, t_range[0], t_range[1])
        print fn
        if os.path.exists(fn):
            print 'fn:', fn
            tmp_fn = '%d.dat' % np.random.randint(0, 10**7)
            cmd = 'sort -gk 1 %s > %s' % (fn, tmp_fn)
            os.system(cmd)
            cmd = 'mv %s %s' % (tmp_fn, fn)
            os.system(cmd)

            d = np.loadtxt(fn)
            print 'plotting', fn

            c = colors[i_]
            ls = linestyles[j_]

            ax1.plot(d[:, 0], d[:, 1], c=c, ls=ls, label=conn_code, lw=3)
            ax1.set_xlabel(xlabel)
#            ylabel = '$\Delta_X$'
            ylabel = 'RMSE for x-prediction'
#            ylabel = '$\Delta_X = \int_{%s}^{%s} |\\vec{x}_{stim}(t) - \\vec{x}_{prediction}(t)| dt$' % (t0, t1)
            ax1.set_ylabel(ylabel)
            title = 'Position prediction error'
            ax1.set_title(title)

            ax2.plot(d[:, 0], d[:, 2], c=c, ls=ls, label=conn_code, lw=3)
            ax2.set_xlabel(xlabel)
            ylabel = 'RMSE for y-prediction'
#            ylabel = '$\Delta_V$'
            # = \int_{%s}^{%s} |\\vec{v}_{stim}(t) - \\vec{v}_{prediction}(t)| dt$' % (t0, t1)
            title = 'Direction prediction error'
            ax2.set_ylabel(ylabel)
#            ax2.set_ylabel('$\Delta_V$')
#            title = '$\Delta_V = \int_{%s}^{%s} |\\vec{v}_{stim}(t) - \\vec{v}_{prediction}(t)| dt$' % (t0, t1)
            ax2.set_title(title)

ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
print 'Saving to', output_fn
pylab.savefig(output_fn, dpi=200)
pylab.show()
