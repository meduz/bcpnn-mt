import numpy as np
import pylab
from NeuroTools import parameters as ntp

folder_name = 'Times/' # where the files are stored
fn_base = '%s/times_dict_np' % folder_name

n_procs = [24, 48, 96, 192, 384]

def get_values(key):
    d = np.zeros(len(n_procs))
    for i_, n_proc in enumerate(n_procs):
        fn = fn_base + '%d.py' % n_proc
        times = ntp.ParameterSet(fn)
        dct = dict(times) 
        d[i_] = dct[key]
    return d



fig = pylab.figure(figsize=(12, 10))
pylab.subplots_adjust(hspace=.4)
fig2 = pylab.figure(figsize=(12, 10))
pylab.subplots_adjust(hspace=.4)

keys = ['t_all', 't_connect', 't_create', 't_record', 't_calc_conns', 't_sim', 'time_to_import']

n_plots = len(keys)
n_rows, n_cols = 4, 2
fig_cnt = 1


for key in keys:
    ax = fig.add_subplot(n_rows, n_cols, fig_cnt)
    ax2 = fig2.add_subplot(n_rows, n_cols, fig_cnt)
    data = get_values(key)
    line1 = ax.plot(n_procs, data, lw=3, label='measured')

    ideal_curve = [ data[i] * 24. / n_procs[i] for i in xrange(len(n_procs))]
    line2 = ax.plot(n_procs, ideal_curve, ls='--', lw=3, label='ideal')
    ax.set_ylabel('Time [s]')
    ax.set_title(key)
    ax.set_xticks(n_procs)

    fig.legend((line1[0], line2[0]), ('measured', 'ideal'), loc=(.6 ,.15))

    d_normed = data / data[0] 
    ideal_curve = [ 24. / n_procs[i] for i in xrange(len(n_procs))]
    line1 = ax2.plot(n_procs, d_normed, lw=3, label='measured')
    line2 = ax2.plot(n_procs, ideal_curve, ls='--', lw=3, label='ideal')
    ax2.set_ylabel('Time compared to %d cores' % (n_procs[0]))
    ax2.set_title(key)
    ax2.set_xticks(n_procs)

    fig2.legend((line1[0], line2[0]), ('measured', 'ideal'), loc=(.6, .15))
    fig_cnt += 1

ax.set_xlabel('Number of processors')
ax2.set_xlabel('Number of processors')

output_fn = folder_name + 'measured.png'
print 'Saving to:', output_fn
fig.savefig(output_fn)
output_fn = folder_name + 'ideal_behaviour.png'
print 'Saving to:', output_fn
fig2.savefig(output_fn)
#ax.set_xscale('log')
#ax.set_title(key)
#ax.set_xlabel('Num cores')
#ax.set_ylabel('Time [s]')
#ax2.set_xlabel('Num cores')
#ax2.set_ylabel('Time compared to %d cores' % (n_procs[0]))


pylab.show()
