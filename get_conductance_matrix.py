import os
import simulation_parameters
import numpy as np
import utils
import pylab
import matplotlib 
import sys
from matplotlib import cm


# load simulation parameters
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary

n_cells = params['n_gids_to_record']

d = np.loadtxt(params['exc_spiketimes_fn_merged'] + '0.ras')
if d.size > 0:
    nspikes = utils.get_nspikes(params['exc_spiketimes_fn_merged'] + '0.ras', n_cells=params['n_exc'])
    spiking_cells = np.nonzero(nspikes)[0]
    fired_spikes = nspikes[spiking_cells]
   
    print 'Number of spikes (sum, max, mean, std):\t%d\t%d\t%.2f\t%.2f' % (nspikes.sum(), nspikes.max(), nspikes.mean(), nspikes.std())
    print '\nspiking cells', len(spiking_cells), spiking_cells, '\n fraction of spiking cells', float(len(spiking_cells))  / params['n_exc']
    print 'fired_spikes mean %.2f +- %.2f' % (fired_spikes.mean(), fired_spikes.std()), fired_spikes
else:
    print 'NO SPIKES'
    exit(1)




def calculate_input_cond():
    input_cond = np.zeros(params['n_exc'])
    for cell in xrange(params['n_exc']):
#        try:
        fn = params['input_st_fn_base'] + str(cell) + '.npy'
        spike_times = np.load(fn)
        nspikes_in = spike_times.size
#        except: # this cell does not get any input
#            print "Missing file: ", fn
#            spike_times = []
#            nspikes_in = 0
        input_cond[cell] = nspikes_in * params['w_input_exc']
    return input_cond

def get_cond_matrix(nspikes, w):
    cond_matrix = np.zeros((params['n_exc'], params['n_exc']))
    for tgt in xrange(params['n_exc']):
        for src in xrange(params['n_exc']):
            if src != tgt:
                cond_matrix[src, tgt] =  w[src, tgt] * nspikes[src]
    return cond_matrix



# compute everything
input_cond = calculate_input_cond()
os.system("python merge_connlist_ee.py")
conn_list_fn = params['merged_conn_list_ee']
print 'debug', conn_list_fn
w, delays = utils.convert_connlist_to_matrix(conn_list_fn, params['n_exc'])
cond_matrix = get_cond_matrix(nspikes, w)
np.savetxt(params['tmp_folder'] + 'input_cond.dat', input_cond)
np.savetxt(params['tmp_folder'] + 'cond_matrix.dat', cond_matrix)

# or load the computed cond_matrix etc (for replotting)
#input_cond = np.loadtxt(params['tmp_folder'] + 'input_cond.dat')
#cond_matrix = np.loadtxt(params['tmp_folder'] + 'cond_matrix.dat')

summed_network_cond = np.zeros(params['n_exc'])
for tgt in xrange(params['n_exc']):
    summed_network_cond[tgt] = cond_matrix[:, tgt].sum()


cmap = 'jet'

m1 = matplotlib.cm.ScalarMappable(cmap=cm.jet)
m1.set_array(np.arange(0, cond_matrix.max(), 0.01))
m2 = matplotlib.cm.ScalarMappable(cmap=cm.jet)
m2.set_array(np.arange(0, summed_network_cond.max(), 0.01))
m3 = matplotlib.cm.ScalarMappable(cmap=cm.jet)
m3.set_array(np.arange(0, input_cond.max(), 0.01))

#fig = pylab.figure(figsize = (11.69, 8.27))
h = 10
fig = pylab.figure(figsize = (np.sqrt(2) * h, h))
ax1 = fig.add_subplot(311)
ax1.set_title('Conductance matrix exc-exc')
ax1.set_ylabel('Source cell indices')
ax1.set_xlabel('Target cell indices')

cax1 = ax1.pcolormesh(cond_matrix, cmap=cmap)
ax1.set_xlim((0, cond_matrix.shape[0]))
ax1.set_ylim((0, cond_matrix.shape[1]))

bbax1 = ax1.get_position()
posax1 = bbax1.get_points()
#print 'bbax1', bbax1



ax2 = fig.add_subplot(312)

cax2 = ax2.pcolormesh(summed_network_cond.reshape((1, params['n_exc'])), cmap=cmap)
ax2.set_xlim((0, params['n_exc']))
ax2.set_yticklabels(())
bbax2 = ax2.get_position()
posax2 = bbax2.get_points()
ax2.set_xlabel('Summed excitatory conductance from network')

ax3 = fig.add_subplot(313)
cax3 = ax3.pcolormesh(input_cond.reshape((1, params['n_exc'])), cmap=cmap)
ax3.set_xlim((0, params['n_exc']))
ax3.set_yticklabels(())
bbax3 = ax3.get_position()
posax3 = bbax3.get_points()
ax3.set_xlabel('Input conductance')

"""
posax[0][0] = position of left border
posax[0][1] = position of lower / bottom border
posax[1][0] = position of right border
posax[1][1] = position of upper / top border
"""

height = .1
shift = .3
dh = .05
posax1[1][1] += 0.05                    # shift upper ax1 up
posax1[0][1] -= shift                   # shift lower ax1 down
posax2[1][1] = posax1[0][1] - (dh)      # adjust upper ax2 to lower ax1
posax2[0][1] = posax2[1][1] - (height)  
posax3[1][1] = posax2[0][1] - (dh)
posax3[0][1] = posax3[1][1] - (height)

print 'posax1', posax1
print 'posax2', posax2
print 'posax3', posax3

bbax1.set_points(posax1)
posax1=ax1.set_position(bbax1) #! Update axes with new position

bbax2.set_points(posax2)
posax2=ax2.set_position(bbax2) #! Update axes with new position

bbax3.set_points(posax3)
posax3=ax3.set_position(bbax3) #! Update axes with new position

cbar1 = fig.colorbar(m1, ax=ax1)

bar2_ticks = [i for i in np.linspace(summed_network_cond.min(), summed_network_cond.max()-0.01, 4)]
cbar2 = fig.colorbar(m2, ax=ax2, ticks=bar2_ticks)
cbar2.ax.set_yticklabels(['%.1f' % i for i in bar2_ticks])

bar3_ticks = [i for i in np.linspace(input_cond.min(), input_cond.max()-0.01, 4)]
cbar3 = fig.colorbar(m3, ax=ax3, ticks=bar3_ticks)
cbar3.ax.set_yticklabels(['%.1f' % i for i in bar3_ticks])

output_fig = params['figures_folder'] + 'conductance_matrix.png'
print 'output_fig:', output_fig
pylab.savefig(output_fig)

try:
    if int(sys.argv[1]) == 1:
        pylab.show()
except:
    pass
