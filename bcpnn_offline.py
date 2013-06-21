import numpy as np
import pylab
try:
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size
    print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
except:
    USE_MPI = False
    pc_id, n_proc, comm = 0, 1, None
    print "MPI not used"

from mpi4py import MPI
comm = MPI.COMM_WORLD

pc_id, n_proc = comm.rank, comm.size

tau_dict = {'tau_zi' : 50.,    'tau_zj' : 5.,
                'tau_ei' : 100.,   'tau_ej' : 100., 'tau_eij' : 100.,
                'tau_pi' : 1000.,  'tau_pj' : 1000., 'tau_pij' : 1000.,
                }

# time axis
dt = 0.1
t_stop = 500
t_axis = np.arange(0, t_stop, dt)

# traces
zi = np.zeros(t_axis.size)


# pre-spikes
t_pre_start = 50.
t_pre_stop = 150.
n_pre = 10

# post-spikes
t_post_start = 50.
t_post_stop = 150.
n_post = 10

spike_list_pre = np.random.uniform(t_pre_start, t_pre_stop, n_pre)
spike_list_post = np.random.uniform(t_post_start, t_post_stop, n_post)


# compute z_i trace
for prespike in spike_list_pre:
    idx = int(prespike / dt)
    zi[idx:] += np.exp(-t_axis[idx:] / tau_dict['tau_zi'])
for postspike in spike_list_post:
    idx = int(postspike / dt)
    zi[idx:] += np.exp(-t_axis[idx:] / tau_dict['tau_zi'])



fig = pylab.figure()
ax1 = fig.add_subplot(111)
#ax2 = pylab.add_subplot(212)

ax1.plot(t_axis, zi, label='zi')
ax1.legend()

pylab.show()
