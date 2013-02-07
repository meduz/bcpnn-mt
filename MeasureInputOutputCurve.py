import os
import sys
import simulation_parameters
import numpy as np
from pyNN.utility import get_script_args
from pyNN.errors import RecordingError
import pylab
import matplotlib.mlab as mlab
import numpy.random as nprnd

class NeuronSimulator(object):

    def __init__(self, parameter_storage):

        self.ps = parameter_storage
        self.output_folder = 'InputOutputCurve/'
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        self.ps.set_filenames(self.output_folder)
        self.params = self.ps.params
        setup(timestep=0.1,min_delay=0.1,max_delay=4.0, rng_seeds_seed=self.params['seed'])


    def create_neurons(self, n_neurons, cell_params, neuron_model):
        """
        For each param one population of neurons is created with n_trials neurons.
        The behavior is thus averaged over n_trials cells ( = n_trials).
        """
        self.n_cells = n_neurons
        self.cell_params = cell_params

        if neuron_model == 'EIF_cond_exp_isfa_ista':
            self.pop = Population(n_neurons, EIF_cond_exp_isfa_ista, cell_params, label='exc_cells')
        else:
            self.pop = Population(n_neurons, IF_cond_exp, cell_params, label='exc_cells')

        rng_v = NumpyRNG(seed = 3147 + self.params['seed'], parallel_safe=True) #if True, slower but does not depend on number of nodes

        self.v_init_dist = RandomDistribution('normal',
                (self.params['v_init'], self.params['v_init_sigma']),
                rng=rng_v,
                constrain='redraw',
                boundaries=(-80, -60))
        self.pop.initialize('v', self.v_init_dist)

    def connect_noise(self, f_exc_noise, f_inh_noise, w_exc_noise, w_inh_noise):

        for nrn in self.pop.all():
            spike_sourceE = create(SpikeSourcePoisson, {'rate' : f_exc_noise})
            spike_sourceI = create(SpikeSourcePoisson, {'rate' : f_inh_noise})
            connE = connect(spike_sourceE, nrn, weight=w_exc_noise, synapse_type='excitatory',delay=1.0)
            connI = connect(spike_sourceI, nrn, weight=w_inh_noise, synapse_type ='inhibitory',delay=1.0)


    def connect_input(self, w_input_exc, f_max, mu=0.5, sigma=.1):
        dt = .1
        time = np.arange(0, self.params['t_sim'], dt)
        L_input = np.zeros((self.n_cells, time.shape[0]))

        t_stop = 500
        dt = .1
        t_axis = np.arange(0, t_stop, dt)
        mu *= t_stop
        sigma *= t_stop
        L_i = mlab.normpdf(t_axis, mu, sigma)
        L_i *= f_max

        n_steps = L_i.size
        for nrn in self.pop.all():
            spike_times= []
            for i in xrange(n_steps):
                r = nprnd.rand()
                if (r <= ((L_i[i]/1000.) * dt)): # rate is given in Hz -> 1/1000.
                    spike_times.append(i * dt) 
            ssa = create(SpikeSourceArray, {'spike_times': spike_times})
            connE = connect(ssa, nrn, weight=w_input_exc, synapse_type='excitatory', delay=1.0)

            
    def run(self, sim_cnt=0):
        print 'Run ... '

        self.pop.record()
        self.pop.record_v()

        run(self.params['t_sim'])
        output_fn = self.params['exc_spiketimes_fn_merged'] + '%d.dat' % sim_cnt
        print 'output_fn spikes', output_fn
        self.pop.printSpikes(output_fn)
        output_fn = self.params['exc_volt_fn_base'] + '%d.v' % sim_cnt
        print 'output_fn volt', output_fn
        self.pop.print_v(output_fn, compatible_output=False)

        end()



    def analyze(self):

        print 'Analyze ... '
        output_folder = self.params['tmp_folder']
        n_bins =  20
        d = np.zeros((n_bins, self.n_values + 1))

        output_fn_spikes = output_folder + 'spikes_vs_%s.dat' % (self.param_name)
        for i_, key in enumerate(self.pop_dict.keys()):

            param_name = self.pop_dict[key]['param_name']
            value = self.pop_dict[key]['value']
            spike_fn = self.params['exc_spiketimes_fn_merged'] + '%d.dat' % i_
            spikes = np.loadtxt(spike_fn)
            n, bins = np.histogram(spikes[:, 0], n_bins, range=(0, self.params['t_sim']))
            n /= float(self.n_trials)
            d[:, i_ + 1] = n
#            d[:, 0] = 
#            volt_fn = self.params['exc_volt_fn_base'] + '%d.v' % i_

        x_axis = bins[:-1]

        color_list = ['b', 'g', 'r', 'y', 'c', 'm', \
                (134./255., 0, 28./255.), '#00f80f', '#deff00', '#ff00e4', '#00ffe6']

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        for i_, key in enumerate(self.pop_dict.keys()):
            label = '%.2e' % (self.pop_dict[key]['value'])
            ax.plot(x_axis, d[:, i_], label=label, c=color_list[i_ % len(color_list)])
            title = 'Sweep for %s' % (self.pop_dict[key]['param_name'])

        # control group
        spike_fn = self.params['exc_spiketimes_fn_merged'] + 'control.dat'
        spikes = np.loadtxt(spike_fn)
        n, bins = np.histogram(spikes[:, 0], n_bins, range=(0, self.params['t_sim']))
        n /= self.n_trials
        d[:, 0] = n
        print 'Saving to:', output_fn_spikes
        np.savetxt(output_fn_spikes, d)

        ax.plot(x_axis, d[:, 0], label='IF_cond_exp', lw=3, ls='--', c='k')

        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Mean number of output spikes')
        ax.legend(loc='upper left')
        ax.set_title(title)
        pylab.show()


if __name__ == '__main__':

    simulator_name  = 'nest'
    exec("from pyNN.%s import *" % simulator_name)

    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    NS = NeuronSimulator(network_params)

    n_values = 10   # number of g_in / f_in values
    n_cells = 20   # population size

    # parameters to sweep over can be:
    #   tau_syn, f_in, w_in
#    cell_params = EIF_cond_exp_isfa_ista.default_parameters

    cell_params = {'cm':1.0, 'tau_refrac':1.0, 'v_thresh':-50.0, 'tau_syn_E':5.0, 'tau_syn_I':10.0, 'tau_m' : 10, 'v_reset' : -70, 'v_rest':-70}
#    cell_params['tau_syn_E'] = 3.
    neuron_model = 'IF_cond_exp'
    NS.create_neurons(n_cells, cell_params, neuron_model)

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

    w_input_exc = 5e-3
    f_max = 5000

    w_exc_noise = 5e-4
    f_exc_noise = 2000
    w_inh_noise = 5e-4
    f_inh_noise = 2000

    NS.connect_input(w_input_exc, f_max, mu=1., sigma=.1)
    NS.connect_noise(f_exc_noise, f_inh_noise, w_exc_noise, w_inh_noise)
    NS.run()

#    if pc_id == 0:
#        NS.analyze()
