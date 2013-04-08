import os
import sys
import simulation_parameters
import numpy as np
from pyNN.utility import get_script_args
from pyNN.errors import RecordingError
import pylab


class NeuronSimulator(object):

    def __init__(self, params):
        self.params = params
        setup(timestep=0.1,min_delay=0.1,max_delay=4.0, rng_seeds_seed=self.params['seed'])


    def create_neurons(self, n_pop, n_nrns_per_pop, param_name, param_value):

        print 'Creating %s neurons ... ' % (n_pop * n_nrns_per_pop)

        self.populations = []

        cell_params = IF_cond_exp.default_parameters

        for i in xrange(n_pop):
            cell_params[param_name] = param_value
            pop = Population(n_nrns_per_pop, IF_cond_exp, cell_params, label='exc_cells')
            self.populations.append(pop)


    def connect_noise(self):

        for pop in self.populations:
            for nrn in pop.all():
                spike_sourceE = create(SpikeSourcePoisson, {'rate' : self.params['f_exc_noise']})
                spike_sourceI = create(SpikeSourcePoisson, {'rate' : self.params['f_inh_noise']})
                connE = connect(spike_sourceE, nrn, weight=self.params['w_exc_noise'], synapse_type='excitatory',delay=1.0)
                connI = connect(spike_sourceI, nrn, weight=self.params['w_inh_noise'], synapse_type ='inhibitory',delay=1.0)


    def connect_input(self, pop_idx, nspikes_in, w=None):

        if w==None:
            w = self.params['w_input_exc']

        self.t_start = 200
        self.t_stim = 200
        self.t_stop = self.t_start + self.t_stim

        print 'Connect %d input spikes to population %d' % (nspikes_in, pop_idx)
        pop = self.populations[pop_idx]
        for nrn in pop.all():
            spike_times = (self.t_stop - self.t_start) * np.random.random(nspikes_in) + self.t_start
            spike_times.sort()
            spike_sourceE = create(SpikeSourceArray, {'spike_times': spike_times})
            connE = connect(spike_sourceE, nrn, weight=w, synapse_type='excitatory', delay=1.0)

            
    def run(self):
        print 'Run ... '

        for pop in self.populations:
            pop.record()
#            pop.record_v()

        self.t_sim = 600
        run(self.t_sim)
        for i_, pop in enumerate(self.populations):
            output_fn = self.params['exc_spiketimes_fn_merged'] + '%d.dat' % i_
            pop.printSpikes(output_fn)
#            output_fn = self.params['exc_volt_fn_base'] + '%d.v' % i_
#            print 'output_fn volt', output_fn
#            pop.print_v(output_fn, compatible_output=False)
        end()



if __name__ == '__main__':

    simulator_name  = 'nest'
    exec("from pyNN.%s import *" % simulator_name)

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


    import simulation_parameters
    ps = simulation_parameters.parameter_storage()
    params = ps.params
    NS = NeuronSimulator(params)

    n_x = 20            # values on x_axis (nspikes_in)
    n_trials = 10       # num neurons per population
    n_parameters = 3    # num curves

    n_min, n_max = 10, 500
    nspikes_in = np.linspace(n_min, n_max, n_x)

    param_name = 'tau_syn_E'
    param_value = 20.
    NS.create_neurons(n_x, n_trials, param_name, param_value)       # high b gives strong spike-frequency adaptation

    for pop_idx, n in enumerate(nspikes_in):
        NS.connect_input(pop_idx, n)

    NS.connect_noise()
    NS.run()

    def analyze():
        print 'Analyze ... '
        output_folder = params['tmp_folder']
        d = np.zeros((n_x, 2))

        output_fn_spikes = output_folder + 'spikes_vs_%s.dat' % (param_name)
        for i_, pop in enumerate(NS.populations):
            spike_fn = params['exc_spiketimes_fn_merged'] + '%d.dat' % i_
            spikes = np.loadtxt(spike_fn)
            gids = spikes[:, 1]
            nspikes = np.zeros(n_trials)
            for trial in xrange(n_trials):
                nspikes[trial] = (gids == trial).nonzero()[0].size

            d[i_, 0] = nspikes.mean()
            d[i_, 1] = nspikes.std()

        x_axis = nspikes_in
        color_list = ['b', 'g', 'r', 'y', 'c', 'm', (134./255., 0, 28./255.), '#00f80f', '#deff00', '#ff00e4', '#00ffe6']

        fig = pylab.figure()
        ax = fig.add_subplot(111)

        ax.errorbar(x_axis, d[:, 0], yerr=d[:, 1])

        ax.set_xlabel('n spikes in')
        ax.set_ylabel('Mean number of output spikes')
        ax.legend(loc='upper left')
#        ax.set_title(title)
        pylab.show()



    if pc_id == 0:
        analyze()
