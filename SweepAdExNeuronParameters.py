import os
import sys
import simulation_parameters
import numpy as np
from pyNN.utility import get_script_args
from pyNN.errors import RecordingError
import pylab


class NeuronSimulator(object):

    def __init__(self, parameter_storage):

        self.ps = parameter_storage
#        self.output_folder = 'AdExParameterSweeps/'
#        if not os.path.exists(self.output_folder):
#            os.mkdir(self.output_folder)
#        self.ps.set_filenames(self.output_folder)
        self.params = self.ps.params
        setup(timestep=0.1,min_delay=0.1,max_delay=4.0, rng_seeds_seed=self.params['seed'])


    def create_neurons(self, n_values, n_trials, param_name, p_min, p_max):
        """
        For each param one population of neurons is created with n_trials neurons.
        The behavior is thus averaged over n_trials cells ( = n_trials).
        There is also one control group of IF_cond_exp neurons
        """
        self.n_trials = n_trials
        self.param_name = param_name
        self.n_values = n_values

        print 'Creating %s neurons ... ' % (n_values * n_trials + n_trials)
        self.pop_control = Population(n_trials, IF_cond_exp, self.params['cell_params_exc'], label='control')
        n_trials = min(max(n_trials, 2), n_trials)

        param_range = np.linspace(p_min, p_max, n_values)
        print '%s parameter range\n' % param_name, '\t', param_range
        self.pop_dict = {}
        self.param_dict = {}

        cell_params = EIF_cond_exp_isfa_ista.default_parameters

        for i in xrange(n_values):

            value = param_range[i]
#            self.param_dict[] = {
            cell_params[param_name] = value
            pop = Population(n_trials, EIF_cond_exp_isfa_ista, cell_params, label='exc_cells')
            self.pop_dict[i] = {'param_name' : param_name, 'value' : value, 'pop' : pop}


    def connect_noise(self):

        for i_, key in enumerate(self.pop_dict.keys()):
            pop = self.pop_dict[key]['pop']
            for nrn in pop.all():
                
                spike_sourceE = create(SpikeSourcePoisson, {'rate' : self.params['f_exc_noise']})
                spike_sourceI = create(SpikeSourcePoisson, {'rate' : self.params['f_inh_noise']})
                connE = connect(spike_sourceE, nrn, weight=self.params['w_exc_noise'], synapse_type='excitatory',delay=1.0)
                connI = connect(spike_sourceI, nrn, weight=self.params['w_inh_noise'], synapse_type ='inhibitory',delay=1.0)

        for nrn in self.pop_control.all():
            spike_sourceE = create(SpikeSourcePoisson, {'rate' : self.params['f_exc_noise']})
            spike_sourceI = create(SpikeSourcePoisson, {'rate' : self.params['f_inh_noise']})
            connE = connect(spike_sourceE, nrn, weight=self.params['w_exc_noise'], synapse_type='excitatory',delay=1.0)
            connI = connect(spike_sourceI, nrn, weight=self.params['w_inh_noise'], synapse_type ='inhibitory',delay=1.0)

    def connect_input(self, gid):
        
        for i_, key in enumerate(self.pop_dict.keys()):
            pop = self.pop_dict[key]['pop']
            for nrn in pop.all():

                spike_times = np.load(self.params['input_st_fn_base'] + str(gid) + '.npy')
                spike_sourceE = create(SpikeSourceArray, {'spike_times': spike_times})
                connE = connect(spike_sourceE, nrn, weight=self.params['w_input_exc'], synapse_type='excitatory', delay=1.0)

        # connect input to the control group
        for nrn in self.pop_control.all():
            connE = connect(spike_sourceE, nrn, weight=self.params['w_input_exc'], synapse_type='excitatory', delay=1.0)

            
    def run(self):
        print 'Run ... '

        self.pop_control.record()
        self.pop_control.record_v()
        for i_, key in enumerate(self.pop_dict.keys()):
            pop = self.pop_dict[key]['pop']
            pop.record()
#            pop.record_v()

        self.pop_control.record()
#        self.pop_control.record_v()

        run(self.params['t_sim'])
        for i_, key in enumerate(self.pop_dict.keys()):
            pop = self.pop_dict[key]['pop']
            output_fn = self.params['exc_spiketimes_fn_merged'] + '%d.dat' % i_
            print 'output_fn spikes', output_fn
            pop.printSpikes(output_fn)
#            output_fn = self.params['exc_volt_fn_base'] + '%d.v' % i_
#            print 'output_fn volt', output_fn
#            pop.print_v(output_fn, compatible_output=False)

        output_fn = self.params['exc_spiketimes_fn_merged'] + 'control.dat'
        self.pop_control.printSpikes(output_fn)
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



"""
cm          0.281   nF  Capacity of the membrane
tau_refrac  0.0     ms  Duration of refractory period
v_spike     0.0     mV  Spike detection threshold
v_reset     -70.6   mV  Reset value for membrane potential after a spike
v_rest      -70.6   mV  Resting membrane potential (Leak reversal potential)
tau_m       9.3667  ms  Membrane time constant
i_offset    0.0     nA  Offset current
a           4.0     nS  Subthreshold adaptation conductance
b           0.0805  nA  Spike-triggered adaptation
delta_T     2.0     mV  Slope factor
tau_w       144.0   ms  Adaptation time constant
v_thresh    -50.4   mV  Spike initiation threshold
e_rev_E     0.0     mV  Excitatory reversal potential
tau_syn_E   5.0     ms  Decay time constant of excitatory synaptic conductance
e_rev_I     -80.0   mV  Inhibitory reversal potential
tau_syn_I   5.0     ms  Decay time constant of the inhibitory synaptic conductance
"""

if __name__ == '__main__':

    simulator_name  = 'nest'
    exec("from pyNN.%s import *" % simulator_name)

    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    NS = NeuronSimulator(network_params)

    n_values = 9
    n_trials = 20

#    NS.create_neurons(n_values, n_trials, 'a', 0.02, 0.4)      # high a gives sub-threshold oscillations, medium can give overshoots to current pulses
    NS.create_neurons(n_values, n_trials, 'b', 0.1, 1.0)       # high b gives strong spike-frequency adaptation
#    NS.create_neurons(n_values, n_trials, 'tau_w', 20, 144)    # 

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

    try:
        gid = int(sys.argv[1])
    except:
        if os.path.exists(network_params.params['gids_to_record_fn']):
            gid = int(np.loadtxt(network_params.params['gids_to_record_fn'])[0])
        else:
            print 'No gid to load spike file for... :('

    print 'Plotting gid', gid

    if n_proc > 1:
        NS.connect_input(gid)
        NS.connect_noise()
        NS.run()

    if pc_id == 0:
        NS.analyze()
