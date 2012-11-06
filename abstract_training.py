import simulation_parameters
import numpy as np
import utils
import Bcpnn
import os
import sys
import time
import random

class AbstractTrainer(object):

    def __init__(self, params, n_speeds, n_cycles, n_stim, comm=None):
        self.params = params
        self.comm = comm
        self.n_stim = n_stim
        self.n_cycles = n_cycles
        self.n_speeds = n_speeds
        self.selected_conns = None
        self.n_time_steps = self.params['t_sim'] / self.params['dt_rate']

        # distribute units among processors
        if comm != None:
            self.pc_id, self.n_proc = comm.rank, comm.size
            my_units = utils.distribute_n(params['n_exc'], self.n_proc, self.pc_id)
            self.my_units = range(my_units[0], my_units[1])
        else:
            self.my_units = range(self.params['n_exc'])
            self.pc_id, self.n_proc = 0, 1

        try:
            self.tuning_prop = np.loadtxt(self.params['tuning_prop_means_fn'])
        except:
            print 'Tuning properties file not found: %s\n Will create new ones' % self.params['tuning_prop_means_fn']
            self.tuning_prop = utils.set_tuning_prop(self.params, mode='hexgrid', v_max=self.params['v_max'])
            np.savetxt(self.params['tuning_prop_means_fn'], self.tuning_prop)

        if comm != None:
            comm.barrier()

        self.initial_value = 1e-2 # should be around 1 / n_units per HC, i.e. 1. / (params['N_theta'] * params['N_V']
        tau_p = self.params['t_sim'] * self.n_stim * self.n_cycles # tau_p should be in the order of t_stimulus * n_iterations * n_cycles
        tau_pij = tau_p
        self.tau_dict = {'tau_zi' : 50.,    'tau_zj' : 5., 
                        'tau_ei' : 100.,   'tau_ej' : 100., 'tau_eij' : 100.,
                        'tau_pi' : tau_p,  'tau_pj' : tau_p, 'tau_pij' : tau_pij,
                        }
        self.eps = .1 * self.initial_value
        self.normalize = True # normalize input within a 'hypercolumn'

        all_conns = []
        # distribute connections among processors
        for i in xrange(params['n_exc']):
            for j in xrange(params['n_exc']):
                if i != j:
                    all_conns.append((i, j))
        self.my_conns = utils.distribute_list(all_conns, n_proc, pc_id)
#        self.my_conns = np.array(self.my_conns)

 
    def create_stimuli(self, random_order=False, test_stim=False):

        distance_from_center = 0.5
        center = (0.5, 0.5)
        thetas = np.linspace(np.pi, 3*np.pi, n_stim, endpoint=False)
        v_default = np.sqrt(self.params['motion_params'][2]**2 + self.params['motion_params'][3]**2)

        seed = 0
        np.random.seed(seed)
        random.seed(seed)

        sigma_theta = 2 * np.pi * 0.05
        random_rotation = sigma_theta * (np.random.rand(self.n_cycles * self.n_stim * self.n_speeds) - .5 * np.ones(self.n_cycles * self.n_stim * self.n_speeds))
        v_min, v_max = 0.3, 0.4
        speeds = np.linspace(v_min, v_max, self.n_speeds)

        output_file = open(self.params['parameters_folder'] + 'input_params.txt', 'w')
        input_str = '#x0\ty0\tu0\tv0\n'

        iteration = 0
        for speed_cycle in xrange(self.n_speeds):
            for cycle in xrange(self.n_cycles):
                stimulus_order = range(self.n_stim)
                if random_order == True:
                    random.shuffle(stimulus_order)

                for stim in stimulus_order:
                    self.iteration = iteration
                    x0 = distance_from_center * np.cos(thetas[stim] + random_rotation[iteration]) + center[0]
                    y0 = distance_from_center * np.sin(thetas[stim] + random_rotation[iteration]) + center[1]
                    u0 = np.cos(np.pi + thetas[stim] + random_rotation[iteration]) * speeds[speed_cycle]#v_default
                    v0 = np.sin(np.pi + thetas[stim] + random_rotation[iteration]) * speeds[speed_cycle]#v_default
                    self.params['motion_params'] = (x0, y0, u0, v0)
                    input_str += '%.4e\t%.4e\t%.4e\t%.4e\n' % (x0, y0, u0, v0)

                    self.training_input_folder = "%sTrainingInput_%d/" % (self.params['folder_name'], iteration)
                    print 'Writing input to %s' % (self.training_input_folder)
                    if not os.path.exists(self.training_input_folder) and self.pc_id == 0:
                        mkdir = 'mkdir %s' % self.training_input_folder
                        print mkdir
                        os.system(mkdir)
                    if self.comm != None:
                        self.comm.barrier()

                    if test_stim:
                        self.create_input_vectors_blanking(t_blank=(0.4, 0.6), normalize=self.normalize)
                    else:
                        self.create_input_vectors(normalize=self.normalize)
                    if self.comm != None:
                        self.comm.barrier()
                    iteration += 1
        output_file.write(input_str)
        output_file.close()




    def create_input_vectors(self, normalize=True):
        output_fn_base = self.training_input_folder + self.params['abstract_input_fn_base']
        n_cells = len(self.my_units)
        dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process 
        time = np.arange(0, params['t_sim'], dt)
        L_input = np.zeros((n_cells, time.shape[0]))
        for i_time, time_ in enumerate(time):
            if (i_time % 100 == 0):
                print "t:", time_
            L_input[:, i_time] = utils.get_input(self.tuning_prop[self.my_units, :], params, time_/params['t_stimulus'])

        for i_, unit in enumerate(self.my_units):
            output_fn = output_fn_base + str(unit) + '.dat'
            np.savetxt(output_fn, L_input[i_, :])

        if self.comm != None:
            self.comm.barrier()

        if normalize:
            self.normalize_input(output_fn_base)

        if self.comm != None:
            self.comm.barrier()


    def create_input_vectors_blanking(self, t_blank=(0.25, 0.75), normalize=True):
        """
        Stimulus is calculated only until
            t_stop * self.params['t_stimulus']
        """
        output_fn_base = self.training_input_folder + self.params['abstract_input_fn_base']
        n_cells = len(self.my_units)
        dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process 
        time = np.arange(0, self.params['t_sim'], dt) # only stimulate until 
        L_input = np.zeros((n_cells, self.params['t_sim'] / dt))

        blank_idx = np.arange(time.shape[0] * t_blank[0], time.shape[0] * t_blank[1])

        for i_time, time_ in enumerate(time):
            if (i_time % 100 == 0):
                print "t:", time_
            L_input[:, i_time] = utils.get_input(self.tuning_prop[self.my_units, :], self.params, time_/self.params['t_stimulus'])

        for i in blank_idx:
            L_input[:, i] = 0.

        for i_, unit in enumerate(self.my_units):
            output_fn = output_fn_base + str(unit) + '.dat'
            np.savetxt(output_fn, L_input[i_, :])

        if self.comm != None:
            self.comm.barrier()

        if normalize:
            self.normalize_input(output_fn_base)

        if self.comm != None:
            self.comm.barrier()



    def normalize_input(self, fn_base):

        if pc_id == 0:
            print 'normalize_input for', fn_base
            L_input = np.zeros((self.n_time_steps, self.params['n_exc']))
            n_hc = self.params['N_RF_X']*self.params['N_RF_Y']
            n_cells_per_hc = self.params['N_theta'] * self.params['N_V']
            n_cells = params['n_exc']
            assert (n_cells == n_hc * n_cells_per_hc)
            for cell in xrange(n_cells):
                fn = fn_base + str(cell) + '.dat'
#                print 'debug', fn
                L_input[:, cell] = np.loadtxt(fn)

            for t in xrange(int(self.n_time_steps)):
                for hc in xrange(n_hc):
                    idx0 = hc * n_cells_per_hc
                    idx1 = (hc + 1) * n_cells_per_hc
                    s = L_input[t, idx0:idx1].sum()
                    if s > 1.0:
                        L_input[t, idx0:idx1] /= s

            for hc in xrange(n_hc):
                idx0 = hc * n_cells_per_hc
                idx1 = (hc + 1) * n_cells_per_hc
                for i in xrange(n_cells_per_hc):
                    cell = hc * n_cells_per_hc + i
                    output_fn = fn_base + str(cell) + '.dat'
                    np.savetxt(output_fn, L_input[:, cell])
#                for t in xrange(int(self.n_time_steps)):
#                    s = L_input[t, idx0:idx1].sum()
#                    print 'debug t s', t, s, s <= 1.0
#                    assert (s <= 1.0), '%.20e %d' % (s, s <= 1.0)

            all_output_fn = params['activity_folder'] + 'input_%d.dat' % (self.iteration)
            print 'Normalized input is written to:', all_output_fn
            np.savetxt(all_output_fn, L_input)

        if self.comm != None:
            self.comm.barrier()




    def train(self):

        self.pi_init = self.initial_value * np.ones(params['n_exc'])
        self.pj_init = self.initial_value * np.ones(params['n_exc'])
        self.pij_init = self.initial_value ** 2 * np.ones((params['n_exc'], params['n_exc']))
        self.wij_init = np.zeros((params['n_exc'], params['n_exc']))
        self.bias_init = np.log(self.initial_value) * np.ones(params['n_exc'])

        comp_times = []
        self.iteration = 0
        for speed in xrange(self.n_speeds):
            for cycle in xrange(self.n_cycles):
                print '\nCYCLE %d\n' % (cycle)
                for stim in xrange(self.n_stim):
                    t0= time.time()
                    # M A K E    D I R E C T O R Y 
                    training_folder = '%sTrainingResults_%d/' % (self.params['folder_name'], self.iteration)
                    if not os.path.exists(training_folder) and self.pc_id == 0:
                        mkdir = 'mkdir %s' % training_folder
                        print mkdir
                        os.system(mkdir)
                    if self.comm != None:
                        self.comm.barrier()

                    # C O M P U T E    
                    self.compute_my_pijs(training_folder, self.iteration)

                    t_comp = time.time() - t0
                    comp_times.append(t_comp)
                    print 'Computation time for training %d: %d sec = %.1f min' % (self.iteration, t_comp, t_comp / 60.)
                    if self.comm != None:
                        self.comm.barrier()
                    self.iteration += 1

        total_time = 0.
        for t in comp_times:
            total_time += t
        print 'Total computation time for %d training iterations: %d sec = %.1f min' % (self.n_stim * self.n_cycles, total_time, total_time/ 60.)

    def compute_my_pijs_new(self, training_folder, iteration=0):
        pre_ids = np.unique(self.my_conns[:, 0])
        post_ids = np.unique(self.my_conns[:, 0])
        tau_dict = self.tau_dict
        zi_traces = np.zeros((self.n_time_steps, pre_ids.size))
        zj_traces = np.zeros((self.n_time_steps, pre_ids.size))
        ei_traces = np.zeros((self.n_time_steps, pre_ids.size))
        ej_traces = np.zeros((self.n_time_steps, pre_ids.size))
        pi_traces = np.zeros((self.n_time_steps, pre_ids.size))
        pj_traces = np.zeros((self.n_time_steps, pre_ids.size))
        bias_values = np.zeros(post_ids.size)

    def compute_my_pijs(self, training_folder, iteration=0):

        conns = self.my_conns
        tau_dict = self.tau_dict
        dt = self.params['dt_rate'] # [ms] time step for the non-homogenous Poisson process 
        print 'pc_id computes pijs for %d connections' % (len(conns))
        my_traces_pre = {}
        my_traces_post = {}
        p_i = {}
        p_j = {}

        pi_string = '#GID\tp_i\n'
        pj_string = '#GID\tp_j\n'
        pij_string = '#pre_id\tpost_id\tp_ij\n'
        wij_string = '#pre_id\tpost_id\tpij[-1]\tw_ij[-1]\tbias\n'
        bias_string = '#GID\tp_j\n'
        z_init = self.initial_value
        e_init = self.initial_value
        eij_init = self.initial_value ** 2 * np.ones((params['n_exc'], params['n_exc']))

        self.training_input_folder = "%sTrainingInput_%d/" % (self.params['folder_name'], iteration)
        input_fn_base = self.training_input_folder + self.params['abstract_input_fn_base']

        for i in xrange(len(conns)):
            if (i % 1000) == 0:
                print "Pc %d conn: \t%d / %d\t%.4f percent complete; Stimulus iteration: %d" % (pc_id, i, len(conns), i * 100./len(conns), self.iteration)
            pre_id = conns[i][0]
            post_id = conns[i][1]
            if my_traces_pre.has_key(pre_id):
                (zi, ei, pi) = my_traces_pre[pre_id]
            else:
                pre_trace = np.loadtxt(input_fn_base + str(pre_id) + '.dat')
                zi, ei, pi = Bcpnn.compute_traces(pre_trace, tau_dict['tau_zi'], tau_dict['tau_ei'], tau_dict['tau_pi'], dt=self.params['dt_rate'], eps=self.eps, initial_value=(z_init, e_init, self.pi_init[pre_id]))
                my_traces_pre[pre_id] = (zi, ei, pi)
                self.pi_init[pre_id] = pi[-1]
                pi_string += '%d\t%.6e\n' % (pre_id, pi[-1])

            if my_traces_post.has_key(post_id):
                (zj, ej, pj) = my_traces_post[post_id]
            else: 
                post_trace = np.loadtxt(input_fn_base  + str(post_id) + '.dat')
                zj, ej, pj = Bcpnn.compute_traces(post_trace, tau_dict['tau_zj'], tau_dict['tau_ej'], tau_dict['tau_pj'], dt=self.params['dt_rate'], eps=self.eps, initial_value=(z_init, e_init, self.pj_init[post_id]))
                my_traces_post[post_id] = (zj, ej, pj)
                self.pj_init[pre_id] = pj[-1]
                pj_string += '%d\t%.6e\n' % (post_id, pj[-1])
                bias_string += '%d\t%.6e\n' % (post_id, np.log(pj[-1]))

            pij, wij, bias = Bcpnn.compute_pij(zi, zj, pi, pj, tau_dict['tau_eij'], tau_dict['tau_pij'], dt=self.params['dt_rate'], \
                    initial_values=(eij_init[pre_id, post_id], self.pij_init[pre_id, post_id], self.wij_init[pre_id, post_id], self.bias_init[post_id]))
            self.pij_init[pre_id, post_id] = pij
            self.wij_init[pre_id, post_id] = wij
            self.bias_init[post_id] = bias
            wij_string += '%d\t%d\t%.6e\t%.6e\t%.6e\n' % (pre_id, post_id, pij, wij, bias)
            pij_string += '%d\t%d\t%.6e\n' % (pre_id, post_id, pij)

        # write selected traces to files
        for c in self.selected_conns:
            if c in self.my_conns:
                pre_id, post_id = c[0], c[1]

                print 'Proc %d prints BCPNN pre-traces for cell %d:' % (self.pc_id, pre_id)
                np.savetxt(self.params['bcpnntrace_folder'] + 'zi_%d_%d.dat' % (iteration, pre_id), my_traces_pre[pre_id][0])
                np.savetxt(self.params['bcpnntrace_folder'] + 'ei_%d_%d.dat' % (iteration, pre_id), my_traces_pre[pre_id][1])
                np.savetxt(self.params['bcpnntrace_folder'] + 'pi_%d_%d.dat' % (iteration, pre_id), my_traces_pre[pre_id][2])
                np.savetxt(self.params['bcpnntrace_folder'] + 'zj_%d_%d.dat' % (iteration, post_id), my_traces_post[post_id][0])
                np.savetxt(self.params['bcpnntrace_folder'] + 'ej_%d_%d.dat' % (iteration, post_id), my_traces_post[post_id][1])
                np.savetxt(self.params['bcpnntrace_folder'] + 'pj_%d_%d.dat' % (iteration, post_id), my_traces_post[post_id][2])

        if self.comm != None:
            self.comm.barrier()

        # for selected connections compute the eij, pij, weight and bias traces
        if self.pc_id == 0:
            for c in self.selected_conns:
                pre_id, post_id = c[0], c[1]
                zi = np.loadtxt(self.params['bcpnntrace_folder'] + 'zi_%d_%d.dat' % (iteration, pre_id))
                pi = np.loadtxt(self.params['bcpnntrace_folder'] + 'pi_%d_%d.dat' % (iteration, pre_id))
                zj = np.loadtxt(self.params['bcpnntrace_folder'] + 'zj_%d_%d.dat' % (iteration, post_id))
                pj = np.loadtxt(self.params['bcpnntrace_folder'] + 'pj_%d_%d.dat' % (iteration, post_id))
                wij, bias, pij, eij = Bcpnn.compute_pij(zi, zj, pi, pj, self.tau_dict['tau_eij'], self.tau_dict['tau_pij'], dt=self.params['dt_rate'], get_traces=True, 
                    initial_values=(eij_init[pre_id, post_id], self.pij_init[pre_id, post_id], self.wij_init[pre_id, post_id], self.bias_init[post_id]))
                np.savetxt(self.params['bcpnntrace_folder'] + 'wij_%d_%d_%d.dat' % (iteration, pre_id, post_id), wij)
                np.savetxt(self.params['bcpnntrace_folder'] + 'bias_%d_%d_%d.dat' % (iteration, pre_id, post_id), bias)
                np.savetxt(self.params['bcpnntrace_folder'] + 'eij_%d_%d_%d.dat' % (iteration, pre_id, post_id), eij)
                np.savetxt(self.params['bcpnntrace_folder'] + 'pij_%d_%d_%d.dat' % (iteration, pre_id, post_id), pij)


        pi_output_fn = training_folder + 'pi_%d.dat' % (self.pc_id)
        pi_f = open(pi_output_fn, 'w')
        pi_f.write(pi_string)
        pi_f.close()

        pj_output_fn = training_folder + 'pj_%d.dat' % (self.pc_id)
        pj_f = open(pj_output_fn, 'w')
        pj_f.write(pj_string)
        pj_f.close()

        pij_output_fn = training_folder + 'pij_%d.dat' % (self.pc_id)
        pij_f = open(pij_output_fn, 'w')
        pij_f.write(pij_string)
        pij_f.close()

        wij_fn = training_folder + 'wij_%d.dat' % (self.pc_id)
        print 'Writing w_ij output to:', wij_fn
        f = file(wij_fn, 'w')
        f.write(wij_string)
        f.close()

        bias_fn = training_folder + 'bias_%d.dat' % (self.pc_id)
        print 'Writing bias output to:', bias_fn
        f = file(bias_fn, 'w')
        f.write(bias_string)
        f.close()

        if self.comm != None:
            self.comm.barrier()
        



if __name__ == '__main__':

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


    PS = simulation_parameters.parameter_storage()
    params = PS.params
    if pc_id == 0:
        PS.create_folders()
        PS.write_parameters_to_file()
    if comm != None:
        comm.barrier()

    n_speeds = 1
    n_cycles = 5
    n_stim = 8

    AT = AbstractTrainer(params, n_speeds, n_cycles, n_stim, comm)
#    cells_to_record = [85, 161, 71, 339, 275]
    cells_to_record = [85, 161, 71, 339, 275]
    selected_connections = []
    for src in cells_to_record:
        for tgt in cells_to_record:
            if src != tgt:
                selected_connections.append((src, tgt))
    AT.selected_conns = selected_connections
                        
    AT.create_stimuli(random_order=True, test_stim=False)
    AT.train()



