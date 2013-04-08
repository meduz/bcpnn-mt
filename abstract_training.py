import simulation_parameters
import numpy as np
import utils
import Bcpnn
import os
import sys
import time
import random
import CreateStimuli

class AbstractTrainer(object):

    def __init__(self, params, comm=None):
        self.params = params
        self.comm = comm
        self.n_speeds = params['n_speeds']
        self.n_cycles = params['n_cycles']
        self.n_directions = params['n_theta']
        self.n_iterations_total = self.params['n_theta'] * self.params['n_speeds'] * self.params['n_cycles'] * self.params['n_stim_per_direction']
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
        self.eps = .1 * self.initial_value
        self.normalize = False# normalize input within a 'hypercolumn'

        all_conns = []
        # distribute connections among processors
        for i in xrange(params['n_exc']):
            for j in xrange(params['n_exc']):
                if i != j:
                    all_conns.append((i, j))
        self.my_conns = utils.distribute_list(all_conns, n_proc, pc_id)

        # setup data structures
        self.my_conns = np.array(self.my_conns)
        np.savetxt('delme_my_conns_%d.txt' % self.pc_id, self.my_conns, fmt='%d\t%d')
        self.pre_ids = np.unique(self.my_conns[:, 0])
        self.post_ids = np.unique(self.my_conns[:, 1])
        self.gid_idx_map_pre = {}
        self.gid_idx_map_post = {}
        for i in xrange(self.pre_ids.size):
            self.gid_idx_map_pre[self.pre_ids[i]] = i
        for i in xrange(self.post_ids.size):
            self.gid_idx_map_post[self.post_ids[i]] = i
        self.my_selected_conns = []


    def set_selected_connections(self, conn_list):
        for c in conn_list:
            pre_id, post_id = c[0], c[1]
            if ((pre_id in self.pre_ids) and (post_id in self.post_ids)):
#            if c in self.my_conns:
                self.my_selected_conns.append((pre_id, post_id))
                print 'Pc_id %d gets %d - %d as seleceted connection' % (self.pc_id, c[0], c[1])


    def create_stimuli(self, random_order=False, test_stim=False):

        mp = np.zeros((self.n_iterations_total, 4))
        n_iterations_per_cycle = self.params['n_theta'] * self.params['n_speeds'] * self.params['n_stim_per_direction']
            
        CS = CreateStimuli.CreateStimuli(self.params, random_order)
        all_speeds, all_starting_pos, all_thetas = CS.get_motion_params(random_order)

        i = 0
        for cycle in xrange(self.params['n_cycles']):
            for stim in xrange(n_iterations_per_cycle):
                self.iteration = stim
                print 'Generating input for iteration %d / %d' % (i, self.n_iterations_total)
                x0, y0 = all_starting_pos[stim, :]
                u0 = np.cos(all_thetas[stim]) * all_speeds[stim]
                v0 = - np.sin(all_thetas[stim]) * all_speeds[stim]
                mp[i, :] = x0, y0, u0, v0
                i += 1
        print 'Saving input params to:', self.params['parameters_folder'] + 'input_params.txt'
        np.savetxt(self.params['parameters_folder'] + 'input_params.txt', mp)


        stim = 0
        for cycle in xrange(self.params['n_cycles']):
            for i in xrange(n_iterations_per_cycle):
                self.iteration = stim
                print 'Generating input for iteration %d / %d' % (stim, self.n_iterations_total)
                x0, y0 = all_starting_pos[i, :]
                u0 = np.cos(all_thetas[i]) * all_speeds[i]
                v0 = - np.sin(all_thetas[i]) * all_speeds[i]
                self.params['motion_params'] = (x0, y0, u0, v0)
                self.training_input_folder = "%sTrainingInput_%d/" % (self.params['folder_name'], stim)
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
                stim += 1


 
    def create_stimuli_going_through_center(self, random_order=False, test_stim=False):
        """
        This function is deprecated and produces the wrong number of stimuli.
        It doesn't take n_stim_per_direction into account.
        """



        distance_from_center = 0.5
        center = (0.5, 0.5)
        thetas = np.linspace(np.pi, 3*np.pi, self.n_directions, endpoint=False)
        v_default = np.sqrt(self.params['motion_params'][2]**2 + self.params['motion_params'][3]**2)

        seed = 0
        np.random.seed(seed)
        random.seed(seed)

        sigma_theta = self.params['sigma_theta_training']
        random_rotation = sigma_theta * (np.random.rand(self.n_cycles * self.n_directions * self.n_speeds) - .5 * np.ones(self.n_cycles * self.n_directions * self.n_speeds))
        v_min, v_max = 0.3, 0.6
        speeds = np.linspace(v_min, v_max, self.n_speeds)

        output_file = open(self.params['parameters_folder'] + 'input_params.txt', 'w')
        input_str = '#x0\ty0\tu0\tv0\n'

        iteration = 0
        for speed_cycle in xrange(self.n_speeds):
            for cycle in xrange(self.n_cycles):
                stimulus_order = range(self.n_directions)
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
            L_input[:, i_time] = utils.get_input(self.tuning_prop[self.my_units, :], params, time_/params['t_stimulus'], motion_params=self.params['motion_params'])

        for i_, unit in enumerate(self.my_units):
            output_fn = output_fn_base + str(unit) + '.dat'
            np.savetxt(output_fn, L_input[i_, :])

        if pc_id == 0:
            full_stim_input = '%sANNActivity/input_%d.dat' % (self.params['folder_name'], self.iteration)
            print 'Saving input for stim %d to %s' % (self.iteration, full_stim_input)
            np.savetxt(full_stim_input, L_input)

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
            L_input[:, i_time] = utils.get_input(self.tuning_prop[self.my_units, :], self.params, time_/self.params['t_stimulus'], motion_params=self.params['motion_params'])
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
            input_scaling_factor = self.params['abstract_input_scaling_factor']
            print 'normalize_input for', fn_base
            L_input = np.zeros((self.n_time_steps, self.params['n_exc']))
            n_hc = self.params['N_RF_X']*self.params['N_RF_Y']
            n_cells_per_hc = self.params['N_theta'] * self.params['N_V']
            n_cells = params['n_exc']
            assert (n_cells == n_hc * n_cells_per_hc)
            for cell in xrange(n_cells):
                fn = fn_base + str(cell) + '.dat'
                L_input[:, cell] = np.loadtxt(fn)
                L_input[:, cell] *= input_scaling_factor

            for t in xrange(int(self.n_time_steps)):
                for hc in xrange(n_hc):
                    idx0 = hc * n_cells_per_hc
                    idx1 = (hc + 1) * n_cells_per_hc
                    s = L_input[t, idx0:idx1].sum()
                    if s > 1:
#                        L_input[t, idx0:idx1] = np.exp(L_input[t, idx0:idx1]) / np.exp(L_input[t, idx0:idx1]).sum()
                        L_input[t, idx0:idx1] /= L_input[t, idx0:idx1].sum()

            for cell in xrange(n_cells):
                output_fn = fn_base + str(cell) + '.dat'
                np.savetxt(output_fn, L_input[:, cell])

            all_output_fn = params['activity_folder'] + 'input_%d.dat' % (self.iteration)
            print 'Normalized input is written to:', all_output_fn
            np.savetxt(all_output_fn, L_input)

        if self.comm != None:
            self.comm.barrier()




    def train(self):

        self.zi_init = self.initial_value * np.ones(params['n_exc'])
        self.zj_init = self.initial_value * np.ones(params['n_exc'])
        self.ei_init = self.initial_value * np.ones(params['n_exc'])
        self.ej_init = self.initial_value * np.ones(params['n_exc'])
        self.pi_init = self.initial_value * np.ones(params['n_exc'])
        self.pj_init = self.initial_value * np.ones(params['n_exc'])
        self.eij_init = self.initial_value ** 2 * np.ones((params['n_exc'], params['n_exc']))
        self.pij_init = self.initial_value ** 2 * np.ones((params['n_exc'], params['n_exc']))
        self.wij_init = np.zeros((params['n_exc'], params['n_exc']))
        self.bias_init = np.log(self.initial_value) * np.ones(params['n_exc'])

        comp_times = []
        for iteration in xrange(self.n_iterations_total):
            self.iteration = iteration
            t0= time.time()
            # M A K E    D I R E C T O R Y 
            self.training_output_folder = '%sTrainingResults_%d/' % (self.params['folder_name'], self.iteration)
            self.training_input_folder = "%sTrainingInput_%d/" % (self.params['folder_name'], self.iteration)
            if not os.path.exists(self.training_output_folder) and self.pc_id == 0:
                mkdir = 'mkdir %s' % self.training_output_folder
                print mkdir
                os.system(mkdir)
            if self.comm != None:
                self.comm.barrier()

            # C O M P U T E    
            self.compute_my_pijs()

            t_comp = time.time() - t0
            comp_times.append(t_comp)
            print 'Computation time for training %d: %d sec = %.1f min' % (self.iteration, t_comp, t_comp / 60.)
            if self.comm != None:
                self.comm.barrier()

        total_time = 0.
        for t in comp_times:
            total_time += t
        print 'Total computation time for %d training iterations: %d sec = %.1f min' % (self.n_iterations_total, total_time, total_time/ 60.)

    def compute_my_pijs(self):

        pre_traces_computed = np.zeros(params['n_exc'], dtype=np.bool)
        post_traces_computed = np.zeros(params['n_exc'], dtype=np.bool)

        tau_dict = self.params['tau_dict']
        zi_traces = self.initial_value * np.ones((self.n_time_steps, self.pre_ids.size), dtype=np.double)
        zj_traces = self.initial_value * np.ones((self.n_time_steps, self.post_ids.size), dtype=np.double)
        ei_traces = self.initial_value * np.ones((self.n_time_steps, self.pre_ids.size), dtype=np.double)
        ej_traces = self.initial_value * np.ones((self.n_time_steps, self.post_ids.size), dtype=np.double)
        pi_traces = self.initial_value * np.ones((self.n_time_steps, self.pre_ids.size), dtype=np.double)
        pj_traces = self.initial_value * np.ones((self.n_time_steps, self.post_ids.size), dtype=np.double)
        eij_trace = self.initial_value ** 2 * np.ones(self.n_time_steps, dtype=np.double)
        pij_trace = self.initial_value ** 2 * np.ones(self.n_time_steps, dtype=np.double)
        wij_trace = np.zeros(self.n_time_steps, dtype=np.double)
        bias_trace = np.log(self.initial_value) * np.ones(self.n_time_steps, dtype=np.double)
        input_fn_base = self.training_input_folder + self.params['abstract_input_fn_base']

        self.my_wijs = np.zeros((self.my_conns[:, 0].size, 4), dtype=np.double) # array for wij and pij
        self.my_bias = np.zeros((self.post_ids.size, 2), dtype=np.double) # array for wij and pij
        bias_idx = 0
        for i in xrange(self.my_conns[:, 0].size):
            if (i % 1000) == 0:
                print "Pc %d conn: \t%d / %d\t%.4f percent complete; Stimulus iteration: %d" % (pc_id, i, self.my_conns[:, 0].size, i * 100./self.my_conns[:, 0].size, self.iteration)
            pre_id = self.my_conns[i, 0]
            post_id = self.my_conns[i, 1]
            if pre_traces_computed[pre_id]:
                idx = self.gid_idx_map_pre[pre_id]
                (zi, ei, pi) = zi_traces[:, idx], ei_traces[:, idx], pi_traces[:, idx]
            else:
                pre_trace = np.loadtxt(input_fn_base + str(pre_id) + '.dat')
                idx = self.gid_idx_map_pre[pre_id]
                zi_traces[0, idx] = self.zi_init[pre_id]
                ei_traces[0, idx] = self.ei_init[pre_id]
                pi_traces[0, idx] = self.pi_init[pre_id]
                Bcpnn.compute_traces_new(pre_trace, zi_traces[:, idx], ei_traces[:, idx], pi_traces[:, idx], \
                        tau_dict['tau_zi'], tau_dict['tau_ei'], tau_dict['tau_pi'], \
                        dt=self.params['dt_rate'], eps=self.eps)
                pre_traces_computed[pre_id] = True
                self.zi_init[pre_id] = zi_traces[-1, idx]
                self.ei_init[pre_id] = ei_traces[-1, idx]
                self.pi_init[pre_id] = pi_traces[-1, idx]

            if post_traces_computed[post_id]:
                idx = self.gid_idx_map_post[post_id]
                (zj, ej, pj) = zj_traces[:, idx], ej_traces[:, idx], pj_traces[:, idx]
            else:
                post_trace = np.loadtxt(input_fn_base + str(post_id) + '.dat')
                idx = self.gid_idx_map_post[post_id]
                zj_traces[0, idx] = self.zj_init[post_id]
                ej_traces[0, idx] = self.ej_init[post_id]
                pj_traces[0, idx] = self.pj_init[post_id]
                Bcpnn.compute_traces_new(post_trace, zj_traces[:, idx], ej_traces[:, idx], pj_traces[:, idx], \
                        tau_dict['tau_zj'], tau_dict['tau_ej'], tau_dict['tau_pj'], \
                        dt=self.params['dt_rate'], eps=self.eps)
                post_traces_computed[post_id] = True
                self.zj_init[post_id] = zj_traces[-1, idx]
                self.ej_init[post_id] = ej_traces[-1, idx]
                self.pj_init[post_id] = pj_traces[-1, idx]
                self.my_bias[bias_idx, :] = post_id, np.log(pj_traces[-1, idx])
                bias_idx += 1

            idx_pre = self.gid_idx_map_pre[pre_id]
            idx_post = self.gid_idx_map_post[post_id]
            eij_trace[0] = self.eij_init[pre_id, post_id]
            pij_trace[0] = self.pij_init[pre_id, post_id]
            wij_trace[0] = self.wij_init[pre_id, post_id]
            bias_trace[0] = self.bias_init[post_id]

            if ((pre_id, post_id) in self.my_selected_conns):
                # write selected traces to files
#                print 'Proc %d prints BCPNN pre-traces for cell %d:' % (self.pc_id, pre_id)
                idx = self.gid_idx_map_pre[pre_id]
                np.savetxt(self.params['bcpnntrace_folder'] + 'zi_%d_%d.dat' % (self.iteration, pre_id), zi_traces[:, idx])
                np.savetxt(self.params['bcpnntrace_folder'] + 'ei_%d_%d.dat' % (self.iteration, pre_id), ei_traces[:, idx])
                np.savetxt(self.params['bcpnntrace_folder'] + 'pi_%d_%d.dat' % (self.iteration, pre_id), pi_traces[:, idx])
#                print 'Proc %d prints BCPNN post-traces for cell %d:' % (self.pc_id, post_id)
                idx = self.gid_idx_map_post[post_id]
                np.savetxt(self.params['bcpnntrace_folder'] + 'zj_%d_%d.dat' % (self.iteration, post_id), zj_traces[:, idx])
                np.savetxt(self.params['bcpnntrace_folder'] + 'ej_%d_%d.dat' % (self.iteration, post_id), ej_traces[:, idx])
                np.savetxt(self.params['bcpnntrace_folder'] + 'pj_%d_%d.dat' % (self.iteration, post_id), pj_traces[:, idx])
                wij, bias, pij, eij = Bcpnn.compute_pij_new(zi_traces[:, idx_pre], zj_traces[:, idx_post], pi_traces[:, idx_pre], pj_traces[:, idx_post], \
                                        eij_trace, pij_trace, wij_trace, bias_trace, \
                                        tau_dict['tau_eij'], tau_dict['tau_pij'], get_traces=True, dt=self.params['dt_rate'])
                np.savetxt(self.params['bcpnntrace_folder'] + 'wij_%d_%d_%d.dat' % (self.iteration, pre_id, post_id), wij)
                np.savetxt(self.params['bcpnntrace_folder'] + 'bias_%d_%d_%d.dat' % (self.iteration, pre_id, post_id), bias)
                np.savetxt(self.params['bcpnntrace_folder'] + 'eij_%d_%d_%d.dat' % (self.iteration, pre_id, post_id), eij)
                np.savetxt(self.params['bcpnntrace_folder'] + 'pij_%d_%d_%d.dat' % (self.iteration, pre_id, post_id), pij)
            else:
                Bcpnn.compute_pij_new(zi_traces[:, idx_pre], zj_traces[:, idx_post], pi_traces[:, idx_pre], pj_traces[:, idx_post], \
                        eij_trace, pij_trace, wij_trace, bias_trace, \
                        tau_dict['tau_eij'], tau_dict['tau_pij'], dt=self.params['dt_rate'])
                    
            # update the nr.0 value for the next stimulus
            self.eij_init[pre_id, post_id] = eij_trace[-1]
            self.pij_init[pre_id, post_id] = pij_trace[-1]
            self.wij_init[pre_id, post_id] = wij_trace[-1]
            self.bias_init[post_id] = bias_trace[-1]
            self.my_wijs[i, :] = pre_id, post_id, wij_trace[-1], pij_trace[-1]

        # store wijs and bias in the tmp folder
        np.savetxt(self.params['tmp_folder'] + 'wij_%d_%d.dat' % (self.iteration, self.pc_id), self.my_wijs)
        np.savetxt(self.params['tmp_folder'] + 'bias_%d_%d.dat' % (self.iteration, self.pc_id), self.my_bias)
        if self.comm != None:
            self.comm.barrier()



    def merge_weight_files(self, n_iterations):
        if self.pc_id == 0:
            for iteration in xrange(n_iterations):
                cmd = 'cat '
                for pc_id in xrange(self.n_proc):
                    cmd += ' %s' % (self.params['tmp_folder'] + 'wij_%d_%d.dat' % (iteration, pc_id))

                output_fn = self.params['weights_folder'] + 'all_weights_%d.dat' % (iteration)
                cmd += ' > %s' % output_fn
                print cmd
                os.system(cmd)

                print 'creating weight matrix for iteration', iteration
                wij_list = np.loadtxt(output_fn)
                wij_matrix = np.zeros((self.params['n_exc'], self.params['n_exc']))
                pij_matrix = np.zeros((self.params['n_exc'], self.params['n_exc']))
                for line in xrange(wij_list[:, 0].size):
                    i, j, wij, pij = wij_list[line, :]
                    wij_matrix[i, j] = wij
                    pij_matrix[i, j] = pij
                np.savetxt(self.params['weights_folder'] + 'weight_matrix_%d.dat' % (iteration), wij_matrix)
#                np.savetxt(self.params['weights_folder'] + 'pij_matrix_%d.dat' % (iteration), pij_matrix)
                
            for iteration in xrange(n_iterations):
                cmd = 'cat '
                for pc_id in xrange(self.n_proc):
                    cmd += ' %s' % (self.params['tmp_folder'] + 'bias_%d_%d.dat' % (iteration, pc_id))

                output_fn = self.params['weights_folder'] + 'all_bias_%d.dat' % (iteration)
                cmd += ' > %s' % output_fn
                print cmd
                os.system(cmd)
                bias_list = np.loadtxt(output_fn)
                bias_array = np.zeros((self.params['n_exc'], 2))
                for line in xrange(bias_list[:, 0].size):
                    cell, bias = bias_list[line, :]
                    bias_array[cell] = bias
                np.savetxt(self.params['weights_folder'] + 'bias_array_%d.dat' % (iteration), bias_array)


    def merge_abstract_input_files(self):

        if self.pc_id == 0:
            print 'Merging abstract input files for stim:'
            n_cells = self.params['n_exc']
            cmd = 'cat '
            for stim in xrange(self.n_iterations_total):
                print stim, '\t'
                if self.normalize == False:
                    """
                    put all the cellwise seperated abstract L_i into one file
                    """
                    L_i = np.zeros((self.n_time_steps, self.params['n_exc']))
                    training_input_folder = "%sTrainingInput_%d/" % (self.params['folder_name'], stim)
                    fn_base = training_input_folder + self.params['abstract_input_fn_base']
                    for cell in xrange(self.params['n_exc']):
                        fn = fn_base + str(cell) + '.dat'
                        L_i[:, cell] = np.loadtxt(fn)
                    output_fn = '%sANNActivity/input_%d.dat' % (self.params['folder_name'], stim)
                    np.savetxt(output_fn, L_i)

                cmd += ' %sANNActivity/input_%d.dat' % (self.params['folder_name'], stim)
            fn_out = '%sParameters/all_inputs_scaled.dat' % (self.params['folder_name'])
            cmd +=  '  > %s' % (fn_out)
            print cmd
            os.system(cmd)

            d = np.loadtxt(fn_out)
            d_trans = d.transpose()
            fn_out = '%sParameters/all_inputs_scaled_transposed.dat' % (self.params['folder_name'])
            print 'Saving transposed input to:', fn_out
            np.savetxt(fn_out, d_trans)
        if comm != None:
            comm.barrier()






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


    AT = AbstractTrainer(params, comm)
#    cells_to_record = [18, 258, 352, 223, 112, 22, 38, 178, 186, 216, 334, 183]
    cells_to_record = []
    selected_connections = []
    for src in cells_to_record:
        for tgt in cells_to_record:
            if src != tgt:
                selected_connections.append((src, tgt))
    AT.set_selected_connections(selected_connections)
#    AT.create_stimuli_going_through_center(random_order=False, test_stim=False)
    AT.create_stimuli(random_order=False, test_stim=False)
    AT.merge_abstract_input_files()
#    AT.train()
#    n_iterations_total = params['n_theta'] * params['n_speeds'] * params['n_cycles'] * params['n_stim_per_direction']
#    AT.merge_weight_files(n_iterations_total)



