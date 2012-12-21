import os
import matplotlib
import simulation_parameters
import numpy as np

class AbstractNetwork(object):

    def __init__(self, params):

        self.params = params
        self.set_iteration(0)
        self.tuning_prop = np.loadtxt(self.params['tuning_prop_means_fn'])
        assert (self.tuning_prop[:, 0].size == self.params['n_exc']), 'Number of cells does not match in %s and simulation_parameters!\n Wrong tuning_prop file?' % self.params['tuning_prop_means_fn']
        self.vx_tuning = self.tuning_prop[:, 2]
        self.vy_tuning = self.tuning_prop[:, 3]


    def set_iteration(self, iteration):
        self.iteration = iteration
        self.training_input_folder = "%sTrainingInput_%d/" % (self.params['folder_name'], iteration)


    def set_weights(self, iteration, no_rec=False):

        if no_rec:
            self.wij = np.zeros((self.params['n_exc'], self.params['n_exc']))
            self.bias = np.zeros((self.params['n_exc'], 2))
            return

        weight_matrix_fn = self.params['weights_folder'] + 'weight_matrix_%d.dat' % (iteration)
        if not os.path.exists(weight_matrix_fn):
            self.wij = self.get_weight_matrix(iteration) # deprecated
            np.savetxt(weight_matrix_fn, self.wij)
        else:
            print 'Loading weight matrix from:', weight_matrix_fn
            self.wij = np.loadtxt(weight_matrix_fn)
            print 'Loading bias values from:', self.params['weights_folder'] + 'bias_array_%d.dat' % (iteration)
            self.bias = np.loadtxt(self.params['weights_folder'] + 'bias_array_%d.dat' % (iteration))


    def calculate_dynamics(self, output_fn_base=None):

        fn_base = self.training_input_folder + self.params['abstract_input_fn_base']
        n_cells = self.params['n_exc']
        n_hc = self.params['N_RF_X']*self.params['N_RF_Y']
        n_cells_per_hc = self.params['N_theta'] * self.params['N_V']

        cell = 0
        fn = fn_base + '%d.dat' % (cell)
        d_sample = np.loadtxt(fn)
        self.n_time_steps = d_sample.size

        stimulus = np.zeros((self.n_time_steps, n_cells))
        self.output_activity = np.zeros((self.n_time_steps, n_cells))

        self.v_pred = np.zeros((self.n_time_steps, 3))
        self.t_axis = np.arange(self.n_time_steps)
        self.t_axis *= self.params['dt_rate']
        self.v_pred[:, 0] = self.t_axis

        print 'Loading files %s' % fn_base, 
        for cell in xrange(n_cells):
            fn = fn_base + '%d.dat' % (cell)
            stimulus[:, cell] = np.loadtxt(fn)

        # when computing the output estimates, the input from the network and the input from the stimulus are taken into account
        # by weighting them
        w_stim = 0.9
        w_net = 1. - w_stim
            
        # initialize the output activity as being the response to the stimulus
        self.output_activity[0, :] = stimulus[0, :]
        print '\t... and computing ...'
        for t in xrange(1, self.n_time_steps):
            print 'iteration %d\tt: %d / %d' % (self.iteration, t, self.n_time_steps)
            for post_hc in xrange(n_hc):
                output_from_hc = np.zeros(n_cells_per_hc)
                for post_ in xrange(n_cells_per_hc):
                    post_gid = post_hc * n_cells_per_hc + post_
                    input_from_network = 0
                    for pre_hc in xrange(n_hc):
                        input_from_hc = 1e-3
                        idx_0 = pre_hc * n_cells_per_hc
                        idx_1 = (pre_hc + 1) * n_cells_per_hc
                        input_from_hc += np.dot(self.wij[idx_0:idx_1, post_gid], self.output_activity[t-1, idx_0:idx_1])
#                        print input_from_hc, 
#                        s = np.dot(self.wij[idx_0:idx_1, post_gid], stimulus[t, idx_0:idx_1])
#                        if s > 1:
#                            input_from_hc = 1.
#                        else:
#                            input_from_hc = s
                        # no exponentiation for pre-activity (=stimulus) needed here, since the stimulus has been created this way already
#                        input_from_hc += np.dot(self.wij[idx_0:idx_1, post_gid], stimulus[t, idx_0:idx_1])
#                        print 'debug input_from_hc', input_from_hc
                        input_from_network += np.log(input_from_hc)
#                        input_from_network += input_from_hc
                    output_from_hc[post_] = w_net * input_from_network + self.bias[post_gid, 1] + w_stim * stimulus[t, post_gid]
#                print '\n'
                post_gid_1 = post_hc * n_cells_per_hc
#                print '\n'
                post_gid_2 = (post_hc + 1) * n_cells_per_hc
                if np.exp(output_from_hc).sum() > 1.:
                    self.output_activity[t, post_gid_1:post_gid_2] = np.exp(output_from_hc) / np.exp(output_from_hc).sum()
                else:
                    self.output_activity[t, post_gid_1:post_gid_2] = np.exp(output_from_hc)
#                    self.output_activity[t, post_gid_1:post_gid_2] = output_from_hc
#                self.output_activity[t, post_gid_1:post_gid_2] = output_from_hc



            # map activity in the range (0, 1)
#            for cell in xrange(n_cells):
#                if (self.output_activity[t, cell] < 0):
#                    self.output_activity[t, cell] = 0

#             normalize activity within one HC to 1 (if larger than 1)
#            for hc in xrange(n_hc):
#                idx_0 = hc * n_cells_per_hc
#                idx_1 = (hc + 1) * n_cells_per_hc
#                o_sum = self.output_activity[t, idx_0:idx_1].sum()
#                if o_sum > 1:
#                    self.output_activity[t, idx_0:idx_1] /= o_sum

            if self.output_activity[t, :].sum() != 0:
                normed_activity = self.output_activity[t, :] / self.output_activity[t, :].sum()
            else:
                normed_activity = np.zeros(n_cells)
            self.v_pred[t, 1] = np.dot(self.vx_tuning, normed_activity)
            self.v_pred[t, 2] = np.dot(self.vy_tuning, normed_activity)


        if output_fn_base == None:
            output_fn_activity = params['activity_folder'] + 'output_activity_%d.dat' % (self.iteration)
        print 'Saving ANN activity to:', output_fn_activity
        np.savetxt(output_fn_activity, self.output_activity)
        output_fn_prediction = params['activity_folder'] + 'prediction_%d.dat' % (self.iteration)
        print 'Saving ANN prediction to:', output_fn_prediction
        np.savetxt(output_fn_prediction, self.v_pred)


    def get_weight_matrix(self, iteration):
        """
        DEPRECATED!!!
        """

        all_wij_fn= '%sTrainingResults_%d/all_wij_%d.dat' % (self.params['folder_name'], iteration, iteration)
        print 'Getting weights from ', all_wij_fn
        if not os.path.exists(all_wij_fn):
            os.system('python merge_abstract_training_files.py %d' % (iteration))
        d = np.loadtxt(all_wij_fn)
        n_cells = self.params['n_exc']
        wij_matrix = np.zeros((n_cells, n_cells))
        bias_matrix = np.zeros(n_cells)
        for line in xrange(d[:, 0].size):
            i, j, pij_, wij, bias_j = d[line, :]
            bias_matrix[j] = bias_j
            wij_matrix[i, j] = wij
        output_fn = params['folder_name'] + 'TrainingResults_%d/' % iteration + 'wij_matrix_%d.dat' % (iteration)
        print 'Saving to:', output_fn
        np.savetxt(output_fn, wij_matrix)
        return wij_matrix


    def eval_prediction(self):

        input_params = np.loadtxt(self.params['parameters_folder'] + 'input_params.txt')
        x0, y0, u0, v0 = input_params[self.iteration, :]
        v_stim = np.zeros((self.n_time_steps, 3))
        v_stim[:, 0] = self.t_axis
        v_stim[:, 1] = u0 * np.ones(self.n_time_steps) 
        v_stim[:, 2] = v0 * np.ones(self.n_time_steps) 

        vx_diff = self.v_pred[:, 1] - v_stim[:, 1]
        vy_diff = self.v_pred[:, 2] - v_stim[:, 2]
        v_diff = [np.sqrt(vx_diff[t]**2 + vy_diff[t]**2) for t in xrange(self.n_time_steps)]

        v_diff_out = np.zeros((self.n_time_steps, 4))
        v_diff_out[:, 0] = self.t_axis
        v_diff_out[:, 1] = vx_diff
        v_diff_out[:, 2] = vy_diff
        v_diff_out[:, 3] = v_diff

        output_fn_prediction_error = params['activity_folder'] + 'prediction_error_%d.dat' % (self.iteration)
        print 'Saving ANN prediction error to:', output_fn_prediction_error
        np.savetxt(output_fn_prediction_error, v_diff_out)


    def test_cycle(self, blank_stim=False):

        distance_from_center = 0.5
        center = (0.5, 0.5)
        thetas = np.linspace(np.pi, 3*np.pi, n_stim, endpoint=False)
#        r = 0.5 # how far the stimulus will move
        v_default = np.sqrt(self.params['motion_params'][2]**2 + self.params['motion_params'][3]**2)
        iteration = 0
        for cycle in xrange(self.n_cycles):
            for stim in stimulus_order:
                x0 = distance_from_center * np.cos(thetas[stim] + random_rotation[iteration]) + center[0]
                y0 = distance_from_center * np.sin(thetas[stim] + random_rotation[iteration]) + center[1]
                u0 = np.cos(np.pi + thetas[stim] + random_rotation[iteration]) * speeds[speed_cycle]#v_default
                v0 = np.sin(np.pi + thetas[stim] + random_rotation[iteration]) * speeds[speed_cycle]#v_default
                self.params['motion_params'] = (x0, y0, u0, v0)
                if blank_stim:
                    self.create_input_vectors_blanking(t_blank=(0.4, 0.6), normalize=self.normalize)
                else:
                    self.create_input_vectors(normalize=self.normalize)
                iteration += 1



if __name__ == '__main__':

    PS = simulation_parameters.parameter_storage()
    params = PS.params

    n_iterations = 24
    n_time_steps = params['t_sim'] / params['dt_rate']
    output_activity_all_iterations = np.zeros((n_iterations * n_time_steps, params['n_exc']),dtype=np.double)

    ANN = AbstractNetwork(params)
    ANN.set_weights(n_iterations-1)#, no_rec=True) # load weight matrix
    for iteration in xrange(n_iterations):
        ANN.set_iteration(iteration)
#        ANN.set_weights(iteration) # load weight matrix
        ANN.calculate_dynamics() # load activity files 
        ANN.eval_prediction()
        output_activity_all_iterations[iteration*n_time_steps:(iteration+1)*n_time_steps, :] = ANN.output_activity

    fn_out = params['activity_folder'] + 'ann_activity_%diterations.dat' % n_iterations
    print 'Saving all network activity to:', fn_out
    np.savetxt(fn_out, output_activity_all_iterations)



#    if plot_dynamics:
#        import matplotlib.animate as animate
        # load output files
#        for t in xrange(n_time_steps):
        # plot cell activities

