import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

def normalize_input(fn_base):
    input_scaling_factor = params['abstract_input_scaling_factor']
    print 'normalize_input for', fn_base
    L_input = np.zeros((n_time_steps, params['n_exc']))
    n_hc = params['N_RF_X']*params['N_RF_Y']
    n_cells_per_hc = params['N_theta'] * params['N_V']
    n_cells = params['n_exc']
    assert (n_cells == n_hc * n_cells_per_hc)
    for cell in xrange(n_cells):
        fn = fn_base + str(cell) + '.dat'
        L_input[:, cell] = np.loadtxt(fn)
        L_input[:, cell] *= input_scaling_factor

    for t in xrange(int(n_time_steps)):
        for hc in xrange(n_hc):
            idx0 = hc * n_cells_per_hc
            idx1 = (hc + 1) * n_cells_per_hc
            s = L_input[t, idx0:idx1].sum()
            if s > 1:
                L_input[t, idx0:idx1] = np.exp(L_input[t, idx0:idx1]) / np.exp(L_input[t, idx0:idx1]).sum()

    for cell in xrange(n_cells):
        output_fn = fn_base + str(cell) + '.dat'
        np.savetxt(output_fn, L_input[:, cell])

    all_output_fn = params['activity_folder'] + 'input_%d.dat' % (iteration)
    print 'Normalized input is written to:', all_output_fn
    np.savetxt(all_output_fn, L_input)



stim = 0
training_input_folder = "%sTrainingInput_%d/" % (params['folder_name'], stim)
output_fn_base = training_input_folder + params['abstract_input_fn_base']
normalize_input(output_fn_base)

