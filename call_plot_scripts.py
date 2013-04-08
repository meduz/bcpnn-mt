import os
import time
import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

#exc_cells = [23, 150]


n_iterations = params['n_theta'] * params['n_speeds'] * params['n_cycles'] * params['n_stim_per_direction']
#n_iterations = 64

for iteration in xrange(n_iterations):
#    os.system('python plot_ann_activity_as_quiver_plot_one_stimulus.py %d' % iteration)
#    output_movie = 'stim_movie_%d.mp4' % (iteration)
#    os.system('python make_movie.py %s %s' % ('network_activity_%02d_\%d.png' % iteration, output_movie))

    os.system('python plot_abstract_activation.py %d' % (iteration))


#    matrix_fn = params['folder_name'] + 'TrainingResults_%d/' % iteration + 'wij_matrix_%d.dat' % (iteration)
#    os.system('python plot_all_pij.py %d' % iteration)
#    os.system('python plot_ann_output_activity.py %d' % (iteration))
#    matrix_fn = 'AndersWij/wij12.dat'
#    for exc_cell in exc_cells:
#        fig_fn = params['figures_folder'] + 'conn_profile_%d_%d.png' % (iteration, exc_cell)
#        os.system('python plot_connectivity_profile_abstract.py %d %s %s %d' % (exc_cell, matrix_fn, fig_fn, iteration))


#for cell in cell_idx:
#cell_idx = range(params['n_exc'])
#for cell in xrange(60, 400):
#    os.system('python plot_connectivity_profile_abstract.py %d AndersWij/wij12.txt AndersWij/%d.png' % (cell, cell))
