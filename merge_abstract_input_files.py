import os
import numpy as np
import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.params
n_cells = params['n_exc']
n_iterations = 40

#n_time_steps = params['t_sim'] / params['dt_rate']
#for iteration in xrange(n_iterations):
#    data = np.zeros((n_time_steps, n_cells))
#    for cell in xrange(n_cells):
#        fn = 'Abstract/TrainingInput_%d/abstract_input_%d.dat' % (iteration, cell)
#        d = np.loadtxt(fn)
#        data[:, cell] = d
#    fn_out = 'Abstract/Parameters/input_%d.dat' % iteration
#    np.savetxt(fn_out, data)


cmd = 'cat '
for i in xrange(n_iterations):
    cmd += ' Abstract/ANNActivity/input_%d.dat' % i
fn_out = 'Abstract/Parameters/all_inputs.dat'
cmd +=  '  > %s' % (fn_out)
print cmd
os.system(cmd)


