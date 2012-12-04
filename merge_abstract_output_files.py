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
    cmd += ' %sANNActivity/output_activity_%d.dat' % (params['folder_name'], i)
fn_out = '%sParameters/all_output_activity.dat' % (params['folder_name'])
cmd +=  '  > %s' % (fn_out)
print cmd
os.system(cmd)

d = np.loadtxt(fn_out)
d_trans = d.transpose()

fn_out = '%sParameters/all_output_activity_transposed.dat' % (params['folder_name'])
print 'Saving transposed input to:', fn_out
np.savetxt(fn_out, d_trans)


