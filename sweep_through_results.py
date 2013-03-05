import numpy as np
import os
import simulation_parameters
import NeuroTools.parameters as NTP
import ResultsCollector
import re
import pylab

try:
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size

except:
    USE_MPI = False
    comm = None
    pc_id, n_proc = 0, 1

network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.params

RC = ResultsCollector.ResultsCollector(params)

#w_ee = 0.035
t_blank = 200
conn_code = 'IIII'

to_match = '^LargeScaleModel_%s_scaleLatency0\.15(.*)tblank%d$' % (conn_code, t_blank)
#to_match = '^LargeScaleModel_%s_scaleLatency0\.15(.*)wee%.2e(.*)tblank(\d+)$' % (conn_code, w_ee)

dir_names = []
for thing in os.listdir('.'):
    m = re.search('%s' % to_match, thing)
    if m:
#        print m.groups()
#        delay_scale = m.groups()[1]
        dir_names.append(thing)

print 'dirnames', dir_names
#exit(1)



output_fn = 'xvdiff_%s_tblank%d.dat' % (conn_code, t_blank)
RC.set_dirs_to_process(dir_names)
#print "RC.dirs_to_process", RC.dirs_to_process
t_range=(0, 3000)
RC.get_xvdiff_integral()#t_range=t_range)
param_name = 'w_tgt_in_per_cell_ee'
#param_name = 't_blank'
RC.get_parameter(param_name)
#RC.get_parameter('w_sigma_x')
#RC.get_parameter('w_sigma_v')
print 'RC param_space', RC.param_space
RC.n_fig_x = 1
RC.n_fig_y = 2
RC.create_fig()
RC.plot_param_vs_xvdiff_integral(param_name, xv='x', fig_cnt=1)#, t_integral=t_range)
RC.plot_param_vs_xvdiff_integral(param_name, xv='v', fig_cnt=2)#, t_integral=t_range)
RC.save_output_data(output_fn)
#pylab.show()
#RC.get_cgxv()
#RC.plot_cgxv_vs_xvdiff()

