import numpy as np
import os
import simulation_parameters
import NeuroTools.parameters as NTP
import ResultsCollector
import re
import pylab

"""
This script requires that plot_prediction.py has been called for all folders that 
are to be processed here (that appear in dir_names).

--> you can use run_plot_prediction.py to automatically do this
"""

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

w_ee = 0.030
#t_blank = 200
conn_code = 'AAAA'
# the parameter to sweep for
#param_name = 'w_tgt_in_per_cell_ee'
param_name = 't_blank'
#param_name = 'delay_scale'
#param_name = 'scale_latency'
t_range=(0, 1000)
#t_range=(0, 3000)

output_fn = 'xvdiff_%s_wee%.2e_tblankSweep_t%d-%d.dat' % (conn_code, w_ee, t_range[0], t_range[1])
print 'output_fn', output_fn

#to_match = '^LargeScaleModel_%s_scaleLatency0\.15(.*)delayScale20_tblank%d$' % (conn_code, t_blank)
#to_match = '^LargeScaleModel_%s_scaleLatency0\.15(.*)wee(.*)_wei4.00e-02_wie6.00e-02_wii1.00e-02_delayScale20_tblank200' % (conn_code)
#to_match = '^LargeScaleModel_%s_scaleLatency(.*)wee%.2e(.*)delayScale10_tblank%d$' % (conn_code, w_ee, t_blank)
#to_match = '^LargeScaleModel_%s_scaleLatency0\.15(.*)wee2\.(\d\d)e(.*)tblank(\d+)$' % (conn_code)
#to_match = '^LargeScaleModel_%s_scaleLatency0\.15(.*)wee3.00e-02_wei4.00e-02_wie6.00e-02_wii1.00e-02_delayScale20_tblank(\d+)$' % (conn_code)
to_match = '^LargeScaleModel_%s_scaleLatency0\.15(.*)wee(.*)delayScale20_tblank(\d+)$' % (conn_code)

dir_names = []
for thing in os.listdir('.'):
    m = re.search('%s' % to_match, thing)
    if m:
        dir_names.append(thing)

print 'dirnames:'
for name in dir_names:
    print name, os.path.exists(name + '/Data/vx_grid.dat')
#exit(1)


RC.set_dirs_to_process(dir_names)
#print "RC.dirs_to_process", RC.dirs_to_process
RC.get_xvdiff_integral(t_range=t_range)
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

