import numpy as np
import os
import simulation_parameters
import NeuroTools.parameters as NTP
import ResultsCollector

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

#RC.collect_files()
dir_names = [#'SmallScale_noBlank_AIII_scaleLatency0.02_wsigmax1.00e-01_wsigmav1.00e-01_wee3.50e-02_wei1.00e-01_wie1.50e-01_wii1.00e-02_delayScale5', \
        'SmallScale_noBlank_AIII_scaleLatency0.02_wsigmax2.00e-01_wsigmav1.00e-01_wee3.50e-02_wei1.00e-01_wie1.50e-01_wii1.00e-02_delayScale5', \
        'SmallScale_noBlank_AIII_scaleLatency0.02_wsigmax2.00e-01_wsigmav2.00e-01_wee3.50e-02_wei1.00e-01_wie1.50e-01_wii1.00e-02_delayScale5', \
#        'SmallScale_noBlank_AIII_scaleLatency0.05_wsigmax1.00e-01_wsigmav1.00e-01_wee3.50e-02_wei1.00e-01_wie1.50e-01_wii1.00e-02_delayScale5', \
        'SmallScale_noBlank_AIII_scaleLatency0.05_wsigmax2.00e-01_wsigmav1.00e-01_wee3.50e-02_wei1.00e-01_wie1.50e-01_wii1.00e-02_delayScale5', \
        'SmallScale_noBlank_AIII_scaleLatency0.05_wsigmax2.00e-01_wsigmav2.00e-01_wee3.50e-02_wei1.00e-01_wie1.50e-01_wii1.00e-02_delayScale5', \
#        'SmallScale_noBlank_AIII_scaleLatency0.10_wsigmax1.00e-01_wsigmav1.00e-01_wee3.50e-02_wei1.00e-01_wie1.50e-01_wii1.00e-02_delayScale5', \
        'SmallScale_noBlank_AIII_scaleLatency0.10_wsigmax2.00e-01_wsigmav1.00e-01_wee3.50e-02_wei1.00e-01_wie1.50e-01_wii1.00e-02_delayScale5', \
        'SmallScale_noBlank_AIII_scaleLatency0.10_wsigmax2.00e-01_wsigmav2.00e-01_wee3.50e-02_wei1.00e-01_wie1.50e-01_wii1.00e-02_delayScale5', \
#        'SmallScale_noBlank_AIII_scaleLatency0.20_wsigmax1.00e-01_wsigmav1.00e-01_wee3.50e-02_wei1.00e-01_wie1.50e-01_wii1.00e-02_delayScale5', \
        'SmallScale_noBlank_AIII_scaleLatency0.20_wsigmax2.00e-01_wsigmav1.00e-01_wee3.50e-02_wei1.00e-01_wie1.50e-01_wii1.00e-02_delayScale5', \
        'SmallScale_noBlank_AIII_scaleLatency0.20_wsigmax2.00e-01_wsigmav2.00e-01_wee3.50e-02_wei1.00e-01_wie1.50e-01_wii1.00e-02_delayScale5', \
#        'SmallScale_noBlank_AIII_scaleLatency0.30_wsigmax1.00e-01_wsigmav1.00e-01_wee3.50e-02_wei1.00e-01_wie1.50e-01_wii1.00e-02_delayScale5', \
        'SmallScale_noBlank_AIII_scaleLatency0.30_wsigmax2.00e-01_wsigmav1.00e-01_wee3.50e-02_wei1.00e-01_wie1.50e-01_wii1.00e-02_delayScale5', \
        'SmallScale_noBlank_AIII_scaleLatency0.30_wsigmax2.00e-01_wsigmav2.00e-01_wee3.50e-02_wei1.00e-01_wie1.50e-01_wii1.00e-02_delayScale5']

RC.dirs_to_process = dir_names
print "RC.dirs_to_process", RC.dirs_to_process
RC.get_xvdiff_integral()
RC.get_cgxv()
RC.plot_cgxv_vs_xvdiff()

