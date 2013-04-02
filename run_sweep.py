import os
import numpy as np
sn = "NetworkSimModuleNoColumns.py"





delay_scale = 5.
w_sigma_v = 2.
w_sigma_x = 50.
w_ee = 0.65
os.system('mpirun -np 2 python  %s %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, delay_scale))

for w_sigma_x in [100.]:
    for w_ee in np.arange(.5, .7, .05):
        os.system('mpirun -np 2 python  %s %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, delay_scale))


for delay_scale in [10., 50., 100., 500., 1000.]:
    for w_sigma_v in [2., 3., 4., 5.]:
        for w_sigma_x in [5., 10., 15., 20., 50., 100.]:
            for w_ee in np.arange(.5, .7, .05):
                os.system('mpirun -np 2 python  %s %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, delay_scale))

#scale_latency = .10
#for delay_scale in [500., 200., 100., 50.]:
#    for w_sigma_v in [.2, .3, .4]:
#        for w_sigma_x in [.2, .3, .4]:
#            for w_ee in [.5, .6, .7]:
#                os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))
