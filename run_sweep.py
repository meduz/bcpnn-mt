import os
import numpy as np
sn = "NetworkSimModuleNoColumns.py"


#for w_sigma_v in [.25, .5, .75, 1., 1.25, 1.5, 2., 3., 4., 5.]:
for w_sigma_v in np.arange(.2, 2.1, .1):
    w_sigma_x = w_sigma_v * 5
    os.system('mpirun -np 8 python  %s %f %f'  % (sn, w_sigma_x, w_sigma_v))

#scale_latency = .10
#for delay_scale in [500., 200., 100., 50.]:
#    for w_sigma_v in [.2, .3, .4]:
#        for w_sigma_x in [.2, .3, .4]:
#            for w_ee in [.5, .6, .7]:
#                os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))
