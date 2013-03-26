import os
import numpy as np
sn = "NetworkSimModuleNoColumns.py"

#for (wsx, wsv) in [(.1, .05)]:
for w_sigma_v in np.arange(.1, .5, .1):
    for w_sigma_x in np.arange(.1, .9, .1):
        for scale_latency in [1., .5, .25, .15, .1, .05]:
            w_ee_range = [.6]
            for w_ee in w_ee_range:
                os.system('mpirun -np 2 python  %s %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency))

#w_sigma_x, w_sigma_v, w_ee, scale_latency = .3, .1, .04, 1
#os.system('mpirun -np 8 python  %s %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency))

#w_sigma_x, w_sigma_v, w_ee, scale_latency = .3, .15, .04, 1
#os.system('mpirun -np 8 python  %s %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency))

#w_sigma_x, w_sigma_v, w_ee, scale_latency = .3, .2, .04, 1
#os.system('mpirun -np 8 python  %s %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency))

