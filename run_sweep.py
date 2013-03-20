import os
import numpy as np
sn = "NetworkSimModuleNoColumns.py"

#for (wsx, wsv) in [(.1, .05)]:
for w_sigma_v in np.arange(.1, .5, .1):
    for w_sigma_x in np.arange(.1, .9, .1):
#        for delay_scale in [1, 2, 5, 10, 15, 20]:
        delay_scale = 1
        w_ee_range = [.04]
#        w_ee_range = np.arange(.04, .07, .02)
        for w_ee in w_ee_range:
            os.system('mpirun -np 4 python  %s %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, delay_scale))

#w_sigma_x, w_sigma_v, w_ee, delay_scale = .3, .1, .04, 1
#os.system('mpirun -np 8 python  %s %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, delay_scale))

#w_sigma_x, w_sigma_v, w_ee, delay_scale = .3, .15, .04, 1
#os.system('mpirun -np 8 python  %s %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, delay_scale))

#w_sigma_x, w_sigma_v, w_ee, delay_scale = .3, .2, .04, 1
#os.system('mpirun -np 8 python  %s %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, delay_scale))

