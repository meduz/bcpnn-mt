import os
import numpy as np
sn = "NetworkSimModuleNoColumns.py"

#for w_sigma_x in np.arange(.1, .7, .2):
#    for w_sigma_v in np.arange(.1, .7, .2):
#        if (w_sigma_x + w_sigma_v) > .4:
#            w_ee_range = np.arange(.04, .07, .0025)
#        else:
#            w_ee_range = np.arange(.02, .05, .0025)
#        for w_ee in w_ee_range:
#            os.system('mpirun -np 8 python  %s %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee))

w_sigma_x, w_sigma_v, w_ee, delay_scale = .3, .1, .04, 1
os.system('mpirun -np 8 python  %s %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, delay_scale))

w_sigma_x, w_sigma_v, w_ee, delay_scale = .3, .15, .04, 1
os.system('mpirun -np 8 python  %s %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, delay_scale))

w_sigma_x, w_sigma_v, w_ee, delay_scale = .3, .2, .04, 1
os.system('mpirun -np 8 python  %s %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, delay_scale))

