import os
import numpy as np
sn = "NetworkSimModuleNoColumns.py"

#w_ee = .38
#for delay_scale in [2, 3, 5, 10, 20]:
#    os.system('mpirun -np 2 python  %s %f %f'  % (sn, w_ee, delay_scale))


#for w_sigma_v in [.5, .75, 1.]:
#    for w_sigma_x in [.5, .75, 1.]:
#        for scale_latency in [.15, .2, .25, .3, .4]:
for w_ee in [.3, .35, .4]:
    for delay_scale in [1000., 500., 250., 100., 50., 25., 10., 5., 2.]:
        os.system('mpirun -np 2 python  %s %f %f'  % (sn, w_ee, delay_scale))
