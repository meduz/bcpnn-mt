import os
import numpy as np
sn = "NetworkSimModuleNoColumns.py"



for w_sigma_v in [.5, .75]:
    for w_sigma_x in [.5, .75]:
        for delay_scale in [1., 2., 3., 5., 10.]:
            for scale_latency in [.1, .2, .3, .4]:
                for w_ee in [.2, .25, .3, .35]:
                    os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))
