import os
import numpy as np
sn = "NetworkSimModuleNoColumns.py"

#for w_sigma_v in [.5, .75]:
#    for w_sigma_x in [.5, .75]:
#for delay_scale in [1., 2., 3., 5., 10.]:
#    for connectivity_radius in [.1, .2, .3, .4]:
#        for w_ee in [.2, .25]:
#            os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, connectivity_radius, delay_scale))

#for delay_scale in [10., 5., 3., 2.]:
#    for connectivity_radius in [.2, .3]:
#        for w_ee in [.25, .3]:
#            os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, connectivity_radius, delay_scale))

for connectivity_radius in [1.]:#, .75, .5, .25]:
    for delay_scale in [1000.]:#, 500., 250.]:#, 100., 50., 25., 10., 5., 2.]:
        for w_ee in [.2, .25, .3, .35, .4]:
            os.system('mpirun -np 2 python  %s %f %f %f'  % (sn, w_ee, connectivity_radius, delay_scale))

#for a in np.arange(.5, 6., .5):
#    for b in np.arange(.1, 1.1, .1):
#        os.system('mpirun -np 2 python  %s %f %f'  % (sn, a, b))
