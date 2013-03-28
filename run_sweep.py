import os
import numpy as np
sn = "NetworkSimModuleNoColumns.py"


#w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .3, .3, .7, .15, 600
#os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))

w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .2, .2, .5, .5, 500
os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))
w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .2, .2, .4, .5, 500
os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))

w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .2, .2, .5, .5, 250
os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))
w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .2, .2, .4, .5, 250
os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))

w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .2, .2, .4, .5, 100
os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))
w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .2, .2, .3, .5, 100
os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))

w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .2, .2, .4, .5, 100
os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))
w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .2, .2, .35, .5, 100
os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))
w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .2, .2, .30, .5, 100
os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))

w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .3, .3, .5, .5, 500
os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))
w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .3, .3, .4, .5, 500
os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))

w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .3, .3, .5, .5, 250
os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))
w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .3, .3, .4, .5, 250
os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))

w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .3, .3, .4, .5, 100
os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))
w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale = .3, .3, .3, .5, 100
os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))


#w_sigma_x, w_sigma_v, w_ee, scale_latency = .3, .1, .5, .1
#for delay_scale in [500., 200., 100., 50.]:
#    for w_ee in [.5, .4, .3, .2]:
#        os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))

#w_sigma_x, w_sigma_v, scale_latency = .4, .1, .1
#for delay_scale in [500., 200., 100., 50.]:
#    for w_ee in [.6, .5, .4]:
#        os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))

#scale_latency = .10
#for delay_scale in [500., 200., 100., 50.]:
#    for w_sigma_v in [.1, .2]:
#        for w_sigma_x in [.1, .2]:
#            for w_ee in [.4, .3, .2]:
#                os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))

#scale_latency = .10
#for delay_scale in [500., 200., 100., 50.]:
#    for w_sigma_v in [.2, .3, .4]:
#        for w_sigma_x in [.2, .3, .4]:
#            for w_ee in [.5, .6, .7]:
#                os.system('mpirun -np 2 python  %s %f %f %f %f %f'  % (sn, w_sigma_x, w_sigma_v, w_ee, scale_latency, delay_scale))
