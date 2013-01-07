"""
Two different tuning curves can be measured:
    1) stimulus orientation  vs  output rate
    2) stimulus distance  vs  output rate
"""
import os
import numpy as np
import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

n_theta = 3
n_trials = 2  # per angle

prepare_input = "python prepare_spike_trains.py "
measurement = "python measure_tuning_curve_one_run.py "

v_theta = np.linspace(-.5 * np.pi, .5 * np.pi, n_theta, endpoint=False)
v0 = .3 # amplitude

f = file(params['input_params_fn'], 'w')
input_params = ''
x0, y0 = .6, .5 #params['motion_params'][0:2]
sim_cnt = 0
for i_theta, theta in enumerate(v_theta):
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    x_start = x0 - vx
    y_start = y0 - vy
    print 'Preparing spikes for ', sim_cnt
    os.system(prepare_input + "%f %f %f %f" % (x_start, y_start, vx, vy))
    input_params = '%f\t%f\t%f\t%f\n' % (x_start, y_start, vx, vy)
    f.write(input_params)
    f.flush()
    for trial in xrange(n_trials):
        print measurement + "%d %f %f %f %f" % (sim_cnt, x_start, y_start, vx, vy)
        os.system(measurement + "%d %f %f %f %f" % (sim_cnt, x_start, y_start, vx, vy))
        sim_cnt = i_theta * n_trials + trial + 1

f.close()

