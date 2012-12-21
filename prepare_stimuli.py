import pylab
import numpy as np

import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary


fig = pylab.figure()
ax = fig.add_subplot(211)

n_theta = params['n_theta']
n_speeds = params['n_speeds']
n_stim_per_direction = params['n_stim_per_direction']
v = 1.1

start_pos = np.linspace(0, 1, n_stim_per_direction + 2)[1:-1]
thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
print 'start_pos', start_pos
print 'thetas', thetas

def get_rotation_matrix(theta):
    R = np.array([[np.cos(theta), np.sin(theta), 0], \
                [-np.sin(theta), np.cos(theta), 0], \
                [0, 0, 1]])
    return R



def get_t(x, tstart, tstop):
    tau = .03 * np.pi 
    if x > tstart and x < tstop:
        f_x = 1 / (1 + np.exp(-((x-3*tau - tstart) / (tau/2))))
    else:
        f_x = 0
    if x >= tstop:
        f_x = 1 / (1 + np.exp(-((tstop + 3 * tau -x)/(tau/2))))
    return f_x


def get_translation_matrix(theta):
#    tx = (theta / (0.5 * np.pi))
#    tx = theta / np.pi * np.sin(theta - (0.5 * np.pi))

    tx = get_t(theta, .6 * np.pi, 1.75*np.pi)
    ty = get_t(theta, .1 * np.pi, 1.20*np.pi)
    T = np.array([[1, 0, tx], \
                [0, 1, ty], \
                [0, 0, 1]])
    return T, tx, ty


x_0 = np.zeros((3, n_stim_per_direction))
x_0[0, :] = start_pos
print 'x_0', x_0
color_list = ['k', 'b', 'g', 'r', 'y', 'c', 'm', '#00f80f', '#deff00', '#ff00e4', '#00ffe6']
legend = []
legend_text = []
txs = []
tys = []
for i_, theta in enumerate(thetas):
    R = get_rotation_matrix(theta)
    T, tx, ty = get_translation_matrix(theta)
    txs.append(tx)
    tys.append(ty)
    vx = v * np.sin(theta)
    vy = v * np.cos(theta)
    for j in xrange(n_stim_per_direction):
        rotated_start_pos = np.dot(R, x_0[:, j])
        translated_start_pos = np.dot(T, [rotated_start_pos[0], rotated_start_pos[1], 1])


        line = ax.plot(rotated_start_pos[0], rotated_start_pos[1], 'o', color=color_list[i_], ms=5, label='theta=%.3f' % theta)
        ax.plot([rotated_start_pos[0], rotated_start_pos[0] + vx], [rotated_start_pos[1], rotated_start_pos[1] + vy], ls=':', lw=2, color=color_list[i_])

        ax.plot(translated_start_pos[0], translated_start_pos[1], 'x', color=color_list[i_], ms=5)
        ax.plot([translated_start_pos[0], translated_start_pos[0] + vx], [translated_start_pos[1], translated_start_pos[1] + vy], lw=2, color=color_list[i_])

    legend.append(line[0])
    legend_text.append('theta = %.2f * pi = %.3f' % (theta / np.pi, theta))

            
ax.legend(legend, legend_text)

ax.set_xlim((-0.5, 1.5))
ax.set_ylim((-0.5, 1.5))
ax.plot([0, 1], [0, 0], 'k--', lw=3)
ax.plot([1, 1], [0, 1], 'k--', lw=3)
ax.plot([1, 0], [1, 1], 'k--', lw=3)
ax.plot([0, 0], [1, 0], 'k--', lw=3)

ax1 = fig.add_subplot(223)
ax2 = fig.add_subplot(224)
x = np.arange(0, 2*np.pi, 0.01)
tx = [get_t(xi, .6 * np.pi, 1.7*np.pi) for xi in x]
ty = [get_t(xi, .1 * np.pi, 1.20*np.pi) for xi in x]
ax1.plot(x, tx, ':')
ax1.plot(thetas, txs, '-o')
ax1.set_xlabel('theta')
ax1.set_ylabel('tx')
ax2.plot(x, ty, ':')
ax2.plot(thetas, tys, '-o')
ax2.set_xlabel('theta')
ax2.set_ylabel('ty')

pylab.show()

