import numpy as np
import CreateConnections as CC
import utils
import simulation_parameters
ps = simulation_parameters.parameter_storage()
params = ps.params

tp = np.loadtxt(params['tuning_prop_means_fn'])
tgt_gid = 0
srcs = [1, 2, 3, 4, 5, 6, 7, 8]
tp_src = tp[srcs, :]
tp_tgt = tp[tgt_gid, :]
#d_ij = utils.torus_distance2D_vec(tp_src[:, 0], tp_tgt[0] * np.ones(n_src), tp_src[:, 1], tp_tgt[1] * np.ones(n_src), w=np.ones(n_src), h=np.ones(n_src))
print 'tp_src', tp_src
print 'tp_tgt', tp_tgt
#print 'd_ij', d_ij

w_sigma_x, w_sigma_v = .1, .1
v_src = np.array((tp_src[:, 2], tp_src[:, 3]))
v_src = v_src.transpose()
print 'v_src', v_src.shape
p, l = CC.get_p_conn_vec(tp[srcs, :], tp[tgt_gid, :], w_sigma_x, w_sigma_v)
print 'p', p
print 'l', l

for src in xrange(len(srcs)):
    p_, l_ = CC.get_p_conn(tp_src[src, :], tp_tgt, w_sigma_x, w_sigma_v)
    print 'src p l', src, p_, p[src], l_, l[src]
#print p
