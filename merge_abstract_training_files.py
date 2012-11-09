import sys
import os
import numpy as np
import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.params

def merge(iteration):
    training_folder = '%sTrainingResults_%d/' % (params['folder_name'], iteration)
    tmp_fn = training_folder + 'all_wij_%d.dat' % (iteration)
    cat_cmd = 'cat %s* > %s' % (training_folder + 'wij_', tmp_fn)
    print cat_cmd
    os.system(cat_cmd)

    tmp_fn = training_folder + 'all_bias_%d.dat' % (iteration)
    cat_cmd = 'cat %s* > %s' % (training_folder + 'bias_', tmp_fn)
    print cat_cmd
    os.system(cat_cmd)

    tmp_fn = training_folder + 'all_pi_%d.dat' % (iteration)
    cat_cmd = 'cat %s* > %s' % (training_folder + 'pi_', tmp_fn)
    print cat_cmd
    os.system(cat_cmd)

    tmp_fn = training_folder + 'all_pj_%d.dat' % (iteration)
    cat_cmd = 'cat %s* > %s' % (training_folder + 'pj_', tmp_fn)
    print cat_cmd
    os.system(cat_cmd)

    tmp_fn = training_folder + 'all_pij_%d.dat' % (iteration)
    cat_cmd = 'cat %s* > %s' % (training_folder + 'pij_', tmp_fn)
    print cat_cmd
    os.system(cat_cmd)

if len(sys.argv) < 2:
    iterations = range(40)
    for iteration in xrange(iterations):
        merge(iteration)
else:
    merge(int(sys.argv[1]))




#cat_cmd = 'cat '
#for iteration in xrange(n_iterations):
#    fn = "%soutput_activity_%d.dat" % (params['activity_folder'], iteration)
#    cat_cmd += ' %s' % fn
#fn = "%soutput_activity_allruns.dat" % (params['activity_folder'])
#cat_cmd += ' > %s' % (fn)
#print cat_cmd
#os.system(cat_cmd)
