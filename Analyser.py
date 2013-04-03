import json
import os
import numpy as np

class Analyser(object):
    def __init__(self, argv):
        if len(argv) > 1:
            if argv[1].isdigit():
                gid = int(argv[1])
            else:
                param_fn = argv[1]
                if os.path.isdir(param_fn):
                    param_fn += '/Parameters/simulation_parameters.json'
                print '\nLoading parameters from %s\n' % (param_fn)
                f = file(param_fn, 'r')
                params = json.load(f)
        else:
            print '\nLoading the default paremters...\n'
            import simulation_parameters
            ps = simulation_parameters.parameter_storage()
            params = ps.params

        self.params = params
        self.conn_list_loaded = [False, False, False, False]
        self.conn_mat_loaded = [False, False, False, False]
        self.conn_lists = {}


    def load_connlist(self, conn_type):
        if conn_type == 'ee':
            loaded = self.conn_list_loaded[0]
        elif conn_type == 'ei':
            loaded = self.conn_list_loaded[1]
        elif conn_type == 'ie':
            loaded = self.conn_list_loaded[2]
        elif conn_type == 'ii':
            loaded = self.conn_list_loaded[3]

        if loaded:
            return

        fn = self.params['merged_conn_list_%s' % conn_type]
        if not os.path.exists(fn):
            print 'Merging connlists for %s' % conn_type 
            tmp_fn = 'delme_tmp_%d' % (np.random.randint(0, 1e8))
            cat_cmd = 'cat %s* > %s' % (self.params['conn_list_ee_fn_base'], tmp_fn)
            sort_cmd = 'sort -gk 1 -gk 2 %s > %s' % (tmp_fn, self.params['merged_conn_list_ee'])
            rm_cmd = 'rm %s' % (tmp_fn)

            os.system(cat_cmd)
            os.system(sort_cmd)
            os.system(rm_cmd)

        assert os.path.exists(fn), 'Could not merge conn_list files into %s\n\n Did the simulation run and finish?\n' % fn
            
        print 'Loading:', fn
        self.conn_lists[conn_type] = np.loadtxt(fn)
            
        if conn_type == 'ee':
            self.conn_list_loaded[0] = True
        elif conn_type == 'ei':
            self.conn_list_loaded[1] = True
        elif conn_type == 'ie':
            self.conn_list_loaded[2] = True
        elif conn_type == 'ii':
            self.conn_list_loaded[3] = True



    def load_tuning_prop(self):
        self.tp_exc = np.loadtxt(self.params['tuning_prop_means_fn'])
        self.tp_inh = np.loadtxt(self.params['tuning_prop_inh_fn'])
