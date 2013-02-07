import numpy as np
import os
import simulation_parameters
import NeuroTools.parameters as NTP

try:
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size

except:
    USE_MPI = False
    comm = None
    pc_id, n_proc = 0, 1

#        network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
#        params = network_params.params

class ResultsCollector(object):

    def __init__(self):
        pass


    def collect_files(self):
        all_dirs = []
        for f in os.listdir('.'):
            if os.path.isdir(f):
                all_dirs.append(f)

        self.mandatory_files = ['Spikes/exc_spikes_merged_.ras', \
                'Parameters/simulation_parameters.info', \
                'Parameters/tuning_prop_means.prm']

        self.dirs_to_process = []
        sim_id = 0
        for dir_name in all_dirs:
            # check if all necessary files exist
            check_passed = True
            for fn in self.mandatory_files:
                fn_ = dir_name + '/' + fn
#                print 'checking', fn_
                if not os.path.exists(fn_):
                    check_passed = False
            if check_passed:
                self.dirs_to_process.append((dir_name, sim_id))
                sim_id += 1


    def build_parameter_space(self):

        # take a sample simulation_parameters.info file to generate all possible keys
        sample_fn = self.dirs_to_process[0][0] + '/Parameters/simulation_parameters.info'
        sample_dict = NTP.ParameterSet(sample_fn)
        all_param_names = sample_dict.keys()

    def check_for_correctness(self):
        for dirname, sim_id in self.dirs_to_process:
            idx = dirname.find('wee')
            idx2 = dirname[idx:].find('_')
            print 'debug', dirname, idx, idx2
            print 'debug', dirname[idx+3:idx+idx2]
            val_in_folder = float(dirname[idx+3:idx+idx2])

            fn = dirname + '/Parameters/simulation_parameters.info'
            param_dict = NTP.ParameterSet(fn)
            if param_dict['w_tgt_in_per_cell_ee'] == val_in_folder:
                print 'Mismatch in folder name and parameter dict:', dirname



RC = ResultsCollector()
RC.collect_files()
RC.check_for_correctness()

