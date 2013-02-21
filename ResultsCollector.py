import numpy as np
import os
import simulation_parameters
import NeuroTools.parameters as NTP
import pylab
import utils

try:
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size

except:
    USE_MPI = False
    comm = None
    pc_id, n_proc = 0, 1


class ResultsCollector(object):

    def __init__(self, params):
        self.params = params
        self.param_space = {}
        self.dirs_to_process = []
        pass

    def set_dirs_to_process(self, list_of_dir_names):

        for i_, folder in enumerate(list_of_dir_names):
            self.dirs_to_process.append(folder)
            self.param_space[i_] = {}


    def collect_files(self):
        all_dirs = []
        for f in os.listdir('.'):
            if os.path.isdir(f):
                all_dirs.append(f)

        self.mandatory_files = ['Spikes/exc_spikes_merged_.ras', \
                'Parameters/simulation_parameters.info', \
                'Parameters/tuning_prop_means.prm']

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
                self.dirs_to_process.append((dir_name, sim_id, {}))
                sim_id += 1

    def get_xvdiff_integral(self):

        self.xdiff_integral = np.zeros(len(self.dirs_to_process))
        self.vdiff_integral = np.zeros(len(self.dirs_to_process))
        results_sub_folder = 'Data/'
        fn_base_x = self.params['xdiff_vs_time_fn']
        fn_base_v = self.params['vdiff_vs_time_fn']

        for i_, folder in enumerate(self.dirs_to_process):
            # if self.dirs_to_process has been created by collect_files()
#            fn_x = folder[0] + '/' + results_sub_folder + fn_base_x
#            fn_v = folder[0] + '/' + results_sub_folder + fn_base_v
            fn_x = folder + '/' + results_sub_folder + fn_base_x
            fn_v = folder + '/' + results_sub_folder + fn_base_v
            xdiff = np.loadtxt(fn_x)
            vdiff = np.loadtxt(fn_v)
            self.xdiff_integral[i_] = xdiff[:, 1].sum()
            self.vdiff_integral[i_] = vdiff[:, 1].sum()


    def get_parameter(self, param_name):
        """
        For all simulations (in self.dirs_to_process) get the according parameter value
        """
        for i_, folder in enumerate(self.dirs_to_process):
            param_fn = folder + '/Parameters/simulation_parameters.info'
            param_fn = utils.convert_to_url(param_fn)
            param_dict = NTP.ParameterSet(param_fn)
            value = param_dict[param_name]
            self.param_space[i_][param_name] = value
            

    def plot_param_vs_xvdiff_integral(self, param_name, xv='x'):

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        
        if xv == 'x':
            xvdiff_integral = self.xdiff_integral
            title = '$\int_0^{t_{sim}} |\\vec{x}_{stim}(t) - \\vec{x}_{prediction}(t)| dt$ vs. %s' % param_name
        else:
            xvdiff_integral = self.vdiff_integral
            title = 'Integral of $|\\vec{v}_{stim}(t) - \\vec{v}_{prediction}(t)|$ vs. %s ' % param_name

        x_data = np.zeros(len(self.dirs_to_process))
        y_data = xvdiff_integral

        for i_, folder in enumerate(self.dirs_to_process):
            param_value = self.param_space[i_][param_name]
            x_data[i_] = param_value

        print 'debug x', x_data
        print 'debug y', y_data
        ax.plot(x_data, y_data, 'o')
        ax.set_xlim((x_data.min() * .9, x_data.max() * 1.1))
        ax.set_ylim((y_data.min() * .9, y_data.max() * 1.1))
        ax.set_xlabel(param_name, fontsize=18)
        ax.set_ylabel('Integral %s' % xv)
        ax.set_title(title)
        pylab.show()
            


    def get_cgxv(self):

        results_sub_folder = 'Data/'
        fn_base = 'scalar_products_between_tuning_prop_and_cgxv.dat'
        self.cgx = np.zeros((len(self.dirs_to_process), 2))
        self.cgv = np.zeros((len(self.dirs_to_process), 2))
        for i_, folder in enumerate(self.dirs_to_process):
            # if self.dirs_to_process has been created by collect_files()
#            fn = folder[0] + '/' + results_sub_folder + fn_base
            fn = folder + '/' + results_sub_folder + fn_base
            d = np.loadtxt(fn)
            self.cgx[i_, 0] = d[:, 0].mean()
            self.cgx[i_, 1] = d[:, 0].std() / np.sqrt(d[:, 0].size)

            self.cgv[i_, 0] = d[:, 1].mean()
            self.cgv[i_, 1] = d[:, 1].std() / np.sqrt(d[:, 1].size)


    def plot_cgxv_vs_xvdiff(self):

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        for i_, folder in enumerate(self.dirs_to_process):
#            y = self.cgx[i_, 0]
#            yerr = self.cgx[i_, 1]

            y = self.cgv[i_, 0]
            yerr = self.cgv[i_, 1]
            x = self.xdiff_integral[i_]
#            x = self.vdiff_integral[i_]
            
            print 'debug ', x, y, folder
            ax.errorbar(x, y, yerr=yerr, ls='o', c='b')
            ax.plot(x, y, 'o', c='b')

        pylab.show() 



    def build_parameter_space(self):

        # take a sample simulation_parameters.info file to generate all possible keys
        sample_fn = self.dirs_to_process[0][0] + '/Parameters/simulation_parameters.info'
        sample_dict = NTP.ParameterSet(sample_fn)
        all_param_names = sample_dict.keys()



    def check_for_correctness(self):
        """
        This function checks if the folder name has the same value for a 
        given parameter (e.g. 'wee') as in the simulation_parameters.info file
        """
        for dirname, sim_id in self.dirs_to_process:
            idx = dirname.find('wee')
            idx2 = dirname[idx:].find('_')
            val_in_folder = float(dirname[idx+3:idx+idx2])

            fn = dirname + '/Parameters/simulation_parameters.info'
            param_dict = NTP.ParameterSet(fn)
            if param_dict['w_tgt_in_per_cell_ee'] != val_in_folder:
                print 'Mismatch in folder name and parameter dict:', dirname



if __name__ == '__main__':
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.params
    RC = ResultsCollector(params)
    RC.collect_files()
    RC.get_xvdiff_integral()
    RC.get_cgxv()

#print "RC.dirs_to_process", RC.dirs_to_process
#RC.check_for_correctness()

