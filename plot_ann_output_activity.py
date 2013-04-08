import sys
import numpy as np
import utils
import pylab
from Plotter import BasicPlotter


class PlotOutputActivity(BasicPlotter):
    """
    plot the ANN activity after training
    and the predicted resulting direction
    """

    def __init__(self, iteration =None, **kwargs):
        BasicPlotter.__init__(self, **kwargs)
        if iteration == None:
            iteration = 0
        activity_fn = self.params['activity_folder'] + 'output_activity_%d.dat' % (iteration)
        prediction_fn = self.params['activity_folder'] + 'prediction_%d.dat' % (iteration)
        prediction_error_fn = self.params['activity_folder'] + 'prediction_error_%d.dat' % (iteration)
        print 'activity_fn:', activity_fn
        print 'prediction_fn:', prediction_fn
        print 'prediction_error_fn:', prediction_error_fn
            
        self.activity = np.loadtxt(activity_fn)
        self.prediction = np.loadtxt(prediction_fn)
        self.prediction_error = np.loadtxt(prediction_error_fn)
        self.iteration = iteration
        rcParams = { 'axes.labelsize' : 16,
                    'label.fontsize': 20,
                    'legend.fontsize': 9}
        pylab.rcParams.update(rcParams)
        self.t_axis = self.prediction[:, 0]
        training_input_folder = "%sTrainingInput_%d/" % (self.params['folder_name'], iteration)
        input_params = np.loadtxt(self.params['parameters_folder'] + 'input_params.txt')
        self.stim_params = input_params[self.iteration, :]

#        self.vx_tuning = self.tuning_prop[:, 2]
#        self.vy_tuning = self.tuning_prop[:, 3]

#        vx_min, vx_max = self.vx_tuning.min(), self.vx_tuning.max()
#        vy_min, vy_max = self.vy_tuning.min(), self.vy_tuning.max()
#        n_vx_bins, n_vy_bins = 20, 20
#        vx_grid = np.linspace(vx_min, vx_max, n_vx_bins, endpoint=True)
#        vy_grid = np.linspace(vy_min, vy_max, n_vy_bins, endpoint=True)
#        self.calculate_v_predicted()

        self.create_fig()

    def plot_data_vs_time(self, data, **kwargs):
        xlabel = kwargs.get('xlabel', 'Time [ms]')
        ylabel = kwargs.get('ylabel', 'y')
        update_subfig_cnt = kwargs.get('update_subfig_cnt', True)

        label = kwargs.get('label', None)
        self.ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, self.subfig_cnt)
        self.ax.plot(self.t_axis, data, lw=2, label=label)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        if update_subfig_cnt :
            self.update_subfig_cnt()

        if label != None:
            self.ax.legend()

    def set_title(self):
        title = 'Stimulus vx=%.2f, vy=%.2f' % (self.stim_params[2], self.stim_params[3])
        self.ax.set_title(title)



    def plot_stim_prediction_as_quiver(self):
        fig = pylab.figure()
        ax = fig.add_subplot(111)

        vx = self.prediction[:, 1].mean()
        vx_std = self.prediction[:, 1].std()
        vy = self.prediction[:, 2].mean()
        vy_std = self.prediction[:, 2].std()
        scale = 1.

        stim_color = 'k'
        pred_color = 'r'
        std_color = 'b'

        ax.quiver(0.5, 0.5, vx+vx_std, vy-vy_std, \
              angles='xy', scale_units='xy', scale=scale, color=std_color, headwidth=4, pivot='middle')
        std = ax.quiver(0.5, 0.5, vx-vx_std, vy+vy_std, \
              angles='xy', scale_units='xy', scale=scale, color=std_color, headwidth=4, pivot='middle')

        pred = ax.quiver(0.5, 0.5, vx, vy, \
              angles='xy', scale_units='xy', scale=scale, color=pred_color, headwidth=4, pivot='middle')

        stim = ax.quiver(0.5, 0.5, self.stim_params[2], self.stim_params[3], 
              angles='xy', scale_units='xy', scale=scale, color=stim_color, headwidth=4, pivot='middle')

#        ax.quiverkey(std, 
#        ax.legend((std, pred, stim), \
#                'Prediction variation over time', \
#                'Average network prediction', \
#                'Stimulus')
        x_lim = (0., 1.)
        y_lim = (0., 1.)
        ax.annotate('Stimulus',                         (.6, .8), fontsize=12, color=stim_color)
        ax.annotate('Prediction variation\nover time',  (.6, .7), fontsize=12, color=std_color)
        ax.annotate('Average network prediction',       (.6, .6), fontsize=12, color=pred_color)
            
        #(0+.1*self.stim_params[2], 0.+0.1*self.stim_params[1]), fontsize=12, color=stim_color)
#        y_lim = (-.9 * self.stim_params[3], .9 * self.stim_params[3])
        print 'stim_params', self.stim_params 
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        



if __name__ == '__main__':
    if (len(sys.argv) < 2):
        iteration = 0
    else:
        iteration = int(sys.argv[1])

    P1 = PlotOutputActivity(iteration, n_fig_x=1, n_fig_y=3)
    print 'debug vx:', P1.prediction[:, 1]
    P1.plot_data_vs_time(P1.prediction[:, 1], ylabel='$v_x$', label='Prediction', update_subfig_cnt=False)
    print 'debug', P1.stim_params[2] * np.ones(P1.t_axis.size)
    P1.plot_data_vs_time(P1.stim_params[2] * np.ones(P1.t_axis.size), ylabel='$v_x$', label='vx_stimulus')
#    P1.ax.set_ylim((0., P1.stim_params[2]*1.05))
    P1.ax.set_ylim((P1.prediction[:, 1].min()*0.95, P1.stim_params[2]*1.05))
    P1.set_title()



    P1.plot_data_vs_time(P1.prediction[:, 2], ylabel='$v_y$', label='Prediction',update_subfig_cnt=False)
    P1.plot_data_vs_time(P1.stim_params[3] * np.ones(P1.t_axis.size), ylabel='$v_y$',label='vy_stimulus')
#    P1.ax.set_ylim((P1.prediction[:, 2].min(), P1.stim_params[3]*1.05))

    P1.plot_data_vs_time(P1.prediction_error[:, 3], ylabel='$|v_{diff}|$', label='Absolute prediction error')
#    P1.plot_data_vs_time(label='vy_stimulus')

    output_fn = P1.params['figures_folder'] + 'ann_prediction_%d.png' % (iteration)
    print 'Saving prediction figure to:', output_fn

    P1.plot_stim_prediction_as_quiver()

    pylab.savefig(output_fn)


    

#    idx = [85, 161, 71, 339]
#    for i in xrange(len(idx)):
#        cell = idx[i]
#        P2.plot_data_vs_time(P2.activity[:, cell], label='%d' % cell, ylabel='Activity')
#    output_fn = P2.params['figures_folder'] + 'ann_sample_activities_%d.png' % (iteration)
#    print 'Saving prediction figure to:', output_fn
    pylab.show()


#n_fig_x = 2
#n_fig_y = 4
#n_plots = n_fig_x * n_fig_y
#fig = pylab.figure()

#np.random.seed(0)




#ax = fig.add_subplot(221)
#ax.plot(t_axis, vx_pred)

#ax = fig.add_subplot(222)
#ax.plot(t_axis, vy_pred)

#plot_grid_vs_time(vx_pred_binned)



#pylab.show()
