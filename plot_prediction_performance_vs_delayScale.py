import os
import numpy as np
import pylab




def plot_performance_vs_delayScale():
    folders_noBlank = 'LargeScaleModel_noBlank_AIII_scaleLatency0.15_wsigmax1.00e-01_wsigmav1.00e-01_wee3.00e-02_wei4.00e-02_wie7.00e-02_wii1.00e-02_delayScale'
    folders_withBlank = 'LargeScaleModel_AIII_scaleLatency0.15_wsigmax1.00e-01_wsigmav1.00e-01_wee3.00e-02_wei4.00e-02_wie7.00e-02_wii1.00e-02_delayScale'
    delays = [2, 5, 10, 20, 50]
    performance_noBlank = np.zeros(len(delays))
    performance = np.zeros(len(delays))
    for i_, delay in enumerate(delays):
        folder = folders_noBlank + str(delay) + '/'
        fn = folder + 'Data/xdiff_vs_time.dat'
        d = np.loadtxt(fn)
        performance_noBlank[i_] = d[:, 1].sum()

        folder = folders_withBlank + str(delay) + '/'
        fn = folder + 'Data/xdiff_vs_time.dat'
        d = np.loadtxt(fn)
        performance[i_] = d[:, 1].sum()
    #    print delay, d[:, 1].sum()

    fig = pylab.figure()
    ax = fig.add_subplot(111)

    ax.plot(delays, performance_noBlank, lw=3, label='no blank')
    ax.plot(delays, performance, lw=3, label='with blank')
    ax.set_xlabel('delay scale')
    ax.set_ylabel('$\int_0^{t_{sim}} |x_{stim}(t) - x_{prediction}(t)| dt$')
    ax.legend()


def plot_performance_vs_scaleLatency():
    def get_folder_name(l):
        return 'LargeScaleModel_AIII_scaleLatency%.2f_wsigmax1.00e-01_wsigmav1.00e-01_wee3.00e-02_wei4.00e-02_wie7.00e-02_wii1.00e-02_delayScale5/' % l
    latencies = [0.10, 0.15, 0.20, 0.30, 0.40]#, 0.50]
    performance = np.zeros(len(latencies))
    for i_, latency in enumerate(latencies):
        folder = get_folder_name(latency)
        fn = folder + 'Data/xdiff_vs_time.dat'
        print fn, os.path.exists(fn)
        d = np.loadtxt(fn)
        performance[i_] = d[:, 1].sum()

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.plot(latencies, performance, lw=3)
#    ax.plot(delays, performance, lw=3, label='with blank')
    ax.set_xlabel('scale latency')
    ax.set_title('$\int_0^{t_{sim}} |x_{stim}(t) - x_{prediction}(t)| dt$ vs scale_latency')


plot_performance_vs_scaleLatency()
pylab.show()
