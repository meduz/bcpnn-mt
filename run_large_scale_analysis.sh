echo 'Starting at' `date`
#echo 'Preparation stopped at' `date`
python plot_prediction.py
python prepare_tuning_prop.py
python merge_connlists.py
python analyse_simple.py
#python analyse_input.py
#python plot_connlist_as_colormap.py
python merge_connlists.py
python plot_weight_and_delay_histogram.py ee 
python plot_weight_and_delay_histogram.py ei
python plot_weight_and_delay_histogram.py ie
python plot_weight_and_delay_histogram.py ii
#python get_conductance_matrix.py 0
python plot_spike_histogram.py exc
python plot_spike_histogram.py inh
python plot_connectivity_profile.py
#python plot_connlist_as_colormap.py 'ee'
#python plot_connlist_as_colormap.py 'ei'
#python plot_connlist_as_colormap.py 'ie'
#python plot_connlist_as_colormap.py 'ii'
#python plot_input.py 205
#python plot_input.py 245
echo 'Stopping at' `date`
