echo 'Starting at' `date`
python prepare_tuning_prop.py
#python prepare_selective_inhibition.py
#mpirun -np 8 python prepare_spike_trains.py
#echo 'Preparation stopped at' `date`
#mpirun -np 8 python prepare_connections.py
mpirun -np 8 python NetworkSimModuleNoColumns.py
#python NetworkSimModuleNoColumns.py
python plot_prediction.py
python merge_connlist_ee.py
python analyse_simple.py
python analyse_input.py
#python plot_connlist_as_colormap.py
python plot_weight_and_delay_histogram.py
python get_conductance_matrix.py 0
python merge_connlists.py
python plot_connlist_as_colormap.py 'ee'
python plot_connlist_as_colormap.py 'ei'
python plot_connlist_as_colormap.py 'ie'
python plot_connlist_as_colormap.py 'ii'
#python plot_input.py 205
#python plot_input.py 245
echo 'Stopping at' `date`
