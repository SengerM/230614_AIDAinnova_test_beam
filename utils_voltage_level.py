from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import logging
from parse_waveforms import load_parsed_from_waveforms_from_EUDAQ_run
import json
import pandas
import utils_batch_level
import DUT_analysis

def load_waveforms_data_for_voltage_point(voltage_point_dn:DatanodeHandler, where:str=None, variables:list=None):
	voltage_point_dn.check_datanode_class('voltage_point')
	
	DUT_analysis_dn = voltage_point_dn.parent
	DUT_analysis_dn.check_datanode_class('DUT_analysis')
	
	TB_batch_dn = DUT_analysis_dn.parent
	TB_batch_dn.check_datanode_class('TB_batch')
	
	setup_config = utils_batch_level.load_setup_configuration_info(TB_batch_dn)
	
	DUT_analysis_configuration_metadata = DUT_analysis.load_DUT_configuration_metadata(DUT_analysis_dn)
	this_DUT_chubut_channels = DUT_analysis_configuration_metadata['chubut_channels_numbers']
	this_DUT_plane_number = DUT_analysis_configuration_metadata['plane_number']
	
	
	with open(voltage_point_dn.path_to_directory_of_task('EUDAQ_runs')/'runs.json', 'r') as ifile:
		EUDAQ_runs_from_this_voltage = json.load(ifile)
	if not isinstance(EUDAQ_runs_from_this_voltage, list) or len(EUDAQ_runs_from_this_voltage) == 0:
		raise RuntimeError(f'No EUDAQ runs associated to this voltage...')
	
	setup_config_this_DUT = setup_config.query(f'plane_number=={this_DUT_plane_number} and chubut_channel in {sorted(this_DUT_chubut_channels)}')
	
	SQL_query_for_n_CAEN_and_CAEN_n_channel = ' or '.join([f'(n_CAEN=={row["n_CAEN"]} and CAEN_n_channel=={row["CAEN_n_channel"]})' for i,row in setup_config_this_DUT.iterrows()])
	
	if where is not None:
		where = '(' + SQL_query_for_n_CAEN_and_CAEN_n_channel + ') and ' + where
	else:
		where = SQL_query_for_n_CAEN_and_CAEN_n_channel
	
	loaded_data = []
	for EUDAQ_run_dn in TB_batch_dn.list_subdatanodes_of_task('EUDAQ_runs'):
		if EUDAQ_run_dn.datanode_name not in EUDAQ_runs_from_this_voltage:
			continue
		data = load_parsed_from_waveforms_from_EUDAQ_run(EUDAQ_run_dn, where=where, variables=variables)
		data['EUDAQ_run'] = int(EUDAQ_run_dn.datanode_name.split('_')[0].replace('run',''))
		data.set_index('EUDAQ_run', append=True, inplace=True)
		loaded_data.append(data)
	loaded_data = pandas.concat(loaded_data)
	loaded_data = loaded_data.reorder_levels([loaded_data.index.names[-1]] + loaded_data.index.names[:-1])
	
	return loaded_data

# ~ def plot_waveforms_distributions(voltage_point_dn:DatanodeHandler, max_points_to_plot=9999, histograms=['Amplitude (V)','Collected charge (V s)','t_50 (s)','Rise time (s)','SNR','Time over 50% (s)'], scatter_plots=[('t_50 (s)','Amplitude (V)'),('Time over 50% (s)','Amplitude (V)')]):
	# ~ with voltage_point_dn.handle_task('plot_waveforms_distributions', check_datanode_class='voltage_point', check_required_tasks='EUDAQ_runs') as task:
		# ~ setup_config = utils_batch_level.load_setup_configuration_info(voltage_point_dn.parent.parent)
		# ~ setup_config
		# ~ save_histograms_here = task.path_to_directory_of_my_task/'histograms'
		# ~ save_histograms_here.mkdir()
		# ~ for var in histograms:
			# ~ data = load_waveforms_data_for_voltage_point(voltage_point_dn=voltage_point_dn, variables=[var])
			

if __name__ == '__main__':
	import sys
	import argparse
	from plotly_utils import set_my_template_as_default
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.INFO,
		format = '%(asctime)s|%(levelname)s|%(message)s',
		datefmt = '%H:%M:%S',
	)
	
	set_my_template_as_default()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--datanode',
		metavar = 'path', 
		help = 'Path to a datanode.',
		required = True,
		dest = 'datanode',
		type = Path,
	)
	args = parser.parse_args()
	
	data = load_waveforms_data_for_voltage_point(
		voltage_point_dn = DatanodeHandler(args.datanode),
		variables = ['Amplitude (V)'],
		where = '`Amplitude (V)` < -.2',
	)
	
	print(data)
