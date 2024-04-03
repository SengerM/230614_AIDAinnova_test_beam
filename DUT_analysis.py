from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import pandas
import logging
import utils_batch_level
import json
import utils_voltage_level
import dominate
import dominate.tags as tags
import corry_stuff

def create_DUT_analysis(TB_batch_dn:DatanodeHandler, DUT_name:str, plane_number:int, chubut_channel_numbers:set):
	"""Creates a datanode to handle the analysis of one DUT.
	
	Arguments
	---------
	TB_batch_dn: DatanodeHandler
		A `DatanodeHandler` pointing to a test beam batch datanode.
	DUT_name: str
		The name of the DUT. It can be anything, and will be used later on
		to identify this analysis.
	plane_number: int
		Plane number as it is in the spreadsheet from the test beam data.
	chubut_channel_numbers: set of int
		Chubut channel numbers from the pixels that belong to this DUT. 
		This is needed because we have tested more than one DUT per plane 
		at DESY, which is not something standard, so now we need to 
		separate them.
	"""
	with TB_batch_dn.handle_task('DUTs_analyses', check_datanode_class='TB_batch', check_required_tasks='batch_info', keep_old_data=True) as task_handler:
		setup_config = utils_batch_level.load_setup_configuration_info(TB_batch_dn)
		if plane_number not in setup_config['plane_number']:
			raise RuntimeError(f'`plane_number` {plane_number} not found in the setup_config from batch {repr(str(TB_batch_dn.pseudopath))}. ')
		if any([ch not in setup_config.query(f'plane_number=={plane_number}')['chubut_channel'] for ch in chubut_channel_numbers]):
			raise RuntimeError(f'At least one `chubut_channel_numbers` {chubut_channel_numbers} not present in the setup_config of batch {repr(str(TB_batch_dn.pseudopath))}. ')
		DUT_analysis_dn = task_handler.create_subdatanode(DUT_name, subdatanode_class='DUT_analysis')
		with DUT_analysis_dn.handle_task('setup_config_metadata') as setup_config_metadata_task:
			with open(setup_config_metadata_task.path_to_directory_of_my_task/'metadata.json', 'w') as ofile:
				json.dump(
					dict(
						plane_number = plane_number,
						chubut_channels_numbers = chubut_channel_numbers,
					), 
					ofile, 
					indent = '\t',
				)
		logging.info(f'DUT analysis {repr(str(DUT_analysis_dn.pseudopath))} was created. ✅')

def create_voltage_point(DUT_analysis_dn:DatanodeHandler, voltage:int, EUDAQ_runs:set):
	"""Create a new voltage point within a DUT analysis.
	
	Arguments
	---------
	DUT_analysis_dn: DatanodeHandler
		A `DatanodeHandler` pointing to a DUT analysis in which to create
		the new voltage	point.
	voltage: int
		The voltage value, e.g. `150`.
	EUDAQ_runs: set of int
		A set of int with the EUDAQ run numbers to be linked to this voltage.
	"""
	with DUT_analysis_dn.handle_task('voltages', check_datanode_class='DUT_analysis', keep_old_data=True) as task:
		TB_batch_dn = DUT_analysis_dn.parent
		EUDAQ_runs_within_this_batch = {int(r.datanode_name.split('_')[0].replace('run','')):r.datanode_name for r in TB_batch_dn.list_subdatanodes_of_task('EUDAQ_runs')}
		
		if any([_ not in EUDAQ_runs_within_this_batch for _ in EUDAQ_runs]):
			raise RuntimeError(f'At least one of the runs {EUDAQ_runs} is not available in batch {repr(str(TB_batch_dn.pseudopath))}. Available runs found are: {sorted(EUDAQ_runs_within_this_batch)}. ')
		
		voltage_dn = task.create_subdatanode(f'{voltage}V', subdatanode_class='voltage_point')
		
		with voltage_dn.handle_task('EUDAQ_runs') as EUDAQ_runs_task:
			with open(EUDAQ_runs_task.path_to_directory_of_my_task/'runs.json', 'w') as ofile:
				json.dump(
					[EUDAQ_runs_within_this_batch[_] for _ in EUDAQ_runs],
					ofile,
					indent = '\t',
				)
	logging.info(f'Voltage point {repr(str(voltage_dn.pseudopath))} was created. ✅')

def load_DUT_configuration_metadata(DUT_analysis_dn:DatanodeHandler):
	DUT_analysis_dn.check_datanode_class('DUT_analysis')
	with open(DUT_analysis_dn.path_to_directory_of_task('setup_config_metadata')/'metadata.json', 'r') as ifile:
		loaded = json.load(ifile)
		this_DUT_chubut_channels = loaded['chubut_channels_numbers']
		this_DUT_plane_number = loaded['plane_number']
	if len(this_DUT_chubut_channels) == 0:
		raise RuntimeError(f'No `chubut channels` associated with DUT in {repr(str(DUT_analysis_dn.pseudopath))}. ')
	if not isinstance(this_DUT_plane_number, int):
		raise RuntimeError(f'Cannot determine plane number for DUT in {repr(str(DUT_analysis_dn.pseudopath))}. ')
	
	return loaded

def plot_waveforms_distributions(DUT_analysis_dn:DatanodeHandler, max_points_to_plot_per_voltage=9999, histograms=['Amplitude (V)','Collected charge (V s)','t_50 (s)','Rise time (s)','SNR','Time over 50% (s)'], scatter_plots=[('t_50 (s)','Amplitude (V)'),('Time over 50% (s)','Amplitude (V)')]):
	
	with DUT_analysis_dn.handle_task('plot_waveforms_distributions', 'DUT_analysis', 'voltages') as task:
		for voltage_dn in DUT_analysis_dn.list_subdatanodes_of_task('voltages'):
			utils_voltage_level.plot_waveforms_distributions(
				voltage_point_dn = voltage_dn,
				max_points_to_plot = max_points_to_plot_per_voltage,
				histograms = histograms,
				scatter_plots = scatter_plots,
			)
		
		voltages = sorted(DUT_analysis_dn.list_subdatanodes_of_task('voltages'), key=DatanodeHandler.datanode_name.__get__)
		
		for kind_of_plot in {'histograms','scatter_plots'}:
			save_plots_here = task.path_to_directory_of_my_task/kind_of_plot
			save_plots_here.mkdir()
			
			for plot_file_name in [_.name for _ in (DUT_analysis_dn.list_subdatanodes_of_task('voltages')[0].path_to_directory_of_task('plot_waveforms_distributions')/kind_of_plot).iterdir()]:
				doc = dominate.document(title=plot_file_name.split('.')[0])
				with doc:
					tags.h1(plot_file_name.split('.')[0] + ' distributions')
					tags.h3(str(DUT_analysis_dn.pseudopath))
					for voltage_dn in voltages:
						tags.iframe(
							src = Path('../..')/((voltage_dn.path_to_directory_of_task('plot_waveforms_distributions')/kind_of_plot/plot_file_name).relative_to(DUT_analysis_dn.path_to_datanode_directory)),
							style = 'height: 90vh; width: 100%; min-height: 666px; min-width: 666px; border: 0;',
						)
				with open(save_plots_here/plot_file_name, 'w') as ofile:
					print(doc, file=ofile)

def load_hits_on_DUT_from_voltage_point(voltage_point_dn:DatanodeHandler):
	"""Load all the hits on the DUT for this voltage point.
	
	Arguments
	---------
	voltage_point_dn: DatanodeHandler
		A `DatanodeHandler` pointing to a voltage point, inside a DUT
		analysis, inside a test beam batch.
	
	Returns
	-------
	hits: pandas.DataFrame
		A data frame of the form
		````
		                          x (m)     y (m)
		n_run n_event n_track                    
		226   45      0       -0.002738  0.001134
			  79      0       -0.002727 -0.002765
			  92      0       -0.003044  0.001085
			  128     0       -0.002434  0.000308
			  132     0       -0.002659 -0.000086
		...                         ...       ...
		227   113475  2       -0.003013 -0.000015
			  113478  0       -0.001994  0.000780
			  113479  0       -0.001950 -0.000560
			  113483  0       -0.002547  0.000459
			  113487  0        0.001486 -0.002903

		[24188 rows x 2 columns]
		```
	"""
	voltage_point_dn.check_datanode_class('voltage_point')
	
	DUT_analysis_dn = voltage_point_dn.parent
	DUT_analysis_dn.check_datanode_class('DUT_analysis')
	
	TB_batch_dn = DUT_analysis_dn.parent
	TB_batch_dn.check_datanode_class('TB_batch')
	
	DUT_configuration_metadata = load_DUT_configuration_metadata(DUT_analysis_dn)
	setup_config = utils_batch_level.load_setup_configuration_info(TB_batch_dn)
	
	DUT_name_as_it_is_in_raw_files = set(setup_config.query(f'plane_number == {DUT_configuration_metadata["plane_number"]}')['DUT_name'])
	if len(DUT_name_as_it_is_in_raw_files) != 1:
		raise RuntimeError('I was expecting a unique DUT name per plane, but this does not seem to be the case... ')
	DUT_name_as_it_is_in_raw_files = list(DUT_name_as_it_is_in_raw_files)[0]
	
	with open(voltage_point_dn.path_to_directory_of_task('EUDAQ_runs')/'runs.json', 'r') as ifile:
		EUDAQ_runs = json.load(ifile)
	if not isinstance(EUDAQ_runs, list) or len(EUDAQ_runs) == 0:
		raise RuntimeError(f'Cannot read what are the EUDAQ runs for voltage point {repr(str(voltage_point_dn.pseudopath))}. ')
	EUDAQ_runs = [dn for dn in TB_batch_dn.list_subdatanodes_of_task('EUDAQ_runs') if dn.datanode_name in EUDAQ_runs]
	
	hits = []
	for EUDAQ_run_dn in EUDAQ_runs:
		_ = corry_stuff.load_hits_on_DUT_from_EUDAQ_run(EUDAQ_run_dn, DUT_name = DUT_name_as_it_is_in_raw_files)
		_ = pandas.concat({int(EUDAQ_run_dn.datanode_name.split('_')[0].replace('run','')): _}, names=['n_run'])
		hits.append(_)
	hits = pandas.concat(hits)
	
	return hits

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
		help = 'Path to a TB_batch datanode.',
		dest = 'datanode',
		type = Path,
	)
	args = parser.parse_args()
	
	load_hits_on_DUT_from_voltage_point(
		voltage_point_dn = DatanodeHandler(args.datanode), 
	)
