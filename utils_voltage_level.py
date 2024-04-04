from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import logging
from parse_waveforms import load_parsed_from_waveforms_from_EUDAQ_run
import json
import pandas
import utils_batch_level
import DUT_analysis
import plotly.express as px
import corry_stuff

def load_waveforms_data(voltage_point_dn:DatanodeHandler, where:str=None, variables:list=None):
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

def plot_waveforms_distributions(voltage_point_dn:DatanodeHandler, max_points_to_plot=9999, histograms=['Amplitude (V)','Collected charge (V s)','t_50 (s)','Rise time (s)','SNR','Time over 50% (s)'], scatter_plots=[('t_50 (s)','Amplitude (V)'),('Time over 50% (s)','Amplitude (V)')]):
	with voltage_point_dn.handle_task('plot_waveforms_distributions', check_datanode_class='voltage_point', check_required_tasks='EUDAQ_runs') as task:
		setup_config = utils_batch_level.load_setup_configuration_info(voltage_point_dn.parent.parent)
		DUT_config_metadata = DUT_analysis.load_DUT_configuration_metadata(voltage_point_dn.parent)
		setup_config = setup_config.query(f'plane_number=={DUT_config_metadata["plane_number"]} and chubut_channel in {sorted(DUT_config_metadata["chubut_channels_numbers"])}') # Keep only relevant part for this DUT.
		setup_config.set_index(['n_CAEN','CAEN_n_channel'], inplace=True)
		
		save_histograms_here = task.path_to_directory_of_my_task/'histograms'
		save_histograms_here.mkdir()
		for var in histograms:
			data = load_waveforms_data(voltage_point_dn=voltage_point_dn, variables=[var])
			data = data.join(setup_config['DUT_name_rowcol'])
			
			logging.info(f'Plotting distribution of {var} in {voltage_point_dn.pseudopath}...')
			fig = px.ecdf(
				title = f'{var.split("(")[0]} distribution<br><sup>{voltage_point_dn.pseudopath}</sup>',
				data_frame = data.sample(n=max_points_to_plot).reset_index().sort_values('DUT_name_rowcol'),
				x = var,
				marginal = 'histogram',
				color = 'DUT_name_rowcol',
			)
			fig.write_html(
				save_histograms_here/f'{var}.html',
				include_plotlyjs = 'cdn',
			)
		
		save_scatter_plots_here = task.path_to_directory_of_my_task/'scatter_plots'
		save_scatter_plots_here.mkdir()
		for xvar, yvar in scatter_plots:
			data = load_waveforms_data(voltage_point_dn=voltage_point_dn, variables=[xvar,yvar])
			data = data.join(setup_config['DUT_name_rowcol'])
			
			logging.info(f'Plotting {yvar} vs {xvar} scatter plot in {voltage_point_dn.pseudopath}...')
			fig = px.scatter(
				title = f'{yvar.split("(")[0]} vs {xvar.split("(")[0]}<br><sup>{voltage_point_dn.pseudopath}</sup>',
				data_frame = data.sample(n=max_points_to_plot).reset_index().sort_values('DUT_name_rowcol'),
				x = xvar,
				y = yvar,
				color = 'DUT_name_rowcol',
				hover_data = ['EUDAQ_run','n_event'],
			)
			fig.write_html(
				save_scatter_plots_here/f'{yvar} vs {xvar}.html',
				include_plotlyjs = 'cdn',
			)

def load_hits_on_DUT(voltage_point_dn:DatanodeHandler):
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
	
	DUT_configuration_metadata = DUT_analysis.load_DUT_configuration_metadata(DUT_analysis_dn)
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
		_ = pandas.concat({int(EUDAQ_run_dn.datanode_name.split('_')[0].replace('run','')): _}, names=['EUDAQ_run'])
		hits.append(_)
	hits = pandas.concat(hits)
	
	return hits

def plot_hits(voltage_point_dn:DatanodeHandler, amplitude_threshold:float):
	"""Plot hits projected onto the DUT.
	
	Arguments
	---------
	voltage_point_dn: DatanodeHandler
		A `DatanodeHandler` pointing to a voltage point.
	amplitude_threshold: float
		Threshold in the amplitude to consider the activation of the pixels
		in the DUT. The threshold is applied to negative values and the
		sign of `amplitude_threshold` is ignored, i.e. this works in this way:
		`'Amplitude (V)' < {-abs(amplitude_threshold)}`.
	"""
	with voltage_point_dn.handle_task('plot_hits', 'voltage_point') as task:
		setup_config = utils_batch_level.load_setup_configuration_info(voltage_point_dn.parent.parent)
		
		DUT_above_threshold = load_waveforms_data(
			voltage_point_dn = voltage_point_dn,
			where = f'`Amplitude (V)` < {-abs(amplitude_threshold)} AND `Time over 50% (s)`>1e-9',
		)
		
		tracks_on_DUT = load_hits_on_DUT(voltage_point_dn)
		
		for df in [DUT_above_threshold, tracks_on_DUT]:
			df.reset_index(inplace=True)
			df.set_index(['EUDAQ_run','n_event'], inplace=True)
		
		hits = DUT_above_threshold.join(tracks_on_DUT, on=['EUDAQ_run','n_event'], how='inner')
		
		hits = hits.reset_index(drop=False).set_index(['n_CAEN','CAEN_n_channel']).join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])['DUT_name_rowcol'])
		
		fig = px.scatter(
			title = f'Hits on DUT<br><sup>{voltage_point_dn.pseudopath}, amplitude<{-amplitude_threshold*1e3:.0f} mV</sup>',
			data_frame = hits.reset_index().sort_values('DUT_name_rowcol'),
			x = 'x (m)',
			y = 'y (m)',
			color = 'DUT_name_rowcol',
			hover_data = ['EUDAQ_run','n_event'],
		)
		fig.update_yaxes(
			scaleanchor = "x",
			scaleratio = 1,
		)
		fig.write_html(
			task.path_to_directory_of_my_task/'hits.html',
			include_plotlyjs = 'cdn',
		)
		logging.info(f'Plotted hits for {repr(str(voltage_point_dn.pseudopath))} ✅')

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
	
	plot_waveforms_distributions(DatanodeHandler(args.datanode))
