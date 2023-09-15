from pathlib import Path
import numpy
import pandas
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import plotly.graph_objects as go
import plotly.express as px
import utils
import logging
import sqlite3
import plotly_utils

def set_cols_and_rows(jitter_estimation, signals_connections):
	for i in ['index','columns']:
		jitter_estimation = pandas.concat(
			[	
				jitter_estimation,
				signals_connections.set_index(['n_CAEN','CAEN_n_channel'])[['DUT_name_rowcol_CAEN_and_trigger_group']],
			],
			ignore_index = False,
			axis = 1,
		)
		jitter_estimation.set_index('DUT_name_rowcol_CAEN_and_trigger_group', inplace=True)
		jitter_estimation.sort_index(inplace=True)
		jitter_estimation = jitter_estimation.T
	jitter_estimation = jitter_estimation.T
	jitter_estimation.index.set_names(names=None,inplace=True)
	return jitter_estimation

def jitter_estimation(bureaucrat:RunBureaucrat, max_events_to_plot:int=int(5e3), amplitude_threshold_volts:float=-20e-3, minimum_number_of_coincidences:int=100):
	bureaucrat.check_these_tasks_were_run_successfully(['parse_waveforms','batch_info'])
	
	INDEX_COLS = ['n_event','n_CAEN','CAEN_n_channel']
	
	logging.info(f'Reading data for run {bureaucrat.run_name}...')
	data = []
	CAENs_names = []
	columns_to_load = INDEX_COLS + ['t_50 (s)','Amplitude (V)']
	for sqlite_file_path in sorted((bureaucrat.path_to_directory_of_task('parse_waveforms')/'parsed_data').iterdir(), key=lambda x: x.stat().st_size):
		number_of_events_already_loaded = sum([len(_.index.get_level_values('n_event').drop_duplicates()) for _ in data])
		logging.info(f'Loading {sqlite_file_path.name}...')
		n_run = int(sqlite_file_path.name.split('_')[0].replace('run',''))
		sqlite_connection = sqlite3.connect(sqlite_file_path)
		df = pandas.read_sql(f'SELECT {",".join([f"`{_}`" for _ in columns_to_load])} FROM dataframe_table WHERE `Amplitude (V)`<={amplitude_threshold_volts} OR CAEN_n_channel in (16,17)', sqlite_connection)
		df.set_index(INDEX_COLS, inplace=True)
		df['n_run'] = n_run
		df.reset_index(inplace=True, drop=False)
		df.set_index(['n_run','n_event','n_CAEN','CAEN_n_channel'], inplace=True)
		data.append(df)
		df = pandas.read_pickle(bureaucrat.path_to_directory_of_task('parse_waveforms')/'CAENs_names'/sqlite_file_path.name.replace('.sqlite','_CAENs_names.pickle'))
		df = df.to_frame()
		df['n_run'] = n_run
		df.set_index('n_run',append=True,inplace=True)
		CAENs_names.append(df)
		
		number_of_events_already_loaded = sum([len(_.index.get_level_values('n_event').drop_duplicates()) for _ in data])
		logging.info(f'Loaded {sqlite_file_path.name}, {number_of_events_already_loaded} events up to now')
	data = pandas.concat(data)
	CAENs_names = pandas.concat(CAENs_names)
	CAENs_names['CAEN_name'] = CAENs_names['CAEN_name'].apply(lambda x: x.replace('CAEN_',''))
	signals_connections = utils.load_setup_configuration_info(bureaucrat)
	logging.info(f'{len(data.reset_index(drop=False)[["n_run","n_event"]].drop_duplicates())} events loaded for {bureaucrat.run_name}')
	
	data['Amplitude (V)'] *= -1
	data.rename(columns={'Amplitude (V)':'-Amplitude (V)'}, inplace=True)
	
	data = data.reset_index(['n_run','n_event'],drop=False).merge(signals_connections.set_index(['n_CAEN','CAEN_n_channel'])[['DUT_name','row','col','CAEN_name','DUT_name_rowcol']], on=['n_CAEN','CAEN_n_channel']).reset_index(drop=False).set_index(['n_run','n_event','n_CAEN','CAEN_n_channel'])
	
	signals_connections['DUT_name_rowcol_CAEN_and_trigger_group'] = signals_connections[['DUT_name_rowcol','CAEN_name','CAEN_trigger_group_n']].apply(lambda x: f'{x["DUT_name_rowcol"]} {x["CAEN_name"]}<sub>{x["CAEN_trigger_group_n"]}</sub>', axis=1)
	
	with bureaucrat.handle_task('jitter_estimation') as employee:
		logging.info('Plotting amplitudes distribution...')
		fig = px.ecdf(
			title = f'Amplitudes distribution when estimating jitter<br><sup>{bureaucrat.run_name}</br>',
			data_frame = data.sample(n=max_events_to_plot).reset_index(drop=False).sort_values(['DUT_name','row','col']).query('CAEN_n_channel not in [16,17]'),
			x = '-Amplitude (V)',
			facet_row = 'row',
			facet_col = 'col',
			color = 'DUT_name',
			hover_data = ['CAEN_name','n_run','n_event'],
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'amplitudes_distribution.html',
			include_plotlyjs = 'cdn',
		)
		
		hit_time = data['t_50 (s)']
		hit_time = hit_time.unstack(['n_CAEN','CAEN_n_channel'])
		n_hits = pandas.Series(index=hit_time.index, data=numpy.nansum(~numpy.isnan(hit_time), axis=1))
		
		logging.info('Calculating jitter...')
		jitter_estimation = hit_time.corr(method=lambda x,y: numpy.nanstd(x-y))
		n_coincidences = hit_time.corr(method=lambda x,y: numpy.count_nonzero(~numpy.isnan(x-y)))
		
		jitter_estimation = set_cols_and_rows(jitter_estimation, signals_connections)
		n_coincidences = set_cols_and_rows(n_coincidences, signals_connections)
		
		n_coincidences[numpy.isnan(n_coincidences)] = 0
		mask = numpy.zeros(n_coincidences.shape,dtype='bool')
		mask[numpy.diag_indices(len(n_coincidences))] = True
		n_coincidences[mask] = 0
		
		jitter_estimation[n_coincidences<minimum_number_of_coincidences] = float('NaN')
		
		logging.info('Plotting jitter...')
		df = jitter_estimation
		mask = numpy.zeros(df.shape,dtype='bool')
		mask[numpy.diag_indices(len(df))] = True
		mask[n_coincidences<minimum_number_of_coincidences] = True
		df[mask] = float('NaN')
		fig = plotly_utils.imshow_logscale(
			df,
			aspect = "auto",
			title = f'Jitter estimation<br><sup>{bureaucrat.run_name}, threshold={abs(amplitude_threshold_volts)*1e3} mV, N coincidences>{minimum_number_of_coincidences}</sup>',
			labels = dict(color=f'Jitter estimation (s)'),
			text_auto = '.2e',
		)
		fig.update_coloraxes(colorbar_title_side = 'right')
		fig.update_layout(
			xaxis = dict(showgrid=False),
			yaxis = dict(showgrid=False)
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/f'jitter_estimation.html',
			include_plotlyjs = 'cdn',
		)
		fig = plotly_utils.imshow_logscale(
			n_coincidences,
			aspect = "auto",
			hoverinfo_z_format = ':d',
			title = f'N coincidences<br><sup>{bureaucrat.run_name}, threshold={abs(amplitude_threshold_volts)*1e3} mV</sup>',
			labels = dict(color=f'Number of coincidences'),
			text_auto = ':.0f',
		)
		fig.update_coloraxes(colorbar_title_side = 'right')
		fig.update_layout(
			xaxis = dict(showgrid=False),
			yaxis = dict(showgrid=False)
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/f'coincidences.html',
			include_plotlyjs = 'cdn',
		)
		logging.info('Finished jitter estimation!')

if __name__=='__main__':
	import argparse
	import sys
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.DEBUG,
		format = '%(asctime)s|%(levelname)s|%(message)s',
		datefmt = '%Y-%m-%d %H:%M:%S',
	)
	
	plotly_utils.set_my_template_as_default()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)
	parser.add_argument('--threshold',
		metavar = 'mV',
		help = 'Threshold, in mili volt, for the amplitude to consider a pixel activation when calculating the hit correlation. The polarity does not matter, i.e. ±10 both have the same effect.',
		required = True,
		dest = 'threshold',
		type = float,
	)
	parser.add_argument('--minimum_coincidences',
		metavar = 'N',
		help = 'Minimum number of coincidences to consider when calculating the jitter. Default is %(default)s.',
		dest = 'minimum_number_of_coincidences',
		type = int,
		default = 100,
	)
	
	args = parser.parse_args()
	bureaucrat = RunBureaucrat(Path(args.directory))
	
	jitter_estimation(
		bureaucrat,
		amplitude_threshold_volts = -abs(args.threshold)/1000,
		minimum_number_of_coincidences = args.minimum_number_of_coincidences,
	)
