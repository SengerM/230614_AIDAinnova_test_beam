from pathlib import Path
import numpy
import pandas
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import plotly.graph_objects as go
import plotly.express as px
import utils
import logging
import sqlite3

def do_quick_plots(bureaucrat:RunBureaucrat, max_events_to_plot=11111):
	bureaucrat.check_these_tasks_were_run_successfully(['parse_waveforms','batch_info'])
	
	logging.info(f'Reading data for run {bureaucrat.run_name}...')
	data = []
	CAENs_names = []
	for sqlite_file_path in sorted((bureaucrat.path_to_directory_of_task('parse_waveforms')/'parsed_data').iterdir(), key=lambda x: x.stat().st_size):
		logging.info(f'Loading {sqlite_file_path.name}')
		n_run = int(sqlite_file_path.name.split('_')[0].replace('run',''))
		df = load_whole_dataframe(sqlite_file_path)
		df['n_run'] = n_run
		df.set_index('n_run',append=True,inplace=True)
		data.append(df)
		
		df = pandas.read_pickle(bureaucrat.path_to_directory_of_task('parse_waveforms')/'CAENs_names'/sqlite_file_path.name.replace('.sqlite','_CAENs_names.pickle'))
		df = df.to_frame()
		df['n_run'] = n_run
		df.set_index('n_run',append=True,inplace=True)
		CAENs_names.append(df)
		
		number_of_events_already_loaded = sum([len(_.index.get_level_values('n_event').drop_duplicates()) for _ in data])
		if number_of_events_already_loaded > max_events_to_plot:
			logging.info(f'Already loaded {number_of_events_already_loaded} events, not loading more SQLite files.')
			break
	data = pandas.concat(data)
	CAENs_names = pandas.concat(CAENs_names)
	CAENs_names['CAEN_name'] = CAENs_names['CAEN_name'].apply(lambda x: x.replace('CAEN_',''))
	signals_connections = utils.load_setup_configuration_info(bureaucrat)
	logging.info(f'Data for run {bureaucrat.run_name} was read.')
	
	signals_connections['rowcol'] = signals_connections[['row','col']].apply(lambda x: f"{x['row']},{x['col']}", axis=1)
	
	data = data.reset_index(drop=False).merge(
		signals_connections.set_index(['n_CAEN','CAEN_n_channel'])[['DUT_name','rowcol','row','col','CAEN_channel_name','CAEN_name','CAEN_trigger_group_n']],
		on = ['n_CAEN','CAEN_n_channel'],
	)
	
	data.set_index(['n_run','n_event','n_CAEN','CAEN_trigger_group_n'], inplace=True)
	data.sort_index(inplace=True)
	
	trigger_time = data.query('CAEN_channel_name in ["trigger_group_0","trigger_group_1"]')['t_50 (s)']
	trigger_time.name = 'Trigger time (s)'
	data['t_50 from trigger (s)'] = data['t_50 (s)'] - trigger_time
	
	for col in {'Amplitude (V)','Collected charge (V s)'}:
		data[f'-{col}'] = -1*data[col]
	
	absolute_n_event = data.reset_index(drop=False)[['n_run','n_event']].drop_duplicates(ignore_index=True)
	absolute_n_event.index.rename('n_event_absolute', inplace=True)
	absolute_n_event.reset_index(drop=False, inplace=True)
	absolute_n_event.set_index(['n_run','n_event'], inplace=True)
	data['n_event_absolute'] = absolute_n_event
	
	logging.info(f'Applying cuts for run {bureaucrat.run_name}...')
	
	data = data.query('`t_50 (s)`<199e-9') # This is because of a defect of the UZH CAEN digitizer, it tends to produce a peak close to the end of the time window, this is not happening with Ljubljana CAEN.
	data = data.query('DUT_name != "trigger"')
	
	data.reset_index(drop=False, inplace=True)
	data.set_index(['n_event_absolute','DUT_name','rowcol'], drop=True, inplace=True)
	data = data.unstack(['DUT_name','rowcol'])
	# ~ data = data.loc[(data[('-Amplitude (V)','time_reference_1','1,0')]>.02)]
	data = data.stack(['DUT_name','rowcol'])
	
	logging.info(f'Producing plots for run {bureaucrat.run_name}...')
	data.reset_index(inplace=True, drop=False)
	data = data.sort_values(by=['DUT_name','rowcol'])
	with bureaucrat.handle_task('some_plots') as employee:
		for col in ['t_50 from trigger (s)','-Amplitude (V)','-Collected charge (V s)','Noise (V)','t_50 (s)','SNR']:
			logging.info(f'Plotting {repr(col)}...')
			fig = px.ecdf(
				data_frame = data,
				x = col,
				color = 'rowcol',
				facet_col = 'DUT_name',
				hover_data = ['n_run','n_event','CAEN_name'],
			)
			fig.update_layout(
				width = 666*len(data['DUT_name'].drop_duplicates()),
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{col}_ECDF.html',
				include_plotlyjs = 'cdn',
			)
			fig.write_image(employee.path_to_directory_of_my_task/f'{col}_ECFD.png')
	
	fig = px.density_heatmap(
		data_frame = data.query('DUT_name not in ["trigger"]'),
		x = 't_50 from trigger (s)',
		y = '-Amplitude (V)',
		facet_col = 'DUT_name',
	)
	fig.update_layout(
		width = 666*len(data['DUT_name'].drop_duplicates()),
	)
	fig.write_html(
		employee.path_to_directory_of_my_task/f't_50_from_trigger_vs_-ampliutde.html',
		include_plotlyjs = 'cdn',
	)
	fig.write_image(employee.path_to_directory_of_my_task/f't_50_from_trigger_vs_-ampliutde.png')
	
def do_correlation_plots(bureaucrat:RunBureaucrat, max_events_to_plot=None):
	bureaucrat.check_these_tasks_were_run_successfully(['parse_waveforms','batch_info'])
	
	INDEX_COLS = ['n_event','n_CAEN','CAEN_n_channel']
	variables_for_which_to_plot_correlation = ['Amplitude (V)']
	
	logging.info(f'Reading data for run {bureaucrat.run_name}...')
	data = []
	CAENs_names = []
	columns_to_load = INDEX_COLS + variables_for_which_to_plot_correlation
	for sqlite_file_path in sorted((bureaucrat.path_to_directory_of_task('parse_waveforms')/'parsed_data').iterdir(), key=lambda x: x.stat().st_size):
		number_of_events_already_loaded = sum([len(_.index.get_level_values('n_event').drop_duplicates()) for _ in data])
		logging.info(f'Loading {sqlite_file_path.name} ({number_of_events_already_loaded} events loaded so far)')
		n_run = int(sqlite_file_path.name.split('_')[0].replace('run',''))
		sqlite_connection = sqlite3.connect(sqlite_file_path)
		df = pandas.read_sql(f'SELECT {",".join([f"`{_}`" for _ in columns_to_load])} FROM dataframe_table', sqlite_connection).set_index(INDEX_COLS)
		df['n_run'] = n_run
		df.set_index('n_run',append=True,inplace=True)
		data.append(df)
		
		df = pandas.read_pickle(bureaucrat.path_to_directory_of_task('parse_waveforms')/'CAENs_names'/sqlite_file_path.name.replace('.sqlite','_CAENs_names.pickle'))
		df = df.to_frame()
		df['n_run'] = n_run
		df.set_index('n_run',append=True,inplace=True)
		CAENs_names.append(df)
		
		number_of_events_already_loaded = sum([len(_.index.get_level_values('n_event').drop_duplicates()) for _ in data])
		if max_events_to_plot is not None and number_of_events_already_loaded > max_events_to_plot:
			logging.info(f'Already loaded {number_of_events_already_loaded} events, not loading more SQLite files.')
			break
	data = pandas.concat(data)
	CAENs_names = pandas.concat(CAENs_names)
	CAENs_names['CAEN_name'] = CAENs_names['CAEN_name'].apply(lambda x: x.replace('CAEN_',''))
	signals_connections = utils.load_setup_configuration_info(bureaucrat)
	logging.info(f'Data for run {bureaucrat.run_name} was read.')
	
	signals_connections['rowcol'] = signals_connections[['row','col']].apply(lambda x: f"{x['row']},{x['col']}", axis=1)
	
	data = data.reset_index(drop=False).merge(
		signals_connections.set_index(['n_CAEN','CAEN_n_channel'])[['DUT_name','rowcol','row','col','CAEN_channel_name','CAEN_name','CAEN_trigger_group_n']],
		on = ['n_CAEN','CAEN_n_channel'],
	)
	
	data.set_index(['n_run','n_event','n_CAEN','CAEN_trigger_group_n'], inplace=True)
	data.sort_index(inplace=True)
	
	absolute_n_event = data.reset_index(drop=False)[['n_run','n_event']].drop_duplicates(ignore_index=True)
	absolute_n_event.index.rename('n_event_absolute', inplace=True)
	absolute_n_event.reset_index(drop=False, inplace=True)
	absolute_n_event.set_index(['n_run','n_event'], inplace=True)
	data['n_event_absolute'] = absolute_n_event
	
	
	logging.info(f'Calculating correlations...')
	correlations = data.reset_index(drop=False).query('`Amplitude (V)`<0')[variables_for_which_to_plot_correlation+['DUT_name','row','col','n_event_absolute']]
	correlations = correlations.query('DUT_name != "trigger"')
	correlations.set_index(['DUT_name','row','col','n_event_absolute'], inplace=True)
	correlations.sort_index(inplace=True)
	correlations = correlations.unstack(['DUT_name','row','col'])
	correlations = correlations.corr()
	
	with bureaucrat.handle_task('correlation_plots') as employee:
		logging.info('Plotting correlations...')
		for col in correlations.columns.get_level_values(0):
			df = correlations.loc[col,col]
			df.columns = [f'{DUT_name} ({row},{col})' for DUT_name,row,col in df.columns]
			df.index = [f'{DUT_name} ({row},{col})' for DUT_name,row,col in df.index]
			fig = px.imshow(
				df, 
				aspect = "auto",
				title = f'{col} correlation<br><sup>{bureaucrat.run_name}</sup>',
				labels = dict(color=f'ρ({col})'),
			)
			fig.update_coloraxes(colorbar_title_side='right')
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{col}_correlation.html',
				include_plotlyjs = 'cdn',
			)

if __name__=='__main__':
	import argparse
	from grafica.plotly_utils.utils import set_my_template_as_default
	import sys
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.DEBUG,
		format = '%(asctime)s|%(levelname)s|%(funcName)s|%(message)s',
		datefmt = '%Y-%m-%d %H:%M:%S',
	)
	
	set_my_template_as_default()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)
	
	args = parser.parse_args()
	bureaucrat = RunBureaucrat(Path(args.directory))
	
	do_correlation_plots(bureaucrat)
