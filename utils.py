import pandas
from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import numpy
import subprocess
import datetime
from contextlib import nullcontext
from progressreporting.TelegramProgressReporter import SafeTelegramReporter4Loops # https://github.com/SengerM/progressreporting
import logging

def save_dataframe(df, name:str, location:Path):
	for extension,method in {'pickle':df.to_pickle,'csv':df.to_csv}.items():
		method(location/f'{name}.{extension}')

def which_test_beam_campaign(bureaucrat:RunBureaucrat):
	"""Returns the name of the test beam campaign, one of '230614_June' or
	'230830_August'."""
	if '230614_June' in str(bureaucrat.path_to_run_directory):
		return '230614_June'
	elif '230830_August' in str(bureaucrat.path_to_run_directory):
		return '230830_August'
	else:
		raise RuntimeError(f'Cannot determine test beam campaign of run {bureaucrat.run_name}')

def setup_batch_info(bureaucrat:RunBureaucrat):
	"""Add some batch-wise information needed for the analysis, like
	for example a link to the setup connection spreadsheet."""
	def setup_batch_info_June_test_beam(bureaucrat:RunBureaucrat):
		bureaucrat.check_these_tasks_were_run_successfully('runs') # So we are sure this is pointing to a batch
		
		n_batch = int(bureaucrat.run_name.split('_')[1])
		if n_batch in {2,3,4}:
			path_to_setup_connection_ods = Path('/media/msenger/230829_gray/AIDAinnova_test_beams/raw_data/230614_June/AIDAInnova_June/setup_connections')/f'Batch{n_batch}.ods'
		elif n_batch in {5,6}:
			path_to_setup_connection_ods = Path('/media/msenger/230829_gray/AIDAinnova_test_beams/raw_data/230614_June/CMS-ETL_June/setup_connections')/f'setup_connections_Batch{n_batch}.ods'
		else:
			raise RuntimeError(f'Cannot determine batch name appropriately!')
		
		with bureaucrat.handle_task('batch_info') as employee:
			for sheet_name in {'planes','signals'}:
				df = pandas.read_excel(
					path_to_setup_connection_ods,
					sheet_name = sheet_name,
				).set_index('plane_number')
				save_dataframe(
					df,
					name = sheet_name,
					location = employee.path_to_directory_of_my_task,
				)
	
	def setup_batch_info_August_test_beam(bureaucrat:RunBureaucrat):
		bureaucrat.check_these_tasks_were_run_successfully('runs') # So we are sure this is pointing to a batch
		
		with bureaucrat.handle_task('batch_info') as employee:
			n_batch = int(bureaucrat.run_name.split('_')[1])
			planes_definition = pandas.read_csv(
				'https://docs.google.com/spreadsheets/d/e/2PACX-1vTuRXCnGCPu8nuTrrh_6M_QaBYwVQZfmLX7cr6OlM-ucf9yx3KIbBN4XBQxc0fTp-O26Y2QIOCkgP98/pub?gid=0&single=true&output=csv',
				dtype = dict(
					batch_number = int,
					plane_number = int,
					DUT_name = str,
					orientation = str,
					high_voltage_source = str,
					low_voltage_source = str,
				),
				index_col = ['batch_number','plane_number'],
			)
			pixels_definition = pandas.read_csv(
				'https://docs.google.com/spreadsheets/d/e/2PACX-1vTuRXCnGCPu8nuTrrh_6M_QaBYwVQZfmLX7cr6OlM-ucf9yx3KIbBN4XBQxc0fTp-O26Y2QIOCkgP98/pub?gid=1673457618&single=true&output=csv',
				dtype = dict(
					batch_number = int,
					plane_number = int,
					chubut_channel_number = int,
					digitizer_name = str,
					digitizer_channel_name = str,
					row = int,
					col = int,
				),
				index_col = ['batch_number','plane_number'],
			)
			for name,df in {'planes_definition':planes_definition, 'pixels_definition':pixels_definition}.items():
				save_dataframe(df.query(f'batch_number=={n_batch}'), name, employee.path_to_directory_of_my_task)
	
	if which_test_beam_campaign(bureaucrat) == '230614_June':
		setup_batch_info_June_test_beam(bureaucrat)
	elif which_test_beam_campaign(bureaucrat) == '230830_August':
		setup_batch_info_August_test_beam(bureaucrat)
	else:
		raise RuntimeError(f'Cannot determine which test beam campaign {bureaucrat.pseudopath} belongs to...')
	logging.info(f'Setup info was set for {bureaucrat.pseudopath} ✅')

CAENs_CHANNELS_MAPPING_TO_INTEGERS = pandas.DataFrame(
	# This codification into integers comes from the producer, see in line 208 of `CAENDT5742Producer.py`. The reason is that EUDAQ can only handle integers tags, or something like this.
	{
		'CAEN_n_channel': list(range(18)),
		'CAEN_channel_name': [f'CH{i}' if i<16 else f'trigger_group_{i-16}' for i in range(18)]
	}
)

def load_setup_configuration_info(batch:RunBureaucrat)->pandas.DataFrame:
	bureaucrat = batch
	bureaucrat.check_these_tasks_were_run_successfully(['runs','batch_info'])
	if not all([b.was_task_run_successfully('parse_waveforms') for b in bureaucrat.list_subruns_of_task('runs')]):
		raise RuntimeError(f'To load the setup configuration it is needed that all of the runs of the batch have had the `parse_waveforms` task performed on them, but does not seem to be the case')
	
	if which_test_beam_campaign(bureaucrat) == '230614_June':
		planes = pandas.read_pickle(bureaucrat.path_to_directory_of_task('batch_info')/'planes.pickle')
		signals_connections = pandas.read_pickle(bureaucrat.path_to_directory_of_task('batch_info')/'signals.pickle')
	elif which_test_beam_campaign(bureaucrat) == '230830_August':
		planes = pandas.read_pickle(bureaucrat.path_to_directory_of_task('batch_info')/'planes_definition.pickle')
		signals_connections = pandas.read_pickle(bureaucrat.path_to_directory_of_task('batch_info')/'pixels_definition.pickle')
		for df in [planes,signals_connections]:
			df.rename(
				columns = {
					'digitizer_name': 'CAEN_name',
					'digitizer_channel_name': 'CAEN_channel_name',
					'chubut_channel_number': 'chubut_channel',
				},
				inplace = True,
			)
	else:
		raise RuntimeError(f'Cannot read setup information for run {bureaucrat.run_name}')
	
	CAENs_names = []
	for run in bureaucrat.list_subruns_of_task('runs'):
		n_run = int(run.run_name.split('_')[0].replace('run',''))
		df = pandas.read_pickle(run.path_to_directory_of_task('parse_waveforms')/f'{run.run_name}_CAENs_names.pickle')
		df = df.to_frame()
		df['n_run'] = n_run
		df.set_index('n_run',append=True,inplace=True)
		CAENs_names.append(df)
	CAENs_names = pandas.concat(CAENs_names)
	CAENs_names['CAEN_name'] = CAENs_names['CAEN_name'].apply(lambda x: x.replace('CAEN_',''))
	
	# Here we assume that the CAENs were not changed within a batch, which is reasonable.
	_ = CAENs_names.reset_index('n_CAEN',drop=False).set_index('CAEN_name',append=True).reset_index('n_run',drop=True)
	_ = _[~_.index.duplicated(keep='first')]
	signals_connections = signals_connections.reset_index(drop=False).merge(
		_,
		on = 'CAEN_name',
	)
	
	signals_connections = signals_connections.merge(
		CAENs_CHANNELS_MAPPING_TO_INTEGERS.set_index('CAEN_channel_name')['CAEN_n_channel'],
		on = 'CAEN_channel_name',
	)
	
	signals_connections = signals_connections.merge(
		planes['DUT_name'],
		on = 'plane_number',
	)
	
	signals_connections['CAEN_trigger_group_n'] = signals_connections['CAEN_n_channel'].apply(lambda x: 0 if x in {0,1,2,3,4,5,6,7,16} else 1 if x in {8,9,10,11,12,13,14,15,17} else -1)
	
	signals_connections['rowcol'] = signals_connections[['row','col']].apply(lambda x: f'{x["row"]}{x["col"]}', axis=1)
	signals_connections['DUT_name_rowcol'] = signals_connections[['DUT_name','row','col']].apply(lambda x: f'{x["DUT_name"]} ({x["row"]},{x["col"]})', axis=1)
	
	return signals_connections

def select_by_multiindex(df:pandas.DataFrame, idx:pandas.MultiIndex)->pandas.DataFrame:
	"""Given a DataFrame and a MultiIndex object, selects the entries
	from the data frame matching the multi index. Example:
	DataFrame:
	```
	       d
	a b c   
	1 9 4  1
	2 8 2  4
	3 7 5  2
	4 6 4  5
	5 5 6  3
	6 4 5  6
	7 3 7  4
	8 2 6  7
	9 1 8  5
	```
	MultiIndex:
	```
	MultiIndex([(1, 9),
            (2, 8),
            (3, 7),
            (4, 6),
            (9, 1)],
           names=['a', 'b'])
	```
	Output:
	```
	       d
	a b c   
	1 9 4  1
	2 8 2  4
	3 7 5  2
	4 6 4  5
	9 1 8  5

	```
	"""
	if not set(idx.names) <= set(df.index.names):
		raise ValueError('Names in `idx` not present in `df.index`')
	if not isinstance(df, pandas.DataFrame) or not isinstance(idx, pandas.MultiIndex):
		raise TypeError('`df` or `idx` are of the wrong type.')
	lvl = df.index.names.difference(idx.names)
	return df[df.index.droplevel(lvl).isin(idx)]

def project_track_in_z(A:numpy.array, B:numpy.array, z:float):
	"""Given two points in a (straight) track, A and B, finds the projection
	at some given z.
	
	Arguments:
	A, B: numpy.array
		Two points along a track. Can be a collection of points from multiple
		tracks, in this case the shape has to be (3,whatever...).
	z: float
		Value of z on which to project the tracks.
	"""
	def dot(a,b):
		return numpy.sum(a*b, axis=0)
	if A.shape != B.shape or A.shape[0] != 3:
		raise ValueError('Either `A` or `B`, or both, is invalid.')
	track_direction = (A-B)/(numpy.linalg.norm(A-B, axis=0))
	return A + track_direction*(z-A[2])/dot(track_direction, numpy.tile(numpy.array([0,0,1]), (int(A.size/3),1)).T)

def guess_where_how_to_run(bureaucrat:RunBureaucrat, raw_level_f:callable, telegram_bot_reporter:SafeTelegramReporter4Loops=None):
	"""Given a `raw_level_f` function that is intended to operate on a
	`raw run`, i.e. a bureaucrat node containing one of the runs from
	the TB (e.g. TB_data_analysis/campaigns/subruns/230614_June/batches/subruns/batch_2_230V/runs/subruns/run000937_230625134927)
	this function will automatically iterate it over all the runs if `bureaucrat`
	points to a batch (e.g. TB_data_analysis/campaigns/subruns/230614_June/batches/subruns/batch_2_230V), 
	or will automatically iterate it over all batches and runs if `bureaucrat` 
	points to a campaign (e.g. TB_data_analysis/campaigns/subruns/230614_June)
	
	Arguments
	---------
	bureaucrat: RunBureaucrat
		The bureaucrat of the top level on which to apply the `raw_level_f`.
	raw_level_f: callable
		The function to be executed recursively. The signature has to be
		`raw_level_f = f(bureaucrat:RunBureaucrat)`.
	telegram_bot_reporter SafeTelegramReporter4Loops, default None
		An optional bot to report the progress and any issues.
	"""
	def run_on_batch_runs(batch_bureaucrat:RunBureaucrat, raw_level_f:callable, telegram_bot_reporter:SafeTelegramReporter4Loops=None):
		batch_bureaucrat.check_these_tasks_were_run_successfully('runs') # To ensure that we are on a "batch node".
		
		runs_to_be_processed = batch_bureaucrat.list_subruns_of_task('runs')
		
		with telegram_bot_reporter.report_loop(
			total_loop_iterations = len(runs_to_be_processed),
			loop_name = f'{raw_level_f.__name__} on runs {bureaucrat.pseudopath}',
		) if telegram_bot_reporter is not None else nullcontext() as reporter:
			for TB_run_bureaucrat in runs_to_be_processed:
				raw_level_f(TB_run_bureaucrat)
				reporter.update(1) if telegram_bot_reporter is not None else None

	def run_on_TB_campaign_batches(TB_campaign_bureaucrat:RunBureaucrat, raw_level_f:callable, telegram_bot_reporter:SafeTelegramReporter4Loops=None):
		TB_campaign_bureaucrat.check_these_tasks_were_run_successfully('batches') # To ensure we are on a "TB campaign node".
		
		batches_to_be_processed = TB_campaign_bureaucrat.list_subruns_of_task('batches')
		
		with telegram_bot_reporter.report_loop(
			total_loop_iterations = len(batches_to_be_processed),
			loop_name = f'{raw_level_f.__name__} on batches of {bureaucrat.pseudopath}',
		) if telegram_bot_reporter is not None else nullcontext() as reporter:
			for batch in batches_to_be_processed:
				run_on_batch_runs(
					batch_bureaucrat = batch,
					raw_level_f = raw_level_f,
					telegram_bot_reporter = telegram_bot_reporter.create_subloop_reporter() if telegram_bot_reporter is not None else None,
				)
				reporter.update(1) if telegram_bot_reporter is not None else None
	
	if bureaucrat.was_task_run_successfully('batches'):
		run_on_TB_campaign_batches(bureaucrat, raw_level_f, telegram_bot_reporter=telegram_bot_reporter)
	elif bureaucrat.was_task_run_successfully('runs'):
		run_on_batch_runs(bureaucrat, raw_level_f, telegram_bot_reporter=telegram_bot_reporter)
	elif bureaucrat.was_task_run_successfully('raw'):
		raw_level_f(bureaucrat)
	else:
		raise RuntimeError(f'Dont know how to process run {repr(bureaucrat.run_name)} located in {bureaucrat.path_to_run_directory}')

def which_kind_of_node(bureaucrat:RunBureaucrat):
	if bureaucrat.was_task_run_successfully('batches'):
		return 'campaign'
	elif bureaucrat.was_task_run_successfully('runs'):
		return 'batch'
	elif bureaucrat.was_task_run_successfully('raw'):
		return 'run'
	elif bureaucrat.was_task_run_successfully('this_is_a_TI-LGAD_analysis'):
		return 'TI-LGAD analysis'
	else:
		return None

def get_run_directory_within_corry_docker(bureaucrat:RunBureaucrat):
	"""Get the absolute path of the run directory within the corry docker
	container."""
	if bureaucrat.exists() == False:
		raise RuntimeError(f'Run pointed to by `bureaucrat` does not exist: {bureaucrat.path_to_run_directory}')
	TB_data_analysis_bureaucrat = bureaucrat
	while True:
		if TB_data_analysis_bureaucrat.parent is None:
			break
		else:
			TB_data_analysis_bureaucrat = TB_data_analysis_bureaucrat.parent
	
	return Path('/data')/bureaucrat.path_to_run_directory.relative_to(TB_data_analysis_bureaucrat.path_to_run_directory.parent)

def run_commands_in_docker_container(command, container_id:str, stdout=None, stderr=None):
	"""Runs one or more commands inside a docker container.
	
	Arguments
	---------
	command: str or list of str
		A string with the command, or a list of strings with multiple
		commands to be executed sequentially.
	"""
	timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
	temp_file = Path(f'/media/msenger/230829_gray/AIDAinnova_test_beams/.{timestamp}_{numpy.random.rand()*1e9:.0f}.sh')
	if not isinstance(command, (str, list)):
		raise TypeError(f'`command` must be a list or a string, received object of type {type(command)}')
	if isinstance(command, str):
		command = [command]
	if any([not isinstance(_, str) for _ in command]):
		raise TypeError(f'`command` must be a list of strings, but at least one element is not.')
	try:
		with open(temp_file, 'w') as ofile:
			print('#!/bin/bash', file=ofile)
			for c in command:
				print(c, file=ofile)
		subprocess.run(['chmod','+x',str(temp_file)])
		result = subprocess.run(
			['docker','exec','-it',container_id,f'/data/{temp_file.name}'],
			stdout = stdout,
			stderr = stderr,
		)
	except:
		raise
	finally:
		temp_file.unlink()
	return result

if __name__=='__main__':
	config = load_setup_configuration_info(RunBureaucrat(Path('/media/msenger/230829_gray/AIDAinnova_test_beams/TB_data_analysis/campaigns/subruns/230614_June/batches/subruns/batch_4')))
	print(config)
