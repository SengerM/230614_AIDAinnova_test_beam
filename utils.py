import pandas
from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import numpy
import subprocess
import datetime
from contextlib import nullcontext
from progressreporting.TelegramProgressReporter import SafeTelegramReporter4Loops # https://github.com/SengerM/progressreporting
import logging

PATH_TO_WHEREVER_THE_DOCKER_DATA_IS_POINTING_TO = Path('/home/msenger/240217_DESY_test_beam')

PLOTS_LABELS = {
	'DUT_name_rowcol': 'DUT (i,j)',
	'Px': 'x (m)',
	'Py': 'y (m)',
	'cluster_size': 'Cluster size',
	'efficiency': 'Efficiency',
	'efficiency (%)': 'Efficiency (%)',
	'efficiency_error': 'Efficiency error',
	'efficiency_error (%)': 'Efficiency error (%)',
}

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

def guess_where_how_to_run(bureaucrat:RunBureaucrat, raw_level_f:callable, telegram_bot_reporter:SafeTelegramReporter4Loops=None):
	"""Given a `raw_level_f` function that is intended to operate on a
	`raw run`, i.e. a bureaucrat node containing one of the runs from
	the TB (e.g. TB_data_analysis/campaigns/subruns/230614_June/batches/subruns/batch_2_230V/EUDAQ_runs/subruns/run000937_230625134927)
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
		batch_bureaucrat.check_these_tasks_were_run_successfully('EUDAQ_runs') # To ensure that we are on a "batch node".
		
		runs_to_be_processed = batch_bureaucrat.list_subruns_of_task('EUDAQ_runs')
		
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
	elif bureaucrat.was_task_run_successfully('EUDAQ_runs'):
		run_on_batch_runs(bureaucrat, raw_level_f, telegram_bot_reporter=telegram_bot_reporter)
	elif bureaucrat.was_task_run_successfully('raw'):
		raw_level_f(bureaucrat)
	else:
		raise RuntimeError(f'Dont know how to process run {repr(bureaucrat.run_name)} located in {bureaucrat.path_to_run_directory}')

def which_kind_of_node(bureaucrat:RunBureaucrat):
	if bureaucrat.was_task_run_successfully('batches'):
		return 'campaign'
	elif bureaucrat.was_task_run_successfully('EUDAQ_runs'):
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
	temp_file = PATH_TO_WHEREVER_THE_DOCKER_DATA_IS_POINTING_TO/f'.{timestamp}_{numpy.random.rand()*1e9:.0f}.sh'
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
