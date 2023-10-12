from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import subprocess
import logging

PATH_TO_caenCliRootWF = Path.home()/'code/Jordis_docker/eudaq/bin/caenCliRootWF'

def raw_to_root(bureaucrat:RunBureaucrat, force:bool=False):
	"""Converts all the raw data files into Root files using the `caenCliRootWF`.
	
	Arguments
	---------
	bureaucrat: RunBureaucrat
		A bureaucrat pointing to the run for which to convert the raw files.
	force: bool, default False
		If `False` and the task `raw_to_root` was already executed successfully
		for the run being handled by `bureaucrat`, nothing is done.
	"""
	bureaucrat.check_these_tasks_were_run_successfully('raw')
	
	if force==False and bureaucrat.was_task_run_successfully('raw_to_root'):
		return
	
	with bureaucrat.handle_task('raw_to_root') as employee:
		logging.info(f'Processing {bureaucrat.path_to_run_directory}...')
		path_to_raw_file = bureaucrat.path_to_directory_of_task('raw')/f'{bureaucrat.run_name}.raw'
		result = subprocess.run(
			[str(PATH_TO_caenCliRootWF), '-i', str(path_to_raw_file), '-o', str(employee.path_to_directory_of_my_task/path_to_raw_file.name.replace('.raw','.root'))],
			cwd = PATH_TO_caenCliRootWF.parent,
			stdout = subprocess.DEVNULL,
			stderr = subprocess.STDOUT,
		)
		try:
			result.check_returncode()
		except subprocess.CalledProcessError as e:
			if e.returncode == -6:
				# This happens always at the end, but the resulting file looks good...
				pass
			else:
				raise e
	logging.info(f'Raw file {path_to_raw_file} was successfully converted into a root file')

def raw_to_root_whole_batch(batch_bureaucrat:RunBureaucrat, force:bool=False):
	batch_bureaucrat.check_these_tasks_were_run_successfully('runs')
	
	for TB_run_bureaucrat in batch_bureaucrat.list_subruns_of_task('runs'):
		raw_to_root(TB_run_bureaucrat, force)

def raw_to_root_whole_TB_campaign(TB_campaign_bureaucrat:RunBureaucrat, force:bool=False):
	TB_campaign_bureaucrat.check_these_tasks_were_run_successfully('batches')
	
	for batch in TB_campaign_bureaucrat.list_subruns_of_task('batches'):
		raw_to_root_whole_batch(batch, force)

if __name__=='__main__':
	import argparse
	import sys
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.INFO,
		format = '%(asctime)s|%(levelname)s|%(message)s',
		datefmt = '%H:%M:%S',
	)
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)
	parser.add_argument(
		'--force',
		help = 'If this flag is passed, it will force the processing even if it was already done beforehand. Old data will be deleted.',
		required = False,
		dest = 'force',
		action = 'store_true'
	)
	
	args = parser.parse_args()
	bureaucrat = RunBureaucrat(Path(args.directory))
	
	if bureaucrat.was_task_run_successfully('batches'):
		raw_to_root_whole_TB_campaign(bureaucrat, force=args.force)
	elif bureaucrat.was_task_run_successfully('runs'):
		raw_to_root_whole_batch(bureaucrat, force=args.force)
	elif bureaucrat.was_task_run_successfully('raw'):
		raw_to_root(bureaucrat, force=args.force)
	else:
		raise RuntimeError(f'Dont know how to process run {repr(bureaucrat.run_name)} located in {bureaucrat.path_to_run_directory}')
