"""All the code in this file assumes that there is a Docker container
running the 'Jordi`s eudaq container'."""

from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import logging
import warnings
import utils
import subprocess

def raw_to_root(bureaucrat:RunBureaucrat, container_id:str, force:bool=False, silent_root:bool=False):
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
		path_to_raw_file = utils.get_run_directory_within_corry_docker(bureaucrat)/'raw'/f'{bureaucrat.run_name}.raw'
		path_to_root_file = utils.get_run_directory_within_corry_docker(bureaucrat)/employee.task_name/f'{bureaucrat.run_name}.root'
		result = utils.run_commands_in_docker_container(
			f'/eudaq/eudaq/bin/caenCliRootWF -i {path_to_raw_file} -o {path_to_root_file}',
			container_id = container_id,
			stdout = subprocess.DEVNULL if silent_root == True else None,
			stderr = subprocess.STDOUT if silent_root == True else None,
		)
		result.check_returncode()
		logging.info(f'Successfully converted raw to root for {bureaucrat.pseudopath} ✅')

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
		type = Path,
	)
	parser.add_argument(
		'--force',
		help = 'If this flag is passed, it will force the processing even if it was already done beforehand. Old data will be deleted.',
		required = False,
		dest = 'force',
		action = 'store_true'
	)
	parser.add_argument('--container_id',
		metavar = 'id', 
		help = 'Id of the docker container running `Jordis eudaq docker`. Once the container is already running, you can get the id with `docker ps`.',
		required = True,
		dest = 'container_id',
		type = str,
	)
	
	args = parser.parse_args()
	raw_to_root(
		bureaucrat = RunBureaucrat(args.directory),
		force = args.force,
		container_id = args.container_id,
		silent_root = True,
	)
