"""All the code in this file assumes that there is a Docker container
running the 'Jordi`s eudaq container'."""

from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
from pathlib import Path
import logging
import warnings
import subprocess
import utils

def raw_to_root(EUDAQ_run_dn:DatanodeHandler, container_id:str, force:bool=False, silent_root:bool=False):
	"""Converts all the raw data files into Root files using the `caenCliRootWF`.
	
	Arguments
	---------
	EUDAQ_run_dn: DatanodeHandler
		A `DatanodeHandler` pointing to the run for which to convert the raw files.
	force: bool, default False
		If `False` and the task `raw_to_root` was already executed successfully
		for the run being handled by `EUDAQ_run_dn`, nothing is done.
	"""
	
	if force==False and EUDAQ_run_dn.was_task_run_successfully('raw_to_root'):
		return
	
	with EUDAQ_run_dn.handle_task('raw_to_root', check_datanode_class='EUDAQ_run', check_required_tasks='raw') as employee:
		logging.info(f'Processing {EUDAQ_run_dn.pseudopath}...')
		path_to_raw_file = utils.get_datanode_directory_within_corry_docker(EUDAQ_run_dn)/'raw'/f'{EUDAQ_run_dn.datanode_name}.raw'
		path_to_root_file = utils.get_datanode_directory_within_corry_docker(EUDAQ_run_dn)/employee.task_name/f'{EUDAQ_run_dn.datanode_name}.root'
		result = utils.run_commands_in_docker_container(
			f'/eudaq/eudaq/bin/caenCliRootWF -i {path_to_raw_file} -o {path_to_root_file}',
			container_id = container_id,
			stdout = subprocess.DEVNULL if silent_root == True else None,
			stderr = subprocess.STDOUT if silent_root == True else None,
		)
		result.check_returncode()
		logging.info(f'Successfully converted raw to root for {EUDAQ_run_dn.pseudopath} ✅')

if __name__=='__main__':
	import argparse
	import sys
	import my_telegram_bots # Secret tokens from my bots
	from progressreporting.TelegramProgressReporter import SafeTelegramReporter4Loops # https://github.com/SengerM/progressreporting
	from utils_run_level import execute_EUDAQ_run_task_on_all_runs_within_batch
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.INFO,
		format = '%(asctime)s|%(levelname)s|%(message)s',
		datefmt = '%H:%M:%S',
	)
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--datanode',
		metavar = 'path', 
		help = 'Path to a datanode',
		required = True,
		dest = 'datanode',
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
	
	dn = DatanodeHandler(args.datanode)
	execute_EUDAQ_run_task_on_all_runs_within_batch(
		TB_batch_dn = dn,
		func = raw_to_root,
		args = {_.datanode_name:dict(container_id=args.container_id, force=args.force, silent_root=False) for _ in dn.list_subdatanodes_of_task('EUDAQ_runs')},
		telegram_bot_reporter = SafeTelegramReporter4Loops(
			bot_token = my_telegram_bots.robobot.token,
			chat_id = my_telegram_bots.chat_ids['Robobot TCT setup'],
		),
	)
