from utils import setup_batch_info
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import logging

if __name__=='__main__':
	import argparse
	import sys
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.DEBUG,
		format = '%(asctime)s|%(levelname)s|%(funcName)s|%(message)s',
		datefmt = '%Y-%m-%d %H:%M:%S',
	)
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to a run.',
		required = True,
		dest = 'directory',
		type = str,
	)
	args = parser.parse_args()
	
	bureaucrat = RunBureaucrat(Path(args.directory))
	if bureaucrat.was_task_run_successfully('runs'):
		setup_batch_info(bureaucrat)
	elif bureaucrat.was_task_run_successfully('batches'):
		for batch in bureaucrat.list_subruns_of_task('batches'):
			setup_batch_info(batch)
	else:
		raise RuntimeError(f'Dont know how to process {bureaucrat.pseudopath} located in {bureaucrat.path_to_run_directory}')
