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
	if force==False and bureaucrat.was_task_run_successfully('raw_to_root'):
		return
	
	with bureaucrat.handle_task('raw_to_root') as employee:
		paht_to_directory_in_which_to_save_the_root_files = employee.path_to_directory_of_my_task/'root_files'
		paht_to_directory_in_which_to_save_the_root_files.mkdir()
		for path_to_raw_file in (employee.path_to_run_directory/'raw').iterdir():
			logging.debug(f'About to process {path_to_raw_file}')
			subprocess.run(
				[str(PATH_TO_caenCliRootWF), '-i', str(path_to_raw_file), '-o', str(paht_to_directory_in_which_to_save_the_root_files/path_to_raw_file.name.replace('.raw','.root'))],
				cwd = PATH_TO_caenCliRootWF.parent,
			)
	
if __name__=='__main__':
	import argparse
	
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
	
	raw_to_root(bureaucrat, force=args.force)
