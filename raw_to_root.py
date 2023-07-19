from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import subprocess

PATH_TO_caenCliRootWF = Path.home()/'code/Jordis_docker/eudaq/bin/caenCliRootWF'

def raw_to_root(bureaucrat:RunBureaucrat, force:bool=False):
	if force==False and bureaucrat.was_task_run_successfully('raw_to_root'):
		return
	
	with bureaucrat.handle_task('raw_to_root') as employee:
		paht_to_directory_in_which_to_save_the_root_files = employee.path_to_directory_of_my_task/'root_files'
		paht_to_directory_in_which_to_save_the_root_files.mkdir()
		for path_to_raw_file in (employee.path_to_run_directory/'raw').iterdir():
			subprocess.run([str(PATH_TO_caenCliRootWF), '-i', str(path_to_raw_file), '-o', str(paht_to_directory_in_which_to_save_the_root_files/path_to_raw_file.name.replace('.raw','.root'))])
	
if __name__=='__main__':
	raw_to_root(RunBureaucrat(Path('/media/msenger/230302_red/June_test_beam/analysis/testing_deleteme')))
