import pandas
from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat

def save_dataframe(df, name:str, location:Path):
	for extension,method in {'pickle':df.to_pickle,'csv':df.to_csv}.items():
		method(location/f'{name}.{extension}')

def setup_batch_info(bureaucrat:RunBureaucrat):
	"""Add some batch-wise information needed for the analysis, like
	for example a link to the setup connection spreadsheet."""
	PATH_TO_SETUP_CONNECTIONS_FILES = Path.home()/'June_test_beam_data/AIDAInnova_June/setup_connections'
	
	with bureaucrat.handle_task('batch_info') as employee:
		with open(bureaucrat.path_to_run_directory/'n_batch', 'r') as ifile:
			n_batch = int(ifile.readline())
		path_to_setup_connection_ods = PATH_TO_SETUP_CONNECTIONS_FILES/f'Batch{n_batch}.ods'
		
		have_to_navigate_backwards = [_ for i,_ in enumerate(employee.path_to_directory_of_my_task.parts) if _!=path_to_setup_connection_ods.parts[i]]
		have_to_navigate_upwards_afterward = [_ for i,_ in enumerate(path_to_setup_connection_ods.parts) if _!=employee.path_to_directory_of_my_task.parts[i]]
		
		relative_path_to_setup_connection_ods = Path('/'.join(['..' for i in have_to_navigate_backwards]))/Path('/'.join(have_to_navigate_upwards_afterward))
		
		(employee.path_to_directory_of_my_task/'setup_connections.ods').symlink_to(relative_path_to_setup_connection_ods)
