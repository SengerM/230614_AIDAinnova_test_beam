"""All the code in this file assumes that there is a Docker container
running the 'Jordi`s corry container'."""

from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import utils
import logging
import subprocess
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper # https://github.com/SengerM/huge_dataframe
import pandas

def run_command_in_corry_docker_container(command:str):
	CONTAINER_ID = '612536858d53' # To obtain this, run `docker ps` from outside the container.
	return subprocess.run(['docker','exec','-it',CONTAINER_ID,command])

def replace_arguments_in_file_template(file_template:Path, output_file:Path, arguments:dict):
	"""Given a file template with arguments denoted by 'ASD(argument_name)DSA',
	replace the arguments and create a new file with the values written
	to it."""
	file_template = Path(file_template)
	if file_template.suffix != '.template':
		raise RuntimeError(f'Cannot process a file that does not have the extension ".template", received {file_template.parts[-1]}. ')
	with open(file_template, 'r') as ifile:
		with open(output_file, 'w') as ofile:
			for line in ifile:
				for arg_name,arg_val in arguments.items():
					line = line.replace(f'ASD({arg_name})DSA',arg_val)
				print(line, file=ofile, end='')

def get_run_directory_within_corry_docker(bureaucrat:RunBureaucrat):
	return Path(f'/data/{utils.which_test_beam_campaign(bureaucrat)}/analysis/{bureaucrat.run_name}')

def corry_mask_noisy_pixels(bureaucrat:RunBureaucrat, force:bool=False):
	TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/01_mask_noisy_pixels')
	
	TASK_NAME = 'corry_mask_noisy_pixels'
	
	if force==False and bureaucrat.was_task_run_successfully(TASK_NAME):
		logging.info(f'Found an already successfull execution of {TASK_NAME} within {bureaucrat.run_name}, will not do anything.')
		return
	
	with bureaucrat.handle_task(TASK_NAME) as employee:
		for path_to_raw_file in (bureaucrat.path_to_run_directory/'raw').iterdir():
			logging.info(f'Creating corry config files for raw file {path_to_raw_file.name}...')
			
			raw_file_name_without_extension = path_to_raw_file.parts[-1].replace(".raw","")
			path_to_raw_file_within_docker_container = get_run_directory_within_corry_docker(bureaucrat)/f'raw/{path_to_raw_file.parts[-1]}'
			output_directory_within_corry_docker = get_run_directory_within_corry_docker(bureaucrat)/f'{employee.task_name}/{raw_file_name_without_extension}/corry_output'
			path_to_geometry_file_within_corry_docker_relative = '../../corry_geometry_for_this_batch.geo'
			
			path_to_where_to_save_the_config_files = employee.path_to_directory_of_my_task/raw_file_name_without_extension
			
			arguments_for_config_files = {
				'01_mask_noisy_pixels.conf': dict(
					OUTPUT_DIRECTORY = f'"{output_directory_within_corry_docker}"',
					GEOMETRY_FILE = f'"{path_to_geometry_file_within_corry_docker_relative}"',
					PATH_TO_RAW_FILE = f'"{path_to_raw_file_within_docker_container}"',
				),
				'tell_corry_docker_to_run_this.sh': dict(
					WORKING_DIRECTORY = str(output_directory_within_corry_docker.parent),
				),
			}
			
			path_to_where_to_save_the_config_files.mkdir()
			for fname in arguments_for_config_files:
				replace_arguments_in_file_template(
					file_template = TEMPLATE_FILES_DIRECTORY/f'{fname}.template',
					output_file = path_to_where_to_save_the_config_files/fname,
					arguments = arguments_for_config_files[fname],
				)
			
			# Make the scripts executable for docker:
			for fname in path_to_where_to_save_the_config_files.iterdir():
				if fname.suffix == '.sh':
					subprocess.run(['chmod','+x',str(fname)])
		logging.info('Corry config files were created for all raw files.')
	
		logging.info('Will now run the corry container on each raw file.')
		for p in employee.path_to_directory_of_my_task.iterdir():
			if p.is_dir() and p.name[:3] == 'run':
				logging.info(f'Running corry docker on {p}...')
				result = run_command_in_corry_docker_container(get_run_directory_within_corry_docker(bureaucrat)/employee.task_name/p.name/'tell_corry_docker_to_run_this.sh')
				result.check_returncode()
		logging.info('All noisy pixels masks were completed!')

def corry_align_telescope(bureaucrat:RunBureaucrat, force:bool=False):
	TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/02_align_telescope')
	
	TASK_NAME = 'corry_align_telescope'
	
	if force==False and bureaucrat.was_task_run_successfully(TASK_NAME):
		logging.info(f'Found an already successfull execution of {TASK_NAME} within {bureaucrat.run_name}, will not do anything.')
		return
	
	bureaucrat.check_these_tasks_were_run_successfully('corry_mask_noisy_pixels')
	
	with bureaucrat.handle_task(TASK_NAME) as employee:
		for path_to_raw_file in (bureaucrat.path_to_run_directory/'raw').iterdir():
			logging.info(f'Creating corry config files for raw file {path_to_raw_file.name}...')
			
			raw_file_name_without_extension = path_to_raw_file.parts[-1].replace(".raw","")
			path_to_raw_file_within_docker_container = get_run_directory_within_corry_docker(bureaucrat)/f'raw/{path_to_raw_file.parts[-1]}'
			output_directory_within_corry_docker = get_run_directory_within_corry_docker(bureaucrat)/f'{employee.task_name}/{raw_file_name_without_extension}/corry_output'
			
			path_to_where_to_save_the_config_files = employee.path_to_directory_of_my_task/raw_file_name_without_extension
			path_to_where_to_save_the_config_files.mkdir()
			
			# First of all, copy the geometry file adding the mask noisy pixels extra lines.
			path_to_geometry_file_with_noisy_pixels_mask = path_to_where_to_save_the_config_files/'corry_geometry_for_this_batch_with_noisy_pixels_mask.geo'
			with open(bureaucrat.path_to_run_directory/'corry_geometry_for_this_batch.geo', 'r') as ifile:
				with open(path_to_geometry_file_with_noisy_pixels_mask, 'w') as ofile:
					for line in ifile:
						print(line, file=ofile, end='')
						for n_mimosa in [0,1,2,3,4,5]:
							if f'[MIMOSA26_{n_mimosa}]' in line:
								print(f'mask_file = "../../corry_mask_noisy_pixels/{raw_file_name_without_extension}/corry_output/MaskCreator/MIMOSA26_{n_mimosa}/mask_MIMOSA26_{n_mimosa}.txt"', file=ofile)
			
			# Now create the config files from the templates.
			arguments_for_config_files = {
				'01_prealign-telescope.conf': dict(
					OUTPUT_DIRECTORY = f'"{output_directory_within_corry_docker}"',
					GEOMETRY_FILE = f'"{path_to_geometry_file_with_noisy_pixels_mask.name}"',
					UPDATED_GEOMETRY_FILE = f'"{output_directory_within_corry_docker}/prealign-telescope.geo"',
					PATH_TO_RAW_FILE = f'"{path_to_raw_file_within_docker_container}"',
				),
				'02_align-telescope.conf': dict(
					OUTPUT_DIRECTORY = f'"{output_directory_within_corry_docker}"',
					GEOMETRY_FILE = f'"{output_directory_within_corry_docker}/prealign-telescope.geo"', # The output from the previous one.
					UPDATED_GEOMETRY_FILE = f'"{output_directory_within_corry_docker}/align-telescope.geo"',
					PATH_TO_RAW_FILE = f'"{path_to_raw_file_within_docker_container}"',
				),
				'03_align-telescope-mille.conf': dict(
					OUTPUT_DIRECTORY = f'"{output_directory_within_corry_docker}"',
					GEOMETRY_FILE = f'"{output_directory_within_corry_docker}/align-telescope.geo"', # The output from the previous one.
					UPDATED_GEOMETRY_FILE = f'"{output_directory_within_corry_docker}/align-telescope-mille.geo"',
					PATH_TO_RAW_FILE = f'"{path_to_raw_file_within_docker_container}"',
				),
				'04_check-alignment-telescope.conf': dict(
					OUTPUT_DIRECTORY = f'"{output_directory_within_corry_docker}"',
					GEOMETRY_FILE = f'"{output_directory_within_corry_docker}/align-telescope-mille.geo"', # The output from the previous one.
					PATH_TO_RAW_FILE = f'"{path_to_raw_file_within_docker_container}"',
				),
				'tell_corry_docker_to_run_this.sh': dict(
					WORKING_DIRECTORY = str(output_directory_within_corry_docker.parent),
					GEOMETRY_FILE_FOR_ALIGNMENT_ITERATIONS = str("corry_output/align-telescope.geo"),
				),
			}
			for fname in arguments_for_config_files:
				replace_arguments_in_file_template(
					file_template = TEMPLATE_FILES_DIRECTORY/f'{fname}.template',
					output_file = path_to_where_to_save_the_config_files/fname,
					arguments = arguments_for_config_files[fname],
				)
			
			# Make the scripts executable for docker:
			for fname in path_to_where_to_save_the_config_files.iterdir():
				if fname.suffix == '.sh':
					subprocess.run(['chmod','+x',str(fname)])
		logging.info('Corry config files were created for all raw files.')
	
		# Finally, run corry on each raw file.
		logging.info('Will now run the corry container on each raw file.')
		for p in employee.path_to_directory_of_my_task.iterdir():
			if p.is_dir() and p.name[:3] == 'run':
				logging.info(f'Running corry docker on {p}...')
				result = run_command_in_corry_docker_container(get_run_directory_within_corry_docker(bureaucrat)/employee.task_name/p.name/'tell_corry_docker_to_run_this.sh')
				result.check_returncode()
		logging.info(f'Telescope alignment was completed for all raw files in run {bureaucrat.run_name}!')

def corry_reconstruct_tracks_with_telescope(bureaucrat:RunBureaucrat, force:bool=False):
	TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/03_reconstruct_tracks')
	
	TASK_NAME = 'corry_reconstruct_tracks_with_telescope'
	
	if force==False and bureaucrat.was_task_run_successfully(TASK_NAME):
		logging.info(f'Found an already successfull execution of {TASK_NAME} within {bureaucrat.run_name}, will not do anything.')
		return
	
	bureaucrat.check_these_tasks_were_run_successfully(['corry_mask_noisy_pixels','corry_align_telescope'])
	
	with bureaucrat.handle_task(TASK_NAME) as employee:
		for path_to_raw_file in (bureaucrat.path_to_run_directory/'raw').iterdir():
			logging.info(f'Creating corry config files for raw file {path_to_raw_file.name}...')
			
			raw_file_name_without_extension = path_to_raw_file.parts[-1].replace(".raw","")
			path_to_raw_file_within_docker_container = get_run_directory_within_corry_docker(bureaucrat)/f'raw/{path_to_raw_file.parts[-1]}'
			output_directory_within_corry_docker = get_run_directory_within_corry_docker(bureaucrat)/f'{employee.task_name}/{raw_file_name_without_extension}/corry_output'
			
			path_to_where_to_save_the_config_files = employee.path_to_directory_of_my_task/raw_file_name_without_extension
			path_to_where_to_save_the_config_files.mkdir()
			
			arguments_for_config_files = {
				'01_reconstruct_tracks.conf': dict(
					OUTPUT_DIRECTORY = f'"{output_directory_within_corry_docker}"',
					GEOMETRY_FILE = f'"{output_directory_within_corry_docker.parent.parent.parent}/corry_align_telescope/{raw_file_name_without_extension}/corry_output/align-telescope-mille.geo"',
					PATH_TO_RAW_FILE = f'"{path_to_raw_file_within_docker_container}"',
					NAME_OF_OUTPUT_FILE_WITH_TRACKS = f'"tracks.root"',
				),
				'tell_corry_docker_to_run_this.sh': dict(
					WORKING_DIRECTORY = str(output_directory_within_corry_docker.parent),
				),
				'corry_tracks_root2csv.py': dict(
					PATH_TO_TRACKSdotROOT_FILE = f"'{output_directory_within_corry_docker}/tracks.root'",
				),
			}
			
			for fname in arguments_for_config_files:
				replace_arguments_in_file_template(
					file_template = TEMPLATE_FILES_DIRECTORY/f'{fname}.template',
					output_file = path_to_where_to_save_the_config_files/fname,
					arguments = arguments_for_config_files[fname],
				)
			
			# Make the scripts executable for docker:
			for fname in path_to_where_to_save_the_config_files.iterdir():
				if fname.suffix == '.sh':
					subprocess.run(['chmod','+x',str(fname)])
		logging.info('Corry config files were created for all raw files.')
		
		logging.info('Will now run the corry container on each raw file.')
		for p in employee.path_to_directory_of_my_task.iterdir():
			if p.is_dir() and p.name[:3] == 'run':
				logging.info(f'Running corry docker on {p}...')
				result = run_command_in_corry_docker_container(get_run_directory_within_corry_docker(bureaucrat)/employee.task_name/p.name/'tell_corry_docker_to_run_this.sh')
				result.check_returncode()
				logging.info(f'Corry docker on {p} finished successfully reconstructing the tracks!')
				
				logging.info(f'Converting the CSV file into an SQLite file on {p}...')
				dtype = dict(
					n_event = int,
					n_track = int,
					is_fitted = bool,
					chi2 = float,
					ndof = float,
					Ax = float,
					Ay = float,
					Az = float,
					Bx = float,
					By = float,
					Bz = float,
				)
				with SQLiteDataFrameDumper(p/'tracks.sqlite', dump_after_n_appends=1e3) as tracks_dumper:
					for df in pandas.read_csv(p/output_directory_within_corry_docker.name/'tracks.csv', chunksize=1111, index_col=['n_event','n_track']):
						tracks_dumper.append(df)
				logging.info('CSV to SQLite file conversion successful, CSV file will be removed.')
				(p/output_directory_within_corry_docker.name/'tracks.csv').unlink()
		logging.info('Tracks were reconstructed for all raw files!')

if __name__ == '__main__':
	import sys
	import argparse
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.INFO,
		format = '%(asctime)s|%(levelname)s|%(funcName)s|%(message)s',
		datefmt = '%Y-%m-%d %H:%M:%S',
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
	
	corry_mask_noisy_pixels(
		bureaucrat = bureaucrat,
		force = args.force,
	)
	corry_align_telescope(
		bureaucrat = bureaucrat,
		force = args.force,
	)
	corry_reconstruct_tracks_with_telescope(
		bureaucrat = bureaucrat,
		force = args.force,
	)
