"""All the code in this file assumes that there is a Docker container
running the 'Jordi`s corry container'."""

from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import utils
import logging
import subprocess
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper # https://github.com/SengerM/huge_dataframe
import pandas
import configparser
import sqlite3
import json

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

def corry_mask_noisy_pixels(EUDAQ_run:RunBureaucrat, corry_container_id:str, force:bool=False, silent_corry:bool=False):
	"""Runs the routine to mask the noisy pixels using corryvreckan. The
	`EUDAQ_run` must point to a 'EUDAQ run' run directory, for example to this:
	TB_data_analysis/campaigns/subruns/230614_June/batches/subruns/batch_2_200V/runs/subruns/run000930_230623222123"""
	TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/01_mask_noisy_pixels')
	
	EUDAQ_run.check_these_tasks_were_run_successfully('raw')
	if not (EUDAQ_run.parent.path_to_run_directory/'corry_geometry_for_this_batch.geo').is_file():
		raise RuntimeError(f'Cannot find `corry_geometry_for_this_batch.geo` file in parent from {EUDAQ_run.pseudopath}, i.e. cannot find it in {EUDAQ_run.parent.path_to_run_directory}')
	
	TASK_NAME = 'corry_mask_noisy_pixels'
	
	if force==False and EUDAQ_run.was_task_run_successfully(TASK_NAME):
		logging.info(f'Found an already successfull execution of {TASK_NAME} within {EUDAQ_run.pseudopath}, will not do anything.')
		return
	
	with EUDAQ_run.handle_task(TASK_NAME) as employee:
		path_to_where_to_save_the_config_files = employee.path_to_directory_of_my_task
		
		# Create a copy of the `.geo` file for the masking process:
		with open(EUDAQ_run.parent.path_to_run_directory/'corry_geometry_for_this_batch.geo', 'r') as ifile:
			with open(employee.path_to_directory_of_my_task/'corry_geometry_for_this_batch_masking.geo', 'w') as ofile:
				for line in ifile:
					print(line, file=ofile, end='')
		
		arguments_for_config_files = {
			'01_mask_noisy_pixels.conf': dict(
				OUTPUT_DIRECTORY = f'"corry_output"',
				GEOMETRY_FILE = f'"corry_geometry_for_this_batch_masking.geo"',
				PATH_TO_RAW_FILE = f'"../raw/{EUDAQ_run.run_name}.raw"',
			),
			'tell_corry_docker_to_run_this.sh': dict(
				WORKING_DIRECTORY = str(utils.get_run_directory_within_corry_docker(EUDAQ_run)/employee.task_name),
			),
		}
		
		for fname in arguments_for_config_files:
			replace_arguments_in_file_template(
				file_template = TEMPLATE_FILES_DIRECTORY/f'{fname}.template',
				output_file = path_to_where_to_save_the_config_files/fname,
				arguments = arguments_for_config_files[fname],
			)
		
		# Make the script executable for docker:
		subprocess.run(['chmod','+x',str(employee.path_to_directory_of_my_task/'tell_corry_docker_to_run_this.sh')])

		logging.info(f'Running mask noisy pixels on {EUDAQ_run.pseudopath}...')
		result = utils.run_commands_in_docker_container(
			command = str(utils.get_run_directory_within_corry_docker(EUDAQ_run)/employee.task_name/'tell_corry_docker_to_run_this.sh'),
			container_id = corry_container_id,
			stdout = subprocess.DEVNULL if silent_corry == True else None,
			stderr = subprocess.STDOUT if silent_corry == True else None,
		)
		result.check_returncode()
		logging.info(f'Mask noisy pixels was completed for {EUDAQ_run.pseudopath} ✅')

def corry_align_telescope(EUDAQ_run:RunBureaucrat, corry_container_id:str, force:bool=False, silent_corry:bool=False):
	"""Runs the routine to align the telescope using corryvreckan. The
	`EUDAQ_run` must point to a 'raw run' run, for example to this:
	TB_data_analysis/campaigns/subruns/230614_June/batches/subruns/batch_2_200V/runs/subruns/run000930_230623222123"""
	TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/02_align_telescope')
	
	TASK_NAME = 'corry_align_telescope'
	
	if force==False and EUDAQ_run.was_task_run_successfully(TASK_NAME):
		logging.info(f'Found an already successfull execution of {TASK_NAME} within {EUDAQ_run.pseudopath}, will not do anything.')
		return
	
	EUDAQ_run.check_these_tasks_were_run_successfully(['raw','corry_mask_noisy_pixels'])
	
	with EUDAQ_run.handle_task(TASK_NAME) as employee:
		path_to_raw_file_within_docker_container = f'../raw/{EUDAQ_run.run_name}.raw'

		# First of all, copy the geometry file adding the mask noisy pixels extra lines.
		path_to_geometry_file_with_noisy_pixels_mask = employee.path_to_directory_of_my_task/'corry_geometry_for_this_batch_with_noisy_pixels_mask.geo'
		with open(EUDAQ_run.parent.path_to_run_directory/'corry_geometry_for_this_batch.geo', 'r') as ifile:
			with open(path_to_geometry_file_with_noisy_pixels_mask, 'w') as ofile:
				for line in ifile:
					print(line, file=ofile, end='')
					for n_mimosa in [0,1,2,3,4,5]:
						if f'[MIMOSA26_{n_mimosa}]' in line:
							print(f'mask_file = "../corry_mask_noisy_pixels/corry_output/MaskCreator/MIMOSA26_{n_mimosa}/mask_MIMOSA26_{n_mimosa}.txt"', file=ofile)
					if '[RD53B_114]' in line:
						print(f'mask_file = "../corry_mask_noisy_pixels/corry_output/MaskCreator/RD53B_114/mask_RD53B_114.txt"', file=ofile)
		
		# Now create the config files from the templates.
		arguments_for_config_files = {
			'tell_corry_docker_to_run_this.sh': dict(
				WORKING_DIRECTORY = str(utils.get_run_directory_within_corry_docker(EUDAQ_run)/employee.task_name),
				GEOMETRY_FILE_FOR_ALIGNMENT_ITERATIONS = "corry_output/align-telescope.geo",
			),
			'01_prealign-telescope.conf': dict(
				OUTPUT_DIRECTORY = f'"corry_output"',
				GEOMETRY_FILE = f'"{path_to_geometry_file_with_noisy_pixels_mask.name}"',
				UPDATED_GEOMETRY_FILE = f'"corry_output/prealign-telescope.geo"',
				PATH_TO_RAW_FILE = f'"{path_to_raw_file_within_docker_container}"',
			),
			'02_align-telescope.conf': dict(
				OUTPUT_DIRECTORY = f'"corry_output"',
				GEOMETRY_FILE = f'"corry_output/prealign-telescope.geo"', # The output from the previous one.
				UPDATED_GEOMETRY_FILE = f'"corry_output/align-telescope.geo"',
				PATH_TO_RAW_FILE = f'"{path_to_raw_file_within_docker_container}"',
			),
			'03_align-telescope-mille.conf': dict(
				OUTPUT_DIRECTORY = f'"corry_output"',
				GEOMETRY_FILE = f'"corry_output/align-telescope.geo"', # The output from the previous one.
				UPDATED_GEOMETRY_FILE = f'"corry_output/align-telescope-mille.geo"',
				PATH_TO_RAW_FILE = f'"{path_to_raw_file_within_docker_container}"',
			),
			'04_check-alignment-telescope.conf': dict(
				OUTPUT_DIRECTORY = f'"corry_output"',
				GEOMETRY_FILE = f'"corry_output/align-telescope-mille.geo"', # The output from the previous one.
				PATH_TO_RAW_FILE = f'"{path_to_raw_file_within_docker_container}"',
			),
		}
		for fname in arguments_for_config_files:
			replace_arguments_in_file_template(
				file_template = TEMPLATE_FILES_DIRECTORY/f'{fname}.template',
				output_file = employee.path_to_directory_of_my_task/fname,
				arguments = arguments_for_config_files[fname],
			)
			
		# Make the script executable for docker:
		subprocess.run(['chmod','+x',str(employee.path_to_directory_of_my_task/'tell_corry_docker_to_run_this.sh')])
		
		logging.info(f'Running corry telescope align on {EUDAQ_run.pseudopath}...')
		result = utils.run_commands_in_docker_container(
			command = str(utils.get_run_directory_within_corry_docker(EUDAQ_run)/employee.task_name/'tell_corry_docker_to_run_this.sh'),
			container_id = corry_container_id,
			stdout = subprocess.DEVNULL if silent_corry == True else None,
			stderr = subprocess.STDOUT if silent_corry == True else None,
		)
		result.check_returncode()
		logging.info(f'Telescope alignment was completed for {EUDAQ_run.pseudopath}✅')

def corry_reconstruct_tracks(EUDAQ_run:RunBureaucrat, corry_container_id:str, force:bool=False, silent_corry:bool=False):
	"""Runs the routine to reconstruct the tracks using corryvreckan on
	all the raw files of the run pointed to by `EUDAQ_run`."""
	TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/03_reconstruct_tracks')
	
	TASK_NAME = 'corry_reconstruct_tracks'
	
	if force==False and EUDAQ_run.was_task_run_successfully(TASK_NAME):
		logging.info(f'Found an already successfull execution of {TASK_NAME} within {EUDAQ_run.pseudopath}, will not do anything.')
		return
	
	EUDAQ_run.check_these_tasks_were_run_successfully(['raw','corry_mask_noisy_pixels','corry_align_telescope'])

	with EUDAQ_run.handle_task(TASK_NAME) as employee:
		arguments_for_config_files = {
			'01_reconstruct_tracks.conf': dict(
				OUTPUT_DIRECTORY = f'"corry_output"',
				GEOMETRY_FILE = f'"../corry_align_telescope/corry_output/align-telescope-mille.geo"',
				PATH_TO_RAW_FILE = f'"../raw/{EUDAQ_run.run_name}.raw"',
				NAME_OF_OUTPUT_FILE_WITH_TRACKS = f'"tracks.root"',
			),
			'tell_corry_docker_to_run_this.sh': dict(
				WORKING_DIRECTORY = str(utils.get_run_directory_within_corry_docker(EUDAQ_run)/employee.task_name),
			),
		}
		
		for fname in arguments_for_config_files:
			replace_arguments_in_file_template(
				file_template = TEMPLATE_FILES_DIRECTORY/f'{fname}.template',
				output_file = employee.path_to_directory_of_my_task/fname,
				arguments = arguments_for_config_files[fname],
			)
		
		# Make the scripts executable for docker:
		for fname in arguments_for_config_files:
			subprocess.run(['chmod','+x',str(employee.path_to_directory_of_my_task/fname)])
		
		logging.info(f'Reconstructing tracks with corry on {EUDAQ_run.pseudopath}...')
		result = utils.run_commands_in_docker_container(
			command = str(utils.get_run_directory_within_corry_docker(EUDAQ_run)/employee.task_name/'tell_corry_docker_to_run_this.sh'),
			container_id = corry_container_id,
			stdout = subprocess.DEVNULL if silent_corry == True else None,
			stderr = subprocess.STDOUT if silent_corry == True else None,
		)
		result.check_returncode()
		logging.info(f'Tracks were reconstructed for {EUDAQ_run.pseudopath} ✅')

def intersect_tracks_with_planes(EUDAQ_run:RunBureaucrat, corry_container_id:str, force:bool=False, silent_corry:bool=False):
	TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/04_tracks_root_to_SQLite')
	
	TASK_NAME = 'intersect_tracks_with_planes'
	
	if force==False and EUDAQ_run.was_task_run_successfully(TASK_NAME):
		logging.info(f'Found an already successfull execution of {TASK_NAME} within {EUDAQ_run.pseudopath}, will not do anything.')
		return
	
	EUDAQ_run.check_these_tasks_were_run_successfully(['corry_reconstruct_tracks'])
	if not (EUDAQ_run.parent.path_to_run_directory/'metadata.json').is_file():
		raise FileNotFoundError(f'Cannot find file `metadata.json` in {EUDAQ_run.parent.path_to_run_directory} in which it should be specified what the z coordinates of the DUTs planes are.')
	
	with EUDAQ_run.handle_task(TASK_NAME) as employee:
		# Read z position of the planes with DUTs from the metadata file:
		with open(EUDAQ_run.parent.path_to_run_directory/'metadata.json', 'r') as ifile:
			metadata = json.load(ifile)
		if not isinstance(metadata, dict) or 'DUTs_planes_z_position' not in metadata.keys() or not isinstance(metadata['DUTs_planes_z_position'], list):
			raise RuntimeError(f'Wrong format in {EUDAQ_run.parent.path_to_run_directory/"metadata.json"} file. I expect to read a dictionary with one of the keys being `"DUTs_planes_z_position": [float, float, float, ...]`')
		if len(metadata['DUTs_planes_z_position']) != 2:
			raise NotImplementedError(f'Currently this functionality is only implemented for 2 planes...')
		
		arguments_for_config_files = {
			'tell_corry_docker_to_run_this.sh': dict(
				WORKING_DIRECTORY = str(utils.get_run_directory_within_corry_docker(EUDAQ_run)/employee.task_name),
			),
			'corry_tracks_root2csv.py': dict(
				PATH_TO_TRACKSdotROOT_FILE = f"'../corry_reconstruct_tracks/corry_output/tracks.root'",
				P1_Z_POSITION = str(metadata['DUTs_planes_z_position'][0]),
				P2_Z_POSITION = str(metadata['DUTs_planes_z_position'][1]),
			),
		}
		
		for fname in arguments_for_config_files:
			replace_arguments_in_file_template(
				file_template = TEMPLATE_FILES_DIRECTORY/f'{fname}.template',
				output_file = employee.path_to_directory_of_my_task/fname,
				arguments = arguments_for_config_files[fname],
			)
		
		# Make the scripts executable for docker:
		for fname in arguments_for_config_files:
			subprocess.run(['chmod','+x',str(employee.path_to_directory_of_my_task/fname)])
		
		logging.info(f'Extracting hit positions from Root file in {EUDAQ_run.pseudopath}...')
		result = utils.run_commands_in_docker_container(
			command = str(utils.get_run_directory_within_corry_docker(EUDAQ_run)/employee.task_name/'tell_corry_docker_to_run_this.sh'),
			container_id = corry_container_id,
			stdout = subprocess.DEVNULL if silent_corry == True else None,
			stderr = subprocess.STDOUT if silent_corry == True else None,
		)
		result.check_returncode()
		
		logging.info(f'Converting the CSV file into an SQLite file on {EUDAQ_run.pseudopath}...')
		dtype = dict(
			n_event = int,
			n_track = int,
			is_fitted = bool,
			chi2 = float,
			ndof = float,
			P1x = float,
			P1y = float,
			P1z = float,
			P2x = float,
			P2y = float,
			P2z = float,
		)
		with SQLiteDataFrameDumper(employee.path_to_directory_of_my_task/'intersects.sqlite', dump_after_n_appends=1e3) as tracks_dumper:
			for df in pandas.read_csv(employee.path_to_directory_of_my_task/'intersects.csv', chunksize=1111, index_col=['n_event','n_track']):
				tracks_dumper.append(df)
		logging.info('CSV to SQLite file conversion successful, CSV file will be removed.')
		(employee.path_to_directory_of_my_task/'intersects.csv').unlink()
		logging.info(f'Tracks intersections with planes extracted for {EUDAQ_run.pseudopath} ✅')

def corry_do_all_steps_in_some_run(EUDAQ_run:RunBureaucrat, corry_container_id:str, force:bool=False, silent_corry:bool=False):
	corry_mask_noisy_pixels(EUDAQ_run=EUDAQ_run, corry_container_id=corry_container_id, force=force, silent_corry=silent_corry)
	corry_align_telescope(EUDAQ_run=EUDAQ_run, corry_container_id=corry_container_id, force=force, silent_corry=silent_corry)
	corry_reconstruct_tracks(EUDAQ_run=EUDAQ_run, corry_container_id=corry_container_id, force=force, silent_corry=silent_corry)
	intersect_tracks_with_planes(EUDAQ_run=EUDAQ_run, corry_container_id=corry_container_id, force=force, silent_corry=silent_corry)

if __name__ == '__main__':
	import sys
	import argparse
	import my_telegram_bots # Secret tokens from my bots
	from progressreporting.TelegramProgressReporter import SafeTelegramReporter4Loops # https://github.com/SengerM/progressreporting
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.INFO,
		format = '%(asctime)s|%(levelname)s|%(funcName)s|%(message)s',
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
	parser.add_argument('--container_id',
		metavar = 'id', 
		help = 'Id of the docker container running `Jordis corry docker`. Once the container is already running, you can get the id with `docker ps`.',
		required = True,
		dest = 'container_id',
		type = str,
	)
	parser.add_argument(
		'--force',
		help = 'If this flag is passed, it will force the processing even if it was already done beforehand. Old data will be deleted.',
		required = False,
		dest = 'force',
		action = 'store_true'
	)
	parser.add_argument(
		'--show_corry_output',
		help = 'If this flag is passed, all the output from corry will be printed, otherwise it is hidden.',
		required = False,
		dest = 'show_corry_output',
		action = 'store_true'
	)
	args = parser.parse_args()
	
	bureaucrat = RunBureaucrat(Path(args.directory))
	
	utils.guess_where_how_to_run(
		bureaucrat = bureaucrat,
		raw_level_f = lambda bureaucrat: corry_do_all_steps_in_some_run(EUDAQ_run=bureaucrat, corry_container_id = args.container_id, force = args.force,	silent_corry = not args.show_corry_output),
		telegram_bot_reporter = SafeTelegramReporter4Loops(
			bot_token = my_telegram_bots.robobot.token,
			chat_id = my_telegram_bots.chat_ids['Robobot TCT setup'],
		),
	)
