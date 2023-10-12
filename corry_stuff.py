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

def corry_mask_noisy_pixels(bureaucrat:RunBureaucrat, corry_container_id:str, force:bool=False, silent_corry:bool=False):
	"""Runs the routine to mask the noisy pixels using corryvreckan. The
	`bureaucrat` must point to a 'raw run' run, for example to this:
	TB_data_analysis/campaigns/subruns/230614_June/batches/subruns/batch_2_200V/runs/subruns/run000930_230623222123"""
	TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/01_mask_noisy_pixels')
	
	bureaucrat.check_these_tasks_were_run_successfully('raw')
	
	TASK_NAME = 'corry_mask_noisy_pixels'
	
	if force==False and bureaucrat.was_task_run_successfully(TASK_NAME):
		logging.info(f'Found an already successfull execution of {TASK_NAME} within {bureaucrat.run_name}, will not do anything.')
		return
	
	with bureaucrat.handle_task(TASK_NAME) as employee:
		path_to_where_to_save_the_config_files = employee.path_to_directory_of_my_task
		
		arguments_for_config_files = {
			'01_mask_noisy_pixels.conf': dict(
				OUTPUT_DIRECTORY = f'"corry_output"',
				GEOMETRY_FILE = f'"../../../../corry_geometry_for_this_batch.geo"',
				PATH_TO_RAW_FILE = f'"../raw/{bureaucrat.run_name}.raw"',
			),
			'tell_corry_docker_to_run_this.sh': dict(
				WORKING_DIRECTORY = str(utils.get_run_directory_within_corry_docker(bureaucrat)/employee.task_name),
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

		logging.info(f'Running mask noisy pixels on {bureaucrat.path_to_run_directory}...')
		result = utils.run_commands_in_docker_container(
			command = str(utils.get_run_directory_within_corry_docker(bureaucrat)/employee.task_name/'tell_corry_docker_to_run_this.sh'),
			container_id = corry_container_id,
			stdout = subprocess.DEVNULL if silent_corry == True else None,
			stderr = subprocess.STDOUT if silent_corry == True else None,
		)
		result.check_returncode()
		logging.info(f'Mask noisy pixels was completed ✅')

def corry_align_telescope(bureaucrat:RunBureaucrat, corry_container_id:str, force:bool=False, silent_corry:bool=False):
	"""Runs the routine to align the telescope using corryvreckan. The
	`bureaucrat` must point to a 'raw run' run, for example to this:
	TB_data_analysis/campaigns/subruns/230614_June/batches/subruns/batch_2_200V/runs/subruns/run000930_230623222123"""
	TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/02_align_telescope')
	
	TASK_NAME = 'corry_align_telescope'
	
	if force==False and bureaucrat.was_task_run_successfully(TASK_NAME):
		logging.info(f'Found an already successfull execution of {TASK_NAME} within {bureaucrat.run_name}, will not do anything.')
		return
	
	bureaucrat.check_these_tasks_were_run_successfully(['raw','corry_mask_noisy_pixels'])
	
	with bureaucrat.handle_task(TASK_NAME) as employee:
		path_to_raw_file_within_docker_container = f'../raw/{bureaucrat.run_name}.raw'

		# First of all, copy the geometry file adding the mask noisy pixels extra lines.
		path_to_geometry_file_with_noisy_pixels_mask = employee.path_to_directory_of_my_task/'corry_geometry_for_this_batch_with_noisy_pixels_mask.geo'
		with open(bureaucrat.parent.path_to_run_directory/'corry_geometry_for_this_batch.geo', 'r') as ifile:
			with open(path_to_geometry_file_with_noisy_pixels_mask, 'w') as ofile:
				for line in ifile:
					print(line, file=ofile, end='')
					for n_mimosa in [0,1,2,3,4,5]:
						if f'[MIMOSA26_{n_mimosa}]' in line:
							print(f'mask_file = "../corry_mask_noisy_pixels/corry_output/MaskCreator/MIMOSA26_{n_mimosa}/mask_MIMOSA26_{n_mimosa}.txt"', file=ofile)
		
		# Now create the config files from the templates.
		arguments_for_config_files = {
			'tell_corry_docker_to_run_this.sh': dict(
				WORKING_DIRECTORY = str(utils.get_run_directory_within_corry_docker(bureaucrat)/employee.task_name),
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
		
		logging.info(f'Running corry telescope align on {bureaucrat.path_to_run_directory}...')
		result = utils.run_commands_in_docker_container(
			command = str(utils.get_run_directory_within_corry_docker(bureaucrat)/employee.task_name/'tell_corry_docker_to_run_this.sh'),
			container_id = corry_container_id,
			stdout = subprocess.DEVNULL if silent_corry == True else None,
			stderr = subprocess.STDOUT if silent_corry == True else None,
		)
		result.check_returncode()
		logging.info(f'Telescope alignment was completed ✅')

def corry_reconstruct_tracks_with_telescope(bureaucrat:RunBureaucrat, corry_container_id:str, force:bool=False, silent_corry:bool=False):
	"""Runs the routine to reconstruct the tracks using corryvreckan on
	all the raw files of the run pointed to by `bureaucrat`. Additionally,
	it creates an SQLite file with the tracks info which is easy to read,
	as well as the ROOT file produced by corry."""
	TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/03_reconstruct_tracks')
	
	TASK_NAME = 'corry_reconstruct_tracks_with_telescope'
	
	if force==False and bureaucrat.was_task_run_successfully(TASK_NAME):
		logging.info(f'Found an already successfull execution of {TASK_NAME} within {bureaucrat.run_name}, will not do anything.')
		return
	
	bureaucrat.check_these_tasks_were_run_successfully(['raw','corry_mask_noisy_pixels','corry_align_telescope'])
	
	with bureaucrat.handle_task(TASK_NAME) as employee:
		arguments_for_config_files = {
			'01_reconstruct_tracks.conf': dict(
				OUTPUT_DIRECTORY = f'"corry_output"',
				GEOMETRY_FILE = f'"../corry_align_telescope/corry_output/align-telescope-mille.geo"',
				PATH_TO_RAW_FILE = f'"../raw/{bureaucrat.run_name}.raw"',
				NAME_OF_OUTPUT_FILE_WITH_TRACKS = f'"tracks.root"',
			),
			'tell_corry_docker_to_run_this.sh': dict(
				WORKING_DIRECTORY = str(utils.get_run_directory_within_corry_docker(bureaucrat)/employee.task_name),
			),
			'corry_tracks_root2csv.py': dict(
				PATH_TO_TRACKSdotROOT_FILE = f"'corry_output/tracks.root'",
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
		
		logging.info(f'Reconstructing tracks with corry on {bureaucrat.path_to_run_directory}...')
		result = utils.run_commands_in_docker_container(
			command = str(utils.get_run_directory_within_corry_docker(bureaucrat)/employee.task_name/'tell_corry_docker_to_run_this.sh'),
			container_id = corry_container_id,
			stdout = subprocess.DEVNULL if silent_corry == True else None,
			stderr = subprocess.STDOUT if silent_corry == True else None,
		)
		result.check_returncode()
		
		logging.info(f'Converting the CSV file into an SQLite file on {bureaucrat.path_to_run_directory}...')
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
		with SQLiteDataFrameDumper(employee.path_to_directory_of_my_task/'tracks.sqlite', dump_after_n_appends=1e3) as tracks_dumper:
			for df in pandas.read_csv(employee.path_to_directory_of_my_task/'corry_output'/'tracks.csv', chunksize=1111, index_col=['n_event','n_track']):
				tracks_dumper.append(df)
		logging.info('CSV to SQLite file conversion successful, CSV file will be removed.')
		(employee.path_to_directory_of_my_task/'corry_output'/'tracks.csv').unlink()
		logging.info(f'Tracks were reconstructed ✅')

def corry_do_all_steps_in_some_run(run:RunBureaucrat, corry_container_id:str, force:bool=False, silent_corry:bool=False):
	corry_mask_noisy_pixels(bureaucrat=run, corry_container_id=corry_container_id, force=force, silent_corry=silent_corry)
	corry_align_telescope(bureaucrat=run, corry_container_id=corry_container_id, force=force, silent_corry=silent_corry)
	corry_reconstruct_tracks_with_telescope(bureaucrat=run, corry_container_id=corry_container_id, force=force, silent_corry=silent_corry)

def load_tracks_for_events_with_track_multiplicity_1(bureaucrat):
	bureaucrat.check_these_tasks_were_run_successfully('corry_reconstruct_tracks_with_telescope')
	
	tracks = []
	for p in bureaucrat.path_to_directory_of_task('corry_reconstruct_tracks_with_telescope').iterdir():
		if not p.is_dir():
			continue
		df = pandas.read_sql(
			'SELECT * FROM dataframe_table GROUP BY n_event HAVING COUNT(n_track) = 1', # Read only events with track multiplicity 1.
			con = sqlite3.connect(p/'tracks.sqlite'),
		)
		df['n_run'] = int(p.parts[-1].split('_')[0].replace('run',''))
		df['n_event'] = df['n_event'] - 1 # Fix an offset that is present in the data, I think it has to do with the extra trigger sent by the TLU when the run starts, that was not sent to the CAENs.
		df.set_index(['n_run','n_event','n_track'], inplace=True)
		tracks.append(df)
	tracks = pandas.concat(tracks)
	
	tracks[['Ax','Ay','Az','Bx','By']] *= 1e-3 # Convert millimeter to meter, it is more natural to work in SI units.
	
	# Check that the track multiplicity is indeed 1 for all events loaded:
	n_tracks_in_event = tracks['is_fitted'].groupby(['n_run','n_event']).count()
	n_tracks_in_event.name = 'n_tracks_in_event'
	if set(n_tracks_in_event) != {1} or len(tracks) == 0:
		raise RuntimeError(f'Failed to load tracks only from events with track multiplicity 1...')
	
	tracks.reset_index('n_track', drop=True, inplace=True)
	return tracks

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
	args = parser.parse_args()
	
	bureaucrat = RunBureaucrat(Path(args.directory))
	
	utils.guess_where_how_to_run(
		bureaucrat = bureaucrat,
		raw_level_f = corry_do_all_steps_in_some_run,
		corry_container_id = args.container_id,
		force = args.force,
		silent_corry = False,
	)
