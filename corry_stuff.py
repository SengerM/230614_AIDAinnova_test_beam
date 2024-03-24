"""All the code in this file assumes that there is a Docker container
running the 'Jordi`s corry container'."""

from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import utils
import logging
import subprocess
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper # https://github.com/SengerM/huge_dataframe
import pandas
import configparser
import sqlite3
import json
import uproot
import numpy

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

def corry_mask_noisy_pixels(EUDAQ_run_dn:DatanodeHandler, corry_container_id:str, force:bool=False, silent_corry:bool=False):
	"""Runs the routine to mask the noisy pixels using corryvreckan."""
	TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/01_mask_noisy_pixels')
	
	TASK_NAME = 'corry_mask_noisy_pixels'
	
	if force==False and EUDAQ_run_dn.was_task_run_successfully(TASK_NAME):
		return
	
	with EUDAQ_run_dn.handle_task(
		TASK_NAME,
		check_datanode_class = 'EUDAQ_run',
		check_required_tasks = {'raw','hardcode_Jordis_geometry_file_Feb2024DESY'},
	) as task_handler:
		# Create a copy of the `.geo` file for the masking process:
		with open(EUDAQ_run_dn.path_to_directory_of_task('hardcode_Jordis_geometry_file_Feb2024DESY')/'geometry.geo', 'r') as ifile:
			with open(task_handler.path_to_directory_of_my_task/'geometry.geo', 'w') as ofile:
				for line in ifile:
					if 'mask_file' in line:
						# We are now creating the mask, so we skip these lines.
						continue
					print(line, file=ofile, end='')
		
		arguments_for_config_files = {
			'01_mask_noisy_pixels.conf': dict(
				OUTPUT_DIRECTORY = f'"corry_output"',
				GEOMETRY_FILE = f'"geometry.geo"',
				PATH_TO_RAW_FILE = f'"../raw/{EUDAQ_run_dn.datanode_name}.raw"',
			),
			'tell_corry_docker_to_run_this.sh': dict(
				WORKING_DIRECTORY = str(utils.get_datanode_directory_within_corry_docker(EUDAQ_run_dn)/task_handler.task_name),
			),
		}
		
		for fname in arguments_for_config_files:
			replace_arguments_in_file_template(
				file_template = TEMPLATE_FILES_DIRECTORY/f'{fname}.template',
				output_file = task_handler.path_to_directory_of_my_task/fname,
				arguments = arguments_for_config_files[fname],
			)
		
		# Make the script executable for docker:
		subprocess.run(['chmod','+x',str(task_handler.path_to_directory_of_my_task/'tell_corry_docker_to_run_this.sh')])

		logging.info(f'Running mask noisy pixels on {EUDAQ_run_dn.pseudopath}...')
		result = utils.run_commands_in_docker_container(
			command = str(utils.get_datanode_directory_within_corry_docker(EUDAQ_run_dn)/task_handler.task_name/'tell_corry_docker_to_run_this.sh'),
			container_id = corry_container_id,
			stdout = subprocess.DEVNULL if silent_corry == True else None,
			stderr = subprocess.STDOUT if silent_corry == True else None,
		)
		result.check_returncode()
		logging.info(f'Mask noisy pixels was completed for {EUDAQ_run_dn.pseudopath} ✅')

# ~ def corry_align_telescope(EUDAQ_run:RunBureaucrat, corry_container_id:str, force:bool=False, silent_corry:bool=False):
	# ~ """Runs the routine to align the telescope using corryvreckan. The
	# ~ `EUDAQ_run` must point to a 'raw run' run, for example to this:
	# ~ TB_data_analysis/campaigns/subruns/230614_June/batches/subruns/batch_2_200V/runs/subruns/run000930_230623222123"""
	# ~ TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/02_align_telescope')
	
	# ~ TASK_NAME = 'corry_align_telescope'
	
	# ~ if force==False and EUDAQ_run.was_task_run_successfully(TASK_NAME):
		# ~ logging.info(f'Found an already successfull execution of {TASK_NAME} within {EUDAQ_run.pseudopath}, will not do anything.')
		# ~ return
	
	# ~ EUDAQ_run.check_these_tasks_were_run_successfully(['raw','corry_mask_noisy_pixels'])
	
	# ~ with EUDAQ_run.handle_task(TASK_NAME) as employee:
		# ~ path_to_raw_file_within_docker_container = f'../raw/{EUDAQ_run.run_name}.raw'

		# ~ # First of all, copy the geometry file adding the mask noisy pixels extra lines.
		# ~ path_to_geometry_file_with_noisy_pixels_mask = employee.path_to_directory_of_my_task/'corry_geometry_for_this_batch_with_noisy_pixels_mask.geo'
		# ~ with open(EUDAQ_run.parent.path_to_run_directory/'corry_geometry_for_this_batch.geo', 'r') as ifile:
			# ~ with open(path_to_geometry_file_with_noisy_pixels_mask, 'w') as ofile:
				# ~ for line in ifile:
					# ~ print(line, file=ofile, end='')
					# ~ for n_mimosa in [0,1,2,3,4,5]:
						# ~ if f'[MIMOSA26_{n_mimosa}]' in line:
							# ~ print(f'mask_file = "../corry_mask_noisy_pixels/corry_output/MaskCreator/MIMOSA26_{n_mimosa}/mask_MIMOSA26_{n_mimosa}.txt"', file=ofile)
					# ~ if '[RD53B_114]' in line:
						# ~ print(f'mask_file = "../corry_mask_noisy_pixels/corry_output/MaskCreator/RD53B_114/mask_RD53B_114.txt"', file=ofile)
		
		# ~ # Now create the config files from the templates.
		# ~ arguments_for_config_files = {
			# ~ 'tell_corry_docker_to_run_this.sh': dict(
				# ~ WORKING_DIRECTORY = str(utils.get_run_directory_within_corry_docker(EUDAQ_run)/employee.task_name),
				# ~ GEOMETRY_FILE_FOR_ALIGNMENT_ITERATIONS = "corry_output/align-telescope.geo",
			# ~ ),
			# ~ '01_prealign-telescope.conf': dict(
				# ~ OUTPUT_DIRECTORY = f'"corry_output"',
				# ~ GEOMETRY_FILE = f'"{path_to_geometry_file_with_noisy_pixels_mask.name}"',
				# ~ UPDATED_GEOMETRY_FILE = f'"corry_output/prealign-telescope.geo"',
				# ~ PATH_TO_RAW_FILE = f'"{path_to_raw_file_within_docker_container}"',
			# ~ ),
			# ~ '02_align-telescope.conf': dict(
				# ~ OUTPUT_DIRECTORY = f'"corry_output"',
				# ~ GEOMETRY_FILE = f'"corry_output/prealign-telescope.geo"', # The output from the previous one.
				# ~ UPDATED_GEOMETRY_FILE = f'"corry_output/align-telescope.geo"',
				# ~ PATH_TO_RAW_FILE = f'"{path_to_raw_file_within_docker_container}"',
			# ~ ),
			# ~ '03_align-telescope-mille.conf': dict(
				# ~ OUTPUT_DIRECTORY = f'"corry_output"',
				# ~ GEOMETRY_FILE = f'"corry_output/align-telescope.geo"', # The output from the previous one.
				# ~ UPDATED_GEOMETRY_FILE = f'"corry_output/align-telescope-mille.geo"',
				# ~ PATH_TO_RAW_FILE = f'"{path_to_raw_file_within_docker_container}"',
			# ~ ),
			# ~ '04_check-alignment-telescope.conf': dict(
				# ~ OUTPUT_DIRECTORY = f'"corry_output"',
				# ~ GEOMETRY_FILE = f'"corry_output/align-telescope-mille.geo"', # The output from the previous one.
				# ~ PATH_TO_RAW_FILE = f'"{path_to_raw_file_within_docker_container}"',
			# ~ ),
		# ~ }
		# ~ for fname in arguments_for_config_files:
			# ~ replace_arguments_in_file_template(
				# ~ file_template = TEMPLATE_FILES_DIRECTORY/f'{fname}.template',
				# ~ output_file = employee.path_to_directory_of_my_task/fname,
				# ~ arguments = arguments_for_config_files[fname],
			# ~ )
			
		# ~ # Make the script executable for docker:
		# ~ subprocess.run(['chmod','+x',str(employee.path_to_directory_of_my_task/'tell_corry_docker_to_run_this.sh')])
		
		# ~ logging.info(f'Running corry telescope align on {EUDAQ_run.pseudopath}...')
		# ~ result = utils.run_commands_in_docker_container(
			# ~ command = str(utils.get_run_directory_within_corry_docker(EUDAQ_run)/employee.task_name/'tell_corry_docker_to_run_this.sh'),
			# ~ container_id = corry_container_id,
			# ~ stdout = subprocess.DEVNULL if silent_corry == True else None,
			# ~ stderr = subprocess.STDOUT if silent_corry == True else None,
		# ~ )
		# ~ result.check_returncode()
		# ~ logging.info(f'Telescope alignment was completed for {EUDAQ_run.pseudopath}✅')

def corry_reconstruct_tracks(EUDAQ_run_dn:DatanodeHandler, corry_container_id:str, force:bool=False, silent_corry:bool=False):
	"""Runs the routine to reconstruct the tracks using corryvreckan on
	all the raw files of the datanode pointed to by `EUDAQ_run_dn`."""
	TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/03_reconstruct_tracks')
	
	TASK_NAME = 'corry_reconstruct_tracks'
	
	if force==False and EUDAQ_run_dn.was_task_run_successfully(TASK_NAME):
		return
	
	with EUDAQ_run_dn.handle_task(
		TASK_NAME, 
		check_datanode_class = 'EUDAQ_run', 
		check_required_tasks = {'raw','corry_mask_noisy_pixels','hardcode_Jordis_geometry_file_Feb2024DESY'},
	) as task_handler:
		arguments_for_config_files = {
			'01_reconstruct_tracks.conf': dict(
				OUTPUT_DIRECTORY = f'"corry_output"',
				GEOMETRY_FILE = f'"../hardcode_Jordis_geometry_file_Feb2024DESY/geometry.geo"',
				PATH_TO_RAW_FILE = f'"../raw/{EUDAQ_run_dn.datanode_name}.raw"',
				NAME_OF_OUTPUT_FILE_WITH_TRACKS = f'"tracks.root"',
			),
			'tell_corry_docker_to_run_this.sh': dict(
				WORKING_DIRECTORY = str(utils.get_datanode_directory_within_corry_docker(EUDAQ_run_dn)/task_handler.task_name),
			),
		}
		
		for fname in arguments_for_config_files:
			replace_arguments_in_file_template(
				file_template = TEMPLATE_FILES_DIRECTORY/f'{fname}.template',
				output_file = task_handler.path_to_directory_of_my_task/fname,
				arguments = arguments_for_config_files[fname],
			)
		
		# Make the scripts executable for docker:
		for fname in arguments_for_config_files:
			subprocess.run(['chmod','+x',str(task_handler.path_to_directory_of_my_task/fname)])
		
		logging.info(f'Reconstructing tracks with corry on {EUDAQ_run_dn.pseudopath}...')
		result = utils.run_commands_in_docker_container(
			command = str(utils.get_datanode_directory_within_corry_docker(EUDAQ_run_dn)/task_handler.task_name/'tell_corry_docker_to_run_this.sh'),
			container_id = corry_container_id,
			stdout = subprocess.DEVNULL if silent_corry == True else None,
			stderr = subprocess.STDOUT if silent_corry == True else None,
		)
		result.check_returncode()
		logging.info(f'Tracks were reconstructed for {EUDAQ_run_dn.pseudopath} ✅')

# ~ def corry_align_CROC(EUDAQ_run:RunBureaucrat, corry_container_id:str, force:bool=False, silent_corry:bool=False):
	# ~ """Runs the routine to reconstruct the tracks using corryvreckan on
	# ~ all the raw files of the run pointed to by `EUDAQ_run`."""
	# ~ TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/04_align_CROC')
	
	# ~ TASK_NAME = 'corry_align_CROC'
	
	# ~ if force==False and EUDAQ_run.was_task_run_successfully(TASK_NAME):
		# ~ logging.info(f'Found an already successfull execution of {TASK_NAME} within {EUDAQ_run.pseudopath}, will not do anything.')
		# ~ return
	
	# ~ EUDAQ_run.check_these_tasks_were_run_successfully(['raw','corry_reconstruct_tracks'])

	# ~ with EUDAQ_run.handle_task(TASK_NAME) as employee:
		# ~ arguments_for_config_files = {
			# ~ '01_prealign.conf': dict(
				# ~ OUTPUT_DIRECTORY = f'"corry_output"',
				# ~ GEOMETRY_FILE = f'"../corry_align_telescope/corry_output/align-telescope-mille.geo"', # Comes from the previous step.
				# ~ UPDATED_GEOMETRY_FILE = f'"corry_output/geometry.geo"', # Create a new one within this step's directory.
				# ~ PATH_TO_RAW_FILE = f'"../raw/{EUDAQ_run.run_name}.raw"',
			# ~ ),
			# ~ '02_align.conf': dict(
				# ~ OUTPUT_DIRECTORY = f'"corry_output"',
				# ~ GEOMETRY_FILE = f'"corry_output/geometry.geo"',
				# ~ UPDATED_GEOMETRY_FILE = f'"corry_output/geometry.geo"', # Overwrite it, this will run many times in an iterative way.
				# ~ PATH_TO_ROOT_FILE_WITH_TRACKS = f'"../corry_reconstruct_tracks/corry_output/tracks.root"', # From previous track reconstruction step.
			# ~ ),
			# ~ '03_check-alignment-all.conf': dict(
				# ~ OUTPUT_DIRECTORY = f'"corry_output"',
				# ~ GEOMETRY_FILE = f'"corry_output/geometry.geo"', # This is the new geometry file resulting from this "corry_align_CROC".
				# ~ PATH_TO_RAW_FILE = f'"../raw/{EUDAQ_run.run_name}.raw"',
			# ~ ),
			# ~ 'tell_corry_docker_to_run_this.sh': dict(
				# ~ WORKING_DIRECTORY = str(utils.get_run_directory_within_corry_docker(EUDAQ_run)/employee.task_name),
			# ~ ),
		# ~ }
		
		# ~ for fname in arguments_for_config_files:
			# ~ replace_arguments_in_file_template(
				# ~ file_template = TEMPLATE_FILES_DIRECTORY/f'{fname}.template',
				# ~ output_file = employee.path_to_directory_of_my_task/fname,
				# ~ arguments = arguments_for_config_files[fname],
			# ~ )
		
		# ~ # Make the scripts executable for docker:
		# ~ for fname in arguments_for_config_files:
			# ~ subprocess.run(['chmod','+x',str(employee.path_to_directory_of_my_task/fname)])
		
		# ~ logging.info(f'Aligning CROC on {EUDAQ_run.pseudopath}...')
		# ~ result = utils.run_commands_in_docker_container(
			# ~ command = str(utils.get_run_directory_within_corry_docker(EUDAQ_run)/employee.task_name/'tell_corry_docker_to_run_this.sh'),
			# ~ container_id = corry_container_id,
			# ~ stdout = subprocess.DEVNULL if silent_corry == True else None,
			# ~ stderr = subprocess.STDOUT if silent_corry == True else None,
		# ~ )
		# ~ result.check_returncode()
		# ~ logging.info(f'Finished aligning CROC on {EUDAQ_run.pseudopath} ✅')

def corry_intersect_tracks_with_planes(EUDAQ_run_dn:DatanodeHandler, corry_container_id:str, force:bool=False, silent_corry:bool=False):
	TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/05_intersect_tracks_with_DUT_planes')
	
	TASK_NAME = 'corry_intersect_tracks_with_planes'
	
	if force==False and EUDAQ_run_dn.was_task_run_successfully(TASK_NAME):
		return
	
	with EUDAQ_run_dn.handle_task(
		TASK_NAME,
		check_datanode_class = 'EUDAQ_run',
		check_required_tasks = {'corry_reconstruct_tracks','hardcode_Jordis_geometry_file_Feb2024DESY'},
	) as task_handler:
		# Read the 'geometry.geo' file from the previous step, to extract automatically DUT_names and z_planes, needed for this step.
		geo_config = configparser.ConfigParser()
		geo_config.read(EUDAQ_run_dn.path_to_directory_of_task('hardcode_Jordis_geometry_file_Feb2024DESY')/'geometry.geo')
		geo_config = {s:dict(geo_config.items(s)) for s in geo_config.sections()} # Convert to a dictionary. Easier object for a configuration.
		# Extract DUT_names and z_planes from the geometry file:
		DUT_names = [_ for _ in geo_config.keys() if geo_config[_].get('role')=='"dut"']
		z_planes = [geo_config[_]['position'].split(',')[2] for _ in DUT_names if 'RD53' not in _] # Extract z_position of the DUTs (except the CROC RD53, which we don't care).
		
		arguments_for_config_files = {
			'tell_corry_docker_to_run_this.sh': dict(
				WORKING_DIRECTORY = str(utils.get_datanode_directory_within_corry_docker(EUDAQ_run_dn)/task_handler.task_name),
			),
			'write-tracktree.conf': dict(
				OUTPUT_DIRECTORY = f'"corry_output"',
				GEOMETRY_FILE = f'"../hardcode_Jordis_geometry_file_Feb2024DESY/geometry.geo"', # From previous step.
				PATH_TO_RAW_FILE = f'"../raw/{EUDAQ_run_dn.datanode_name}.raw"',
				DUT_NAMES = str(DUT_names)[1:-1].replace("'",'"'),
				Z_PLANES = str(z_planes)[1:-1].replace("'",''),
			),
		}
		
		for fname in arguments_for_config_files:
			replace_arguments_in_file_template(
				file_template = TEMPLATE_FILES_DIRECTORY/f'{fname}.template',
				output_file = task_handler.path_to_directory_of_my_task/fname,
				arguments = arguments_for_config_files[fname],
			)
		
		# Make the scripts executable for docker:
		for fname in arguments_for_config_files:
			subprocess.run(['chmod','+x',str(task_handler.path_to_directory_of_my_task/fname)])
		
		logging.info(f'Intersecting tracks with planes on {EUDAQ_run_dn.pseudopath}...')
		result = utils.run_commands_in_docker_container(
			command = str(utils.get_datanode_directory_within_corry_docker(EUDAQ_run_dn)/task_handler.task_name/'tell_corry_docker_to_run_this.sh'),
			container_id = corry_container_id,
			stdout = subprocess.DEVNULL if silent_corry == True else None,
			stderr = subprocess.STDOUT if silent_corry == True else None,
		)
		result.check_returncode()
		logging.info(f'Tracks intersections with planes completed on {EUDAQ_run_dn.pseudopath} ✅')

def convert_tracks_root_file_to_easy_SQLite(EUDAQ_run_dn:DatanodeHandler, force:bool=False):
	TASK_NAME = 'convert_tracks_root_file_to_easy_SQLite'
	
	if force==False and EUDAQ_run_dn.was_task_run_successfully(TASK_NAME):
		return
	
	with EUDAQ_run_dn.handle_task(
		TASK_NAME,
		check_datanode_class = 'EUDAQ_run',
		check_required_tasks = 'corry_intersect_tracks_with_planes',
	) as task_handler:
		logging.info(f'Converting ROOT file with tracks into SQLite file for {task_handler.pseudopath}...')
		path_to_root_file = EUDAQ_run_dn.path_to_directory_of_task('corry_intersect_tracks_with_planes')/'corry_output/TreeWriterTracks/tracks.root'
		with uproot.open(path_to_root_file) as root_file:
			with SQLiteDataFrameDumper(task_handler.path_to_directory_of_my_task/'tracks.sqlite', dump_after_n_appends=1e3) as dumper:
				tracks_tree = root_file['tracks']
				
				EventIDs = tracks_tree['EventID'].array()
				data_raw = {_:tracks_tree[_].array() for _ in tracks_tree.keys() if _ not in {'EventID'}}
				
				# Now we iterate. Unfortunately, the format is such that this cannot be avoided, because the track multiplicity is nested in a dimension of the arrays.
				for n_event,EventID in enumerate(EventIDs):
					for n_track,TrackID in enumerate(data_raw['TrackID'][n_event]): # This is the thing that, unfortunately, cannot be avoided...
						this_track_data = {
							key: data_raw[key][n_event][n_track] for key in tracks_tree.keys() if key not in {'EventID'}
						}
						this_track_data['EventID'] = EventID # This branch has a different format from the other branches.
						this_track_data['n_event'] = EventID # This is the index I will use later on to match the data from the digitizers with the tracks. I don't use `n_track` (the variable) here because there is an arbitrary offset WRT `EventID`, given by the fact that all events for which Corry could not reconstruct any track are not here, creating thus this offset.
						this_track_data['n_track'] = n_track
						this_track_data = pandas.DataFrame(this_track_data, index=[0])
						this_track_data.set_index(['n_event','n_track'], inplace=True)
						dumper.append(this_track_data)
			logging.info(f'Finished converting ROOT file with tracks into SQLite for {EUDAQ_run_dn.pseudopath} ✅')

def hardcode_Jordis_geometry_file_Feb2024DESY(EUDAQ_run_dn:DatanodeHandler):
	"""This function is here because at the moment I cannot make the 
	reconstruction machinery to properly align the RD53 plane. Jordi sent
	a `.geo` file that he somehow managed to produce, and seems to work.
	Since this should not change a lot within the test beam, this function
	just copy-pastes that geometry file and make it available."""
	TEMPLATE_FILES_DIRECTORY = Path(__file__).parent.resolve()/Path('corry_templates/hardcode')
	
	with EUDAQ_run_dn.handle_task(
		'hardcode_Jordis_geometry_file_Feb2024DESY',
		check_datanode_class = 'EUDAQ_run',
	) as task_handler:
		with open(TEMPLATE_FILES_DIRECTORY/'Jordi_geometry_aligned_all_Feb2024DESY.geo', 'r') as ifile:
			with open(task_handler.path_to_directory_of_my_task/'geometry.geo', 'w') as ofile:
				for line in ifile:
					print(line, file=ofile, end='')

def corry_do_all_steps_in_some_run(EUDAQ_run_dn:DatanodeHandler, corry_container_id:str, force:bool=False, silent_corry:bool=False):
	hardcode_Jordis_geometry_file_Feb2024DESY(EUDAQ_run_dn=EUDAQ_run_dn)
	corry_mask_noisy_pixels(EUDAQ_run_dn=EUDAQ_run_dn, corry_container_id=corry_container_id, force=force, silent_corry=silent_corry)
	# ~ corry_align_telescope(EUDAQ_run=EUDAQ_run, corry_container_id=corry_container_id, force=force, silent_corry=silent_corry)
	corry_reconstruct_tracks(EUDAQ_run_dn=EUDAQ_run_dn, corry_container_id=corry_container_id, force=force, silent_corry=silent_corry)
	# ~ corry_align_CROC(EUDAQ_run=EUDAQ_run, corry_container_id=corry_container_id, force=force, silent_corry=silent_corry)
	corry_intersect_tracks_with_planes(EUDAQ_run_dn=EUDAQ_run_dn, corry_container_id=corry_container_id, force=force, silent_corry=silent_corry)
	convert_tracks_root_file_to_easy_SQLite(EUDAQ_run_dn=EUDAQ_run_dn, force=force)

if __name__ == '__main__':
	import sys
	import argparse
	import my_telegram_bots # Secret tokens from my bots
	from progressreporting.TelegramProgressReporter import SafeTelegramReporter4Loops # https://github.com/SengerM/progressreporting
	from utils_run_level import execute_EUDAQ_run_task_on_all_runs_within_batch
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.INFO,
		format = '%(asctime)s|%(levelname)s|%(funcName)s|%(message)s',
		datefmt = '%H:%M:%S',
	)
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--datanode',
		metavar = 'path', 
		help = 'Path to a datanode.',
		required = True,
		dest = 'datanode',
		type = Path,
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
	
	dn = DatanodeHandler(args.datanode)
	execute_EUDAQ_run_task_on_all_runs_within_batch(
		TB_batch_dn = dn,
		func = corry_do_all_steps_in_some_run,
		args = {_.datanode_name:dict(corry_container_id=args.container_id, force=args.force, silent_corry=not args.show_corry_output) for _ in dn.list_subdatanodes_of_task('EUDAQ_runs')},
		telegram_bot_reporter = SafeTelegramReporter4Loops(
			bot_token = my_telegram_bots.robobot.token,
			chat_id = my_telegram_bots.chat_ids['Robobot TCT setup'],
		),
	)
