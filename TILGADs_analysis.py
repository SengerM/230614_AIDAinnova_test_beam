from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import utils
import logging
import pandas
import sqlite3
import plotly.express as px
import numpy
import plotly_utils

def setup_TI_LGAD_analysis(bureaucrat, DUT_name):
	"""Setup a directory structure to perform further analysis of a TI-LGAD
	that is inside this batch."""
	bureaucrat.check_these_tasks_were_run_successfully(['batch_info','corry_reconstruct_tracks_with_telescope','parse_waveforms'])
	
	with bureaucrat.handle_task('TI-LGADs_analyses', drop_old_data=False) as employee:
		setup_configuration_info = utils.load_setup_configuration_info(bureaucrat)
		if DUT_name not in set(setup_configuration_info['DUT_name']):
			raise RuntimeError(f'DUT_name {repr(DUT_name)} not present within the set of DUTs in {bureaucrat.run_name}, which is {set(setup_configuration_info["DUT_name"])}')
		
		logging.info(f'Creating directory for analysis of {DUT_name}...')
		TILGAD_bureaucrat = employee.create_subrun(DUT_name)
		for task_name in ['corry_reconstruct_tracks_with_telescope','parse_waveforms','batch_info']:
			(TILGAD_bureaucrat.path_to_run_directory/task_name).symlink_to(Path('../../../'+task_name))
		
		with TILGAD_bureaucrat.handle_task('TI_LGAD_analysis_setup'):
			pass
		
		logging.info(f'Directory for analysis of {DUT_name} created in "{TILGAD_bureaucrat.path_to_run_directory}"')

if __name__ == '__main__':
	import sys
	import argparse
	from plotly_utils import set_my_template_as_default
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.INFO,
		format = '%(asctime)s|%(levelname)s|%(message)s',
		datefmt = '%H:%M:%S',
	)
	
	set_my_template_as_default()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)
	args = parser.parse_args()
	
	bureaucrat = RunBureaucrat(Path(args.directory))
	setup_TI_LGAD_analysis(bureaucrat, DUT_name='TI143')
