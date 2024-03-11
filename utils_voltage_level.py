from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import pandas
import logging

def create_voltage_point(TB_batch:RunBureaucrat, voltage:int, EUDAQ_runs:list):
	"""Create a new voltage point and link the runs.
	
	Arguments
	---------
	TB_batch: RunBureaucrat
		A bureaucrat pointing to the batch in which to create the new voltage
		point.
	voltage: int
		The voltage value, e.g. `150`.
	EUDAQ_runs: list of int
		A list of int with the EUDAQ run numbers to be linked to this voltage.
	"""
	TB_batch.check_these_tasks_were_run_successfully('EUDAQ_runs')
	
	runs_within_this_batch = {int(_.run_name.split('_')[0].replace('run','')): _.run_name for _ in TB_batch.list_subruns_of_task('EUDAQ_runs')}
	
	with TB_batch.handle_task('voltages', drop_old_data=False) as employee:
		voltage_point = employee.create_subrun(f'{voltage}V', if_exists='skip')
		with voltage_point.handle_task('voltage_batch') as _:
			with open(_.path_to_directory_of_my_task/'README.md','w') as ofile:
				print(f'The sole purpose of this task, called "voltage_batch", is to indicate that this contains the data from a voltage batch, i.e. a constant voltage.', file=ofile)
		with voltage_point.handle_task('EUDAQ_runs') as employee_voltage_runs:
			path_to_subruns = employee_voltage_runs.path_to_directory_of_my_task/'subruns'
			path_to_subruns.mkdir(exist_ok=True)
			for n_run in EUDAQ_runs:
				new_link_in = path_to_subruns/runs_within_this_batch[n_run]
				link_to = f'../../../../../EUDAQ_runs/subruns/{runs_within_this_batch[n_run]}'
				new_link_in.symlink_to(link_to)
				logging.info(f'Linked {new_link_in.name} in {voltage_point.pseudopath}')

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
		help = 'Path to a batch.',
		required = True,
		dest = 'directory',
		type = Path,
	)
	parser.add_argument('--voltage',
		metavar = 'V', 
		help = 'Voltage value, as an integer number, e.g. 150.',
		required = True,
		dest = 'voltage',
		type = int,
	)
	parser.add_argument('--EUDAQ_runs',
		help = 'Runs numbers from EUDAQ that go with this voltage, e.g. `100 101 102`. Note they go as integer numbers.',
		required = True,
		dest = 'EUDAQ_runs',
		type = int,
		nargs = '+',
	)
	args = parser.parse_args()
	
	create_voltage_point(
		TB_batch = RunBureaucrat(args.directory),
		voltage = args.voltage,
		EUDAQ_runs = args.EUDAQ_runs,
	)
