from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import pandas
import logging
import utils_batch_level
import json

def create_DUT_analysis(TB_batch_dn:DatanodeHandler, DUT_name:str, plane_number:int, chubut_channel_numbers:set):
	"""Creates a datanode to handle the analysis of one DUT.
	
	Arguments
	---------
	TB_batch_dn: DatanodeHandler
		A `DatanodeHandler` pointing to a test beam batch datanode.
	DUT_name: str
		The name of the DUT. It can be anything, and will be used later on
		to identify this analysis.
	plane_number: int
		Plane number as it is in the spreadsheet from the test beam data.
	chubut_channel_numbers: set of int
		Chubut channel numbers from the pixels that belong to this DUT. 
		This is needed because we have tested more than one DUT per plane 
		at DESY, which is not something standard, so now we need to 
		separate them.
	"""
	with TB_batch_dn.handle_task('DUTs_analyses', check_datanode_class='TB_batch', check_required_tasks='batch_info', keep_old_data=True) as task_handler:
		setup_config = utils_batch_level.load_setup_configuration_info(TB_batch_dn)
		if plane_number not in setup_config['plane_number']:
			raise RuntimeError(f'`plane_number` {plane_number} not found in the setup_config from batch {repr(str(TB_batch_dn.pseudopath))}. ')
		if any([ch not in setup_config.query(f'plane_number=={plane_number}')['chubut_channel'] for ch in chubut_channel_numbers]):
			raise RuntimeError(f'At least one `chubut_channel_numbers` {chubut_channel_numbers} not present in the setup_config of batch {repr(str(TB_batch_dn.pseudopath))}. ')
		DUT_analysis_dn = task_handler.create_subdatanode(DUT_name, subdatanode_class='DUT_analysis')
		with DUT_analysis_dn.handle_task('setup_config_metadata') as setup_config_metadata_task:
			with open(setup_config_metadata_task.path_to_directory_of_my_task/'metadata.json', 'w') as ofile:
				json.dump(
					dict(
						plane_number = plane_number,
						chubut_channels_numbers = chubut_channel_numbers,
					), 
					ofile, 
					indent = '\t',
				)
		logging.info(f'DUT analysis {repr(str(DUT_analysis_dn.pseudopath))} was created. ✅')

def create_voltage_point(DUT_analysis_dn:DatanodeHandler, voltage:int, EUDAQ_runs:set):
	"""Create a new voltage point within a DUT analysis.
	
	Arguments
	---------
	DUT_analysis_dn: DatanodeHandler
		A `DatanodeHandler` pointing to a DUT analysis in which to create
		the new voltage	point.
	voltage: int
		The voltage value, e.g. `150`.
	EUDAQ_runs: set of int
		A set of int with the EUDAQ run numbers to be linked to this voltage.
	"""
	with DUT_analysis_dn.handle_task('voltages', check_datanode_class='DUT_analysis', keep_old_data=True) as task:
		TB_batch_dn = DUT_analysis_dn.parent
		EUDAQ_runs_within_this_batch = {int(r.datanode_name.split('_')[0].replace('run','')):r.datanode_name for r in TB_batch_dn.list_subdatanodes_of_task('EUDAQ_runs')}
		
		if any([_ not in EUDAQ_runs_within_this_batch for _ in EUDAQ_runs]):
			raise RuntimeError(f'At least one of the runs {EUDAQ_runs} is not available in batch {repr(str(TB_batch_dn.pseudopath))}. Available runs found are: {sorted(EUDAQ_runs_within_this_batch)}. ')
		
		voltage_dn = task.create_subdatanode(f'{voltage}V', subdatanode_class='voltage_point')
		
		with voltage_dn.handle_task('EUDAQ_runs') as EUDAQ_runs_task:
			with open(EUDAQ_runs_task.path_to_directory_of_my_task/'runs.json', 'w') as ofile:
				json.dump(
					[EUDAQ_runs_within_this_batch[_] for _ in EUDAQ_runs],
					ofile,
					indent = '\t',
				)
	logging.info(f'Voltage point {repr(str(voltage_dn.pseudopath))} was created. ✅')

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
	parser.add_argument('--datanode',
		help = 'Path to a TB_batch datanode.',
		dest = 'datanode',
		type = Path,
	)
	parser.add_argument('--voltage',
		help = 'Value of voltage.',
		dest = 'voltage',
		type = int,
	)
	parser.add_argument('--EUDAQ_runs',
		help = 'EUDAQ run numbers as integers',
		dest = 'EUDAQ_runs',
		type = int,
		nargs = '+',
	)
	args = parser.parse_args()
	
	create_voltage_point(
		DUT_analysis_dn = DatanodeHandler(args.datanode), 
		voltage = args.voltage,
		EUDAQ_runs = args.EUDAQ_runs,
	)
