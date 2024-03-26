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

def create_voltage_point(DUT_analysis_dn:DatanodeHandler, voltage:int, EUDAQ_runs:list):
	"""Create a new voltage point and link the runs.
	
	Arguments
	---------
	DUT_analysis: RunBureaucrat
		A bureaucrat pointing to a DUT analysis in which to create the new voltage
		point.
	voltage: int
		The voltage value, e.g. `150`.
	EUDAQ_runs: list of int
		A list of int with the EUDAQ run numbers to be linked to this voltage.
	"""
	raise NotImplementedError()

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
	parser.add_argument('--DUT_name',
		help = 'Name for the DUT analysis',
		dest = 'DUT_name',
		type = str,
	)
	parser.add_argument('--plane_number',
		help = 'Plane number of this DUT.',
		dest = 'plane_number',
		type = int,
	)
	parser.add_argument('--chubut_channel_numbers',
		help = 'Chubut channels belonging to this DUT.',
		dest = 'chubut_channel_numbers',
		type = int,
		nargs = '+',
	)
	args = parser.parse_args()
	
	create_DUT_analysis(
		TB_batch_dn = DatanodeHandler(args.datanode), 
		DUT_name = args.DUT_name,
		plane_number = args.plane_number,
		chubut_channel_numbers = args.chubut_channel_numbers,
	)
