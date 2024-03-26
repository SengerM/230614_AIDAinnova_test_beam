from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import logging
from DUT_analysis import create_DUT_analysis

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
