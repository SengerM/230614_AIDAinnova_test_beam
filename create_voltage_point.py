from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import logging
from DUT_analysis import create_voltage_point

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
