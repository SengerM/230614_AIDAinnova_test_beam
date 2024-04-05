from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import logging
import EUDAQRun

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
		help = 'Path to an EUDAQ run datanode.',
		dest = 'datanode',
		type = Path,
	)
	args = parser.parse_args()
	
	EUDAQRun.find_EUDAQ_offset(
		EUDAQ_run_dn = DatanodeHandler(args.datanode), 
	)
