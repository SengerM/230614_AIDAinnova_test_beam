from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import logging
from EfficiencyAnalysis import create_two_pixels_efficiency_analysis
from VoltagePoint import DatanodeHandlerVoltagePoint

if __name__ == '__main__':
	import sys
	from plotly_utils import set_my_template_as_default
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.INFO,
		format = '%(asctime)s|%(levelname)s|%(message)s',
		datefmt = '%H:%M:%S',
	)
	
	set_my_template_as_default()
	
	create_two_pixels_efficiency_analysis(
		voltage_point_dn = DatanodeHandlerVoltagePoint(input('Path to voltage point datanode? ')),
		analysis_name = input('Analysis name? '),
		left_pixel_chubut_channel_number = int(input('Left pixel chubut channel number? ')),
		right_pixel_chubut_channel_number = int(input('Right pixel chubut channel number? ')),
		x_center_of_the_pair_of_pixels = float(input('x center of the pair of pixels, in original coordinates? ')),
		y_center_of_the_pair_of_pixels = float(input('y center of the pair of pixels, in original coordinates? ')),
		rotation_angle_deg = float(input('Rotation angle so that left and right pixels are actually left and right? (deg) ')),
		y_acceptance_width = float(input('y acceptance width? ')),
	)
