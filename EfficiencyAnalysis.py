from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import logging
import VoltagePoint
from pathlib import Path
import json

def create_two_pixels_efficiency_analysis(voltage_point_dn:DatanodeHandler, analysis_name:str, left_pixel_chubut_channel_number:int, right_pixel_chubut_channel_number:int, x_center_of_the_pair_of_pixels:float, y_center_of_the_pair_of_pixels:float, rotation_angle_deg:float, y_acceptance_width:float):
	"""Create a new "two pixels efficiency analysis" inside a voltage point.
	
	Arguments
	---------
	voltage_point_dn: DatanodeHandler
		A `DatanodeHandler` pointing to a voltage point.
	"""
	with voltage_point_dn.handle_task('two_pixels_efficiency_analyses', check_datanode_class='voltage_point', keep_old_data=True) as task:
		analysis_dn = task.create_subdatanode(analysis_name, subdatanode_class='two_pixels_efficiency_analysis')
		
		with analysis_dn.handle_task('analysis_config') as analysis_config_task:
			with open(analysis_config_task.path_to_directory_of_my_task/'config.json', 'w') as ofile:
				json.dump(
					dict(
						left_pixel_chubut_channel_number = left_pixel_chubut_channel_number,
						right_pixel_chubut_channel_number = right_pixel_chubut_channel_number,
						x_center_of_the_pair_of_pixels = x_center_of_the_pair_of_pixels,
						y_center_of_the_pair_of_pixels = y_center_of_the_pair_of_pixels,
						rotation_angle_deg = rotation_angle_deg,
						y_acceptance_width = y_acceptance_width,
					),
					indent = '\t',
					fp = ofile,
				)
		
	logging.info(f'Analysis efficiency {repr(str(analysis_dn.pseudopath))} was created. ✅')
	

class DatanodeHandlerTwoPixelsEfficiencyAnalysis(DatanodeHandler):
	def __init__(self, path_to_datanode:Path):
		super().__init__(path_to_datanode=path_to_datanode, check_datanode_class='two_pixels_efficiency_analysis')
	
	@property
	def parent(self):
		return super().parent.as_type(VoltagePoint.DatanodeHandlerVoltagePoint)
	
