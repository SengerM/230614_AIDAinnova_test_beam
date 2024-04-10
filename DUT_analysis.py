from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import pandas
import logging
import TBBatch
import json
import VoltagePoint
import dominate
import dominate.tags as tags
import EfficiencyAnalysis
import plotly.express as px
from utils import save_dataframe
	
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
		setup_config = TBBatch.DatanodeHandlerTBBatch.load_setup_configuration_info(TB_batch_dn)
		if plane_number not in setup_config['plane_number']:
			raise RuntimeError(f'`plane_number` {plane_number} not found in the setup_config from batch {repr(str(TB_batch_dn.pseudopath))}. ')
		if any([ch not in setup_config.query(f'plane_number=={plane_number}')['chubut_channel'].values for ch in chubut_channel_numbers]):
			raise RuntimeError(f'At least one `chubut_channel_numbers` {chubut_channel_numbers} not present in the setup_config of batch {repr(str(TB_batch_dn.pseudopath))} for plane number {plane_number}. ')
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

class DatanodeHandlerDUTAnalysis(DatanodeHandler):
	def __init__(self, path_to_datanode:Path):
		super().__init__(path_to_datanode=path_to_datanode, check_datanode_class='DUT_analysis')
	
	def list_subdatanodes_of_task(self, task_name:str):
		subdatanodes = super().list_subdatanodes_of_task(task_name)
		if task_name == 'voltages':
			subdatanodes = [_.as_type(VoltagePoint.DatanodeHandlerVoltagePoint) for _ in subdatanodes]
		return subdatanodes
	
	@property
	def parent(self):
		return super().parent.as_type(TBBatch.DatanodeHandlerTBBatch)
	
	def load_DUT_configuration_metadata(self):
		self.check_datanode_class('DUT_analysis')
		with open(self.path_to_directory_of_task('setup_config_metadata')/'metadata.json', 'r') as ifile:
			loaded = json.load(ifile)
			this_DUT_chubut_channels = loaded['chubut_channels_numbers']
			this_DUT_plane_number = loaded['plane_number']
		if len(this_DUT_chubut_channels) == 0:
			raise RuntimeError(f'No `chubut channels` associated with DUT in {repr(str(self.pseudopath))}. ')
		if not isinstance(this_DUT_plane_number, int):
			raise RuntimeError(f'Cannot determine plane number for DUT in {repr(str(self.pseudopath))}. ')
		
		return loaded
	
	def plot_waveforms_distributions(self, max_points_to_plot_per_voltage=9999, histograms=['Amplitude (V)','Collected charge (V s)','t_50 (s)','Rise time (s)','SNR','Time over 50% (s)'], scatter_plots=[('t_50 (s)','Amplitude (V)'),('Time over 50% (s)','Amplitude (V)')]):
		with self.handle_task('plot_waveforms_distributions', 'DUT_analysis', 'voltages') as task:
			for voltage_dn in self.list_subdatanodes_of_task('voltages'):
				voltage_dn.plot_waveforms_distributions(
					max_points_to_plot = max_points_to_plot_per_voltage,
					histograms = histograms,
					scatter_plots = scatter_plots,
				)
			
			voltages = sorted(self.list_subdatanodes_of_task('voltages'), key=DatanodeHandler.datanode_name.__get__)
			
			for kind_of_plot in {'histograms','scatter_plots'}:
				save_plots_here = task.path_to_directory_of_my_task/kind_of_plot
				save_plots_here.mkdir()
				
				for plot_file_name in [_.name for _ in (self.list_subdatanodes_of_task('voltages')[0].path_to_directory_of_task('plot_waveforms_distributions')/kind_of_plot).iterdir()]:
					doc = dominate.document(title=plot_file_name.split('.')[0])
					with doc:
						tags.h1(plot_file_name.split('.')[0] + ' distributions')
						tags.h3(str(self.pseudopath))
						for voltage_dn in voltages:
							tags.iframe(
								src = Path('../..')/((voltage_dn.path_to_directory_of_task('plot_waveforms_distributions')/kind_of_plot/plot_file_name).relative_to(self.path_to_datanode_directory)),
								style = 'height: 90vh; width: 100%; min-height: 666px; min-width: 666px; border: 0;',
							)
					with open(save_plots_here/plot_file_name, 'w') as ofile:
						print(doc, file=ofile)
		logging.info(f'Finished plotting waveforms distributions in {self.pseudopath} ✅')
	
	def plot_hits(self, amplitude_threshold:float):
		"""Plot hits projected onto the DUT.
		
		Arguments
		---------
		amplitude_threshold: float
			Threshold in the amplitude to consider the activation of the pixels
			in the DUT. For more details see the definition of the function
			in `VoltagePoint.py`.
		"""
		with self.handle_task('plot_hits', 'DUT_analysis', 'voltages') as task:
			for voltage_dn in self.list_subdatanodes_of_task('voltages'):
				voltage_dn.plot_hits(amplitude_threshold = amplitude_threshold)
			
			voltages = sorted(self.list_subdatanodes_of_task('voltages'), key=DatanodeHandler.datanode_name.__get__)
			
			doc = dominate.document(title=f'Hits on {self.pseudopath}')
			with doc:
				tags.h1('Hits on DUT')
				tags.h3(str(self.pseudopath))
				for voltage_dn in voltages:
					tags.iframe(
						src = Path('..')/((voltage_dn.path_to_directory_of_task('plot_hits')/'hits.html').relative_to(self.path_to_datanode_directory)),
						style = 'height: 90vh; width: 100%; min-height: 666px; min-width: 666px; border: 0;',
					)
			with open(task.path_to_directory_of_my_task/'hits.html', 'w') as ofile:
				print(doc, file=ofile)
			
			logging.info(f'Finished plotting hits in {self.pseudopath} ✅')
	
	def centering_transformation(self):
		"""Prompts for the transformation parameters to be used later on.
		Since this is not easy to automatize, it relies on a human to get
		the transformation parameters."""
		with self.handle_task('centering_transformation', check_required_tasks='plot_hits') as task:
			input(f'Open the plot with the hits in {self.pseudopath} and press enter. ')
			while True:
				x_center = float(input('What is the x center of the DUT? '))
				y_center = float(input('What is the y center of the DUT? '))
				rotation_angle = float(input('What is the rotation angle of the DUT? (in deg) '))
				transformation_parameters = dict(
					x_center = x_center,
					y_center = y_center,
					rotation_angle = rotation_angle,
				)
				
				# Now let's plot things using these transformation parameters and ask the user if he is happy with that:
				for voltage_dn in self.list_subdatanodes_of_task('voltages'):
					voltage_dn.plot_hits(amplitude_threshold = .01, transformation_parameters=transformation_parameters)
				
				voltages = sorted(self.list_subdatanodes_of_task('voltages'), key=DatanodeHandler.datanode_name.__get__)
				
				doc = dominate.document(title=f'Transformed hits on {self.pseudopath}')
				with doc:
					tags.h1('Hits on DUT with transformation applied')
					tags.h3(str(self.pseudopath))
					tags.p('Transformation parameters:\n' + json.dumps(transformation_parameters, indent=4))
					for voltage_dn in voltages:
						tags.iframe(
							src = Path('..')/((voltage_dn.path_to_directory_of_task('plot_hits_transformed')/'hits.html').relative_to(self.path_to_datanode_directory)),
							style = 'height: 90vh; width: 100%; min-height: 666px; min-width: 666px; border: 0;',
						)
				with open(task.path_to_directory_of_my_task/'hits_transformed.html', 'w') as ofile:
					print(doc, file=ofile)
				
				if 'yes' == input(f'Look at the plot in {self.pseudopath} inside {repr(str(task.task_name))	}. Are you happy with the results? (yes/no) '):
					with open(task.path_to_directory_of_my_task/'transformation_parameters.json', 'w') as ofile:
						json.dump(transformation_parameters, ofile)
					break
				print(f"Let's start over...")
		logging.info(f'Transformation parameters configured for {self.pseudopath} ✅')
	
	def create_two_pixels_efficiency_analyses(self, analysis_name:str, left_pixel_chubut_channel_number:int, right_pixel_chubut_channel_number:int, x_center_of_the_pair_of_pixels:float, y_center_of_the_pair_of_pixels:float, rotation_angle_deg:float, y_acceptance_width:float):
		"""Create a "two pixels efficiency analysis" for all the voltage 
		points within this DUT. For the arguments, see the documentation
		of the function that actually does the job."""
		
		for voltage_point_dn in self.list_subdatanodes_of_task('voltages'):
			EfficiencyAnalysis.create_two_pixels_efficiency_analysis(
				voltage_point_dn = voltage_point_dn,
				analysis_name = analysis_name,
				left_pixel_chubut_channel_number = left_pixel_chubut_channel_number,
				right_pixel_chubut_channel_number = right_pixel_chubut_channel_number,
				x_center_of_the_pair_of_pixels = x_center_of_the_pair_of_pixels,
				y_center_of_the_pair_of_pixels = y_center_of_the_pair_of_pixels,
				rotation_angle_deg = rotation_angle_deg,
				y_acceptance_width = y_acceptance_width,
			)
		logging.info(f'Created "two pixels efficiency analyses" for all the voltages in {self.pseudopath} ✅')
	
	def run_two_pixels_efficiency_analyses(self, hit_amplitude_threshold:float, bin_size_meters:float, max_chi2ndof:float=None):
		"""Run all the analyses that are defined in the subtree of this
		DUT, all with the same parameters."""
		CATEGORY_ORDERS = {
			'which_pixel': ['none','left','right','both'],
		}
		
		with self.handle_task('run_two_pixels_efficiency_analyses') as task:
			efficiency_data = [] # Store the data here so then I can create some plots sumarizing at the DUT level.
			for voltage_dn in self.list_subdatanodes_of_task('voltages'):
				for two_pixels_analysis_dn in voltage_dn.list_subdatanodes_of_task('two_pixels_efficiency_analyses'):
					two_pixels_analysis_dn.plot_hits(hit_amplitude_threshold, max_chi2ndof=max_chi2ndof)
					two_pixels_analysis_dn.binned_efficiency_vs_distance(
						hit_amplitude_threshold = hit_amplitude_threshold,
						bin_size_meters = bin_size_meters,
						max_chi2ndof=max_chi2ndof,
					)
					eff = pandas.read_pickle(two_pixels_analysis_dn.path_to_directory_of_task('efficiency_vs_distance')/'efficiency.pickle')
					eff['voltage'] = int(voltage_dn.datanode_name.replace('V',''))
					eff['two_pixels_efficiency_analysis_name'] = two_pixels_analysis_dn.datanode_name
					efficiency_data.append(eff)
			efficiency_data = pandas.concat(efficiency_data)
			efficiency_data.set_index(['voltage','two_pixels_efficiency_analysis_name'], append=True, inplace=True)
			
			save_dataframe(efficiency_data, 'efficiency', task.path_to_directory_of_my_task)
			
			plots_subtitle = f'{self.pseudopath}, amplitude < {-abs(hit_amplitude_threshold)*1e3} mV' + (f', χ<sup>2</sup>/n<sub>deg of freedom</sub> < {max_chi2ndof}' if max_chi2ndof is not None else '')
			for analysis_name, data in efficiency_data.groupby('two_pixels_efficiency_analysis_name'):
				fig = px.line(
					title = f'Efficiency vs distance {analysis_name}<br><sup>{plots_subtitle}</sup>',
					data_frame = data.reset_index().sort_values(['voltage','Position (m)']),
					color = 'which_pixel',
					y = 'val (%)',
					x = 'Position (m)',
					error_y = 'err_high (%)',
					error_y_minus = 'err_low (%)',
					facet_col = 'voltage',
					line_shape = 'hvh',
					labels = {
						'val (%)': 'Efficiency (%)',
					},
					category_orders = CATEGORY_ORDERS,
				)
				fig.write_html(
					task.path_to_directory_of_my_task/f'efficiency_vs_distance_{analysis_name}.html',
					include_plotlyjs = 'cdn',
				)
		logging.info(f'Plots of efficiency vs distance available in {self.pseudopath} ✅')
	
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
	parser.add_argument(
		'--datanode',
		help = 'Path to a DUT analysis datanode.',
		dest = 'datanode',
		type = Path,
	)
	parser.add_argument(
		'--plot_waveforms_distributions',
		dest = 'plot_waveforms_distributions',
		action = 'store_true',
	)
	parser.add_argument(
		'--plot_hits',
		dest = 'plot_hits',
		action = 'store_true',
	)
	parser.add_argument(
		'--centering_transformation',
		dest = 'centering_transformation',
		action = 'store_true',
	)
	parser.add_argument(
		'--run_two_pixels_efficiency_analyses',
		dest = 'run_two_pixels_efficiency_analyses',
		action = 'store_true',
	)
	parser.add_argument(
		'--hit_amplitude_threshold',
		dest = 'hit_amplitude_threshold',
		type = float,
		required = False,
	)
	parser.add_argument(
		'--max_chi2_n_dof',
		dest = 'max_chi2_n_dof',
		type = float,
		required = False,
	)
	parser.add_argument(
		'--bin_size_meters',
		dest = 'bin_size_meters',
		type = float,
		required = False,
		default = 50e-6
	)
	args = parser.parse_args()
	
	DUT_analysis_dn = DatanodeHandlerDUTAnalysis(args.datanode)
	if args.plot_waveforms_distributions:
		DUT_analysis_dn.plot_waveforms_distributions()
	if args.plot_hits:
		DUT_analysis_dn.plot_hits(amplitude_threshold = .01)
	if args.centering_transformation:
		DUT_analysis_dn.centering_transformation()
	if args.run_two_pixels_efficiency_analyses:
		DUT_analysis_dn.run_two_pixels_efficiency_analyses(
			hit_amplitude_threshold = args.hit_amplitude_threshold,
			bin_size_meters = args.bin_size_meters,
			max_chi2ndof = args.max_chi2_n_dof,
		)
