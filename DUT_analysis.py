from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import pandas
import logging
import TBBatch
import json
import VoltagePoint
import dominate
import dominate.tags as tags
	
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
		help = 'Path to a TB_batch datanode.',
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
	args = parser.parse_args()
	
	DUT_analysis_dn = DatanodeHandlerDUTAnalysis(args.datanode)
	if args.plot_waveforms_distributions:
		DUT_analysis_dn.plot_waveforms_distributions()
	if args.plot_hits:
		DUT_analysis_dn.plot_hits(amplitude_threshold = .01)
