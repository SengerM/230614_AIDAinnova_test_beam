from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import numpy
import pandas
import utils
import logging
import plotly.graph_objects as go
import plotly.express as px
import dominate # https://github.com/Knio/dominate
import DUT_analysis
import EUDAQRun

class DatanodeHandlerTBBatch(DatanodeHandler):
	def __init__(self, path_to_datanode:Path):
		super().__init__(path_to_datanode=path_to_datanode, check_datanode_class='TB_batch')
	
	def list_subdatanodes_of_task(self, task_name:str):
		subdatanodes = super().list_subdatanodes_of_task(task_name)
		if task_name == 'DUTs_analyses':
			subdatanodes = [_.as_type(DUT_analysis.DatanodeHandlerDUTAnalysis) for _ in subdatanodes]
		if task_name == 'EUDAQ_runs':
			subdatanodes = [_.as_type(EUDAQRun.DatanodeHandlerEUDAQRun) for _ in subdatanodes]
		return subdatanodes

	def setup_batch_info(self):
		"""Add some batch-wise information needed for the analysis, like
		for example a link to the setup connection spreadsheet."""
		def setup_batch_info_June_test_beam(TB_batch_dn:DatanodeHandler):
			raise NotImplementedError('This function has to be rewritten translating `RunBureaucrat` to `DatanodeHandler`. I did not have the time to do it.')
			TB_batch.check_these_tasks_were_run_successfully('EUDAQ_runs') # So we are sure this is pointing to a batch
			
			n_batch = int(TB_batch.run_name.split('_')[1])
			if n_batch in {2,3,4}:
				path_to_setup_connection_ods = Path('/media/msenger/230829_gray/AIDAinnova_test_beams/raw_data/230614_June/AIDAInnova_June/setup_connections')/f'Batch{n_batch}.ods'
			elif n_batch in {5,6}:
				path_to_setup_connection_ods = Path('/media/msenger/230829_gray/AIDAinnova_test_beams/raw_data/230614_June/CMS-ETL_June/setup_connections')/f'setup_connections_Batch{n_batch}.ods'
			else:
				raise RuntimeError(f'Cannot determine batch name appropriately!')
			
			with TB_batch.handle_task('batch_info') as employee:
				for sheet_name in {'planes','signals'}:
					df = pandas.read_excel(
						path_to_setup_connection_ods,
						sheet_name = sheet_name,
					).set_index('plane_number')
					utils.save_dataframe(
						df,
						name = sheet_name,
						location = employee.path_to_directory_of_my_task,
					)
		
		def setup_batch_info_August_test_beam(TB_batch_dn:DatanodeHandler):
			raise NotImplementedError('This function has to be rewritten translating `RunBureaucrat` to `DatanodeHandler`. I did not have the time to do it.')
			TB_batch.check_these_tasks_were_run_successfully('EUDAQ_runs') # So we are sure this is pointing to a batch
			
			with TB_batch.handle_task('batch_info') as employee:
				n_batch = int(TB_batch.run_name.split('_')[1])
				planes_definition = pandas.read_csv(
					'https://docs.google.com/spreadsheets/d/e/2PACX-1vTuRXCnGCPu8nuTrrh_6M_QaBYwVQZfmLX7cr6OlM-ucf9yx3KIbBN4XBQxc0fTp-O26Y2QIOCkgP98/pub?gid=0&single=true&output=csv',
					dtype = dict(
						batch_number = int,
						plane_number = int,
						DUT_name = str,
						orientation = str,
						high_voltage_source = str,
						low_voltage_source = str,
					),
					index_col = ['batch_number','plane_number'],
				)
				pixels_definition = pandas.read_csv(
					'https://docs.google.com/spreadsheets/d/e/2PACX-1vTuRXCnGCPu8nuTrrh_6M_QaBYwVQZfmLX7cr6OlM-ucf9yx3KIbBN4XBQxc0fTp-O26Y2QIOCkgP98/pub?gid=1673457618&single=true&output=csv',
					dtype = dict(
						batch_number = int,
						plane_number = int,
						chubut_channel_number = int,
						digitizer_name = str,
						digitizer_channel_name = str,
						row = int,
						col = int,
					),
					index_col = ['batch_number','plane_number'],
				)
				for name,df in {'planes_definition':planes_definition, 'pixels_definition':pixels_definition}.items():
					utils.save_dataframe(df.query(f'batch_number=={n_batch}'), name, employee.path_to_directory_of_my_task)
		
		def setup_batch_info_240212_DESY_test_beam(TB_batch_dn:DatanodeHandler):
			with TB_batch_dn.handle_task('batch_info', check_datanode_class='TB_batch') as task_handler:
				n_batch = int(TB_batch_dn.datanode_name.split('_')[1])
				logging.info(f'Fetching data from Google spreadsheets for {TB_batch_dn.pseudopath}...')
				planes_definition = pandas.read_csv(
					'https://docs.google.com/spreadsheets/d/e/2PACX-1vTSddEy02EJHBoeyBsh44eupHMynrxzYHOPc8WogDnaLpPtm7Dy-PdAyB-aQjp2xawgsWciw0RxUMRK/pub?gid=0&single=true&output=csv',
					dtype = dict(
						batch_number = int,
						plane_number = int,
						DUT_name = str,
						orientation = str,
						high_voltage_source = str,
						low_voltage_source = str,
					),
					index_col = ['batch_number','plane_number'],
				)
				pixels_definition = pandas.read_csv(
					'https://docs.google.com/spreadsheets/d/e/2PACX-1vTSddEy02EJHBoeyBsh44eupHMynrxzYHOPc8WogDnaLpPtm7Dy-PdAyB-aQjp2xawgsWciw0RxUMRK/pub?gid=1673457618&single=true&output=csv',
					dtype = dict(
						batch_number = int,
						plane_number = int,
						chubut_channel_number = int,
						digitizer_name = str,
						digitizer_channel_name = str,
						row = int,
						col = int,
					),
					index_col = ['batch_number','plane_number'],
				)
				for name,df in {'planes_definition':planes_definition, 'pixels_definition':pixels_definition}.items():
					utils.save_dataframe(df.query(f'batch_number=={n_batch}'), name, task_handler.path_to_directory_of_my_task)
		
		TB_campaign = self.parent # The parent of the batch should be the TB campaign.
		if TB_campaign is None or TB_campaign.datanode_class != 'TB_campaign':
			raise RuntimeError(f'I was expecting that the parent of batch "{self.pseudopath}" is a test beam campaign, but it is not. ')
		
		match TB_campaign.datanode_name:
			case '230614_June':
				setup_batch_info_June_test_beam(self)
			case '230830_August':
				setup_batch_info_August_test_beam(self)
			case '240212_DESY':
				setup_batch_info_240212_DESY_test_beam(self)
			case _:
				raise RuntimeError(f'Cannot determine which test beam campaign {self.pseudopath} belongs to...')
		logging.info(f'Setup info was set for batch {self.pseudopath} ✅')

	def load_setup_configuration_info(self)->pandas.DataFrame:
		# Check that all the tasks we need were properly executed already:
		self.check_these_tasks_were_run_successfully(['EUDAQ_runs','batch_info'])
		if not all([dn.was_task_run_successfully('parse_waveforms') for dn in self.list_subdatanodes_of_task('EUDAQ_runs')]):
			raise RuntimeError(f'To load the setup configuration it is needed that all of the runs of the batch have had the `parse_waveforms` task performed on them, but does not seem to be the case')
		
		TB_campaign = self.parent # The parent of the batch should be the TB campaign.
		if TB_campaign is None or TB_campaign.datanode_class != 'TB_campaign':
			raise RuntimeError(f'I was expecting that the parent of batch "{self.pseudopath}" is a test beam campaign, but it is not. ')
		
		match TB_campaign.datanode_name: # This is the test beam campaign, the parent of every batch.
			case '230614_June':
				planes = pandas.read_pickle(self.path_to_directory_of_task('batch_info')/'planes.pickle')
				signals_connections = pandas.read_pickle(self.path_to_directory_of_task('batch_info')/'signals.pickle')
			case '230830_August':
				planes = pandas.read_pickle(self.path_to_directory_of_task('batch_info')/'planes_definition.pickle')
				signals_connections = pandas.read_pickle(self.path_to_directory_of_task('batch_info')/'pixels_definition.pickle')
				for df in [planes,signals_connections]:
					df.rename(
						columns = {
							'digitizer_name': 'CAEN_name',
							'digitizer_channel_name': 'CAEN_channel_name',
							'chubut_channel_number': 'chubut_channel',
						},
						inplace = True,
					)
			case '240212_DESY':
				planes = pandas.read_pickle(self.path_to_directory_of_task('batch_info')/'planes_definition.pickle')
				signals_connections = pandas.read_pickle(self.path_to_directory_of_task('batch_info')/'pixels_definition.pickle')
				for df in [planes,signals_connections]:
					df.rename(
						columns = {
							'digitizer_name': 'CAEN_name',
							'digitizer_channel_name': 'CAEN_channel_name',
							'chubut_channel_number': 'chubut_channel',
						},
						inplace = True,
					)
			case _:
				raise RuntimeError(f'Cannot read setup information for batch "{self.datanode_name}". ')
		
		CAENs_names = []
		for run_dn in self.list_subdatanodes_of_task('EUDAQ_runs'):
			n_run = int(run_dn.datanode_name.split('_')[0].replace('run',''))
			df = pandas.read_pickle(run_dn.path_to_directory_of_task('parse_waveforms')/f'parsed_from_waveforms_CAENs_names.pickle')
			df = df.to_frame()
			df['n_run'] = n_run
			df.set_index('n_run',append=True,inplace=True)
			CAENs_names.append(df)
		CAENs_names = pandas.concat(CAENs_names)
		CAENs_names['CAEN_name'] = CAENs_names['CAEN_name'].apply(lambda x: x.replace('CAEN_',''))
		
		# Here we assume that the CAENs were not changed within a batch, which is not only reasonable but also what we expect.
		_ = CAENs_names.reset_index('n_CAEN',drop=False).set_index('CAEN_name',append=True).reset_index('n_run',drop=True)
		_ = _[~_.index.duplicated(keep='first')]
		signals_connections = signals_connections.reset_index(drop=False).merge(
			_,
			on = 'CAEN_name',
		)
		
		CAENs_CHANNELS_MAPPING_TO_INTEGERS = pandas.DataFrame(
			# This codification into integers comes from the producer, see in line 208 of `CAENDT5742Producer.py`. The reason is that EUDAQ can only handle integers tags, or something like this.
			{
				'CAEN_n_channel': list(range(18)),
				'CAEN_channel_name': [f'CH{i}' if i<16 else f'trigger_group_{i-16}' for i in range(18)]
			}
		)
		
		signals_connections = signals_connections.merge(
			CAENs_CHANNELS_MAPPING_TO_INTEGERS.set_index('CAEN_channel_name')['CAEN_n_channel'],
			on = 'CAEN_channel_name',
		)
		
		signals_connections = signals_connections.merge(
			planes[['DUT_name','z (m)']],
			on = 'plane_number',
		)
		
		signals_connections['CAEN_trigger_group_n'] = signals_connections['CAEN_n_channel'].apply(lambda x: 0 if x in {0,1,2,3,4,5,6,7,16} else 1 if x in {8,9,10,11,12,13,14,15,17} else -1)
		
		signals_connections['rowcol'] = signals_connections[['row','col']].apply(lambda x: f'{x["row"]}{x["col"]}', axis=1)
		signals_connections['DUT_name_rowcol'] = signals_connections[['DUT_name','row','col']].apply(lambda x: f'{x["DUT_name"]} ({x["row"]},{x["col"]})', axis=1)
		
		return signals_connections
	
	@property
	def setup_config(self):
		if not hasattr(self, '_setup_config'):
			self._setup_config = self.load_setup_configuration_info()
		return self._setup_config
	
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
		metavar = 'path', 
		help = 'Path to a datanode.',
		required = True,
		dest = 'datanode',
		type = Path,
	)
	parser.add_argument(
		'--setup_batch_info',
		help = 'If this flag is passed, it will execute `setup_batch_info`.',
		required = False,
		dest = 'setup_batch_info',
		action = 'store_true'
	)
	args = parser.parse_args()
	
	dn = DatanodeHandlerTBBatch(args.datanode)
	if args.setup_batch_info:
		dn.setup_batch_info()
