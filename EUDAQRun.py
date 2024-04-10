from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import numpy
import pandas
import sqlite3
import logging
from progressreporting.TelegramProgressReporter import SafeTelegramReporter4Loops # https://github.com/SengerM/progressreporting
from contextlib import nullcontext
import corry_stuff
import TBBatch
import plotly.express as px
import json

def execute_EUDAQ_run_task_on_all_runs_within_batch(TB_batch_dn:DatanodeHandler, func:callable, args:dict, telegram_bot_reporter:SafeTelegramReporter4Loops=None):
	TB_batch_dn.check_datanode_class('TB_batch')
	
	EUDAQ_runs_to_be_processed = TB_batch_dn.list_subdatanodes_of_task('EUDAQ_runs')
	
	with telegram_bot_reporter.report_loop(
		total_loop_iterations = len(EUDAQ_runs_to_be_processed),
		loop_name = f'{func.__name__} on {TB_batch_dn.pseudopath}',
	) if telegram_bot_reporter is not None else nullcontext() as reporter:
		for EUDAQ_run_dn in EUDAQ_runs_to_be_processed:
			logging.info(f'Running `{func.__name__}` on {EUDAQ_run_dn.pseudopath}...')
			func(EUDAQ_run_dn, **(args[EUDAQ_run_dn.datanode_name]))
			reporter.update(1) if telegram_bot_reporter is not None else None
	logging.info(f'Finished running `{func.__name__}` on all EUDAQ_runs within TB_batch `{TB_batch_dn.pseudopath}` ✅')

class DatanodeHandlerEUDAQRun(DatanodeHandler):
	def __init__(self, path_to_datanode:Path):
		super().__init__(path_to_datanode=path_to_datanode, check_datanode_class='EUDAQ_run')
	
	@property
	def parent(self):
		return super().parent.as_type(TBBatch.DatanodeHandlerTBBatch) # Forcing EUDAQ runs to always be inside a TB batch.
	
	def load_parsed_from_waveforms(self, where:str, variables:list=None):
		"""Load data parsed from waveforms from an EUDAQ run.
		
		Arguments
		---------
		where: str
			The statement that will be placed in the SQL query.
		variables: list of str
			A list with the variables to be loaded, e.g. `['Amplitude (V)','t_50 (s)']`.
		"""
		logging.info(f'Reading {variables} from {self.pseudopath}...')
		
		if isinstance(variables, str):
			variables = [variables]
		if variables is not None:
			if len(variables) == 0:
				variables = None
			variables = ',' + ','.join([f'`{_}`' for _ in variables])
		else:
			variables = ''
		data = pandas.read_sql(
			f'SELECT n_event,n_CAEN,CAEN_n_channel{variables} FROM dataframe_table WHERE {where}',
			con = sqlite3.connect(self.path_to_directory_of_task('parse_waveforms')/f'parsed_from_waveforms.sqlite'),
		)
		data.set_index(['n_event','n_CAEN','CAEN_n_channel'], inplace=True)
		return data
	
	def find_EUDAQ_offset(self):
		with self.handle_task(
			'find_EUDAQ_offset', 
			check_required_tasks = ['parse_waveforms','convert_tracks_root_file_to_easy_SQLite'],
		) as task:
			setup_config = self.parent.load_setup_configuration_info()
			CAENs_detected_a_hit = self.load_parsed_from_waveforms(
				where = '`Amplitude (V)` < -.01 AND `Time over 50% (s)` > 1e-9', # This is what we consider an active pixel.
				variables = 'Amplitude (V)',
			)
			telescope_data = pandas.read_sql(
				f'SELECT n_event,n_track,CAEN_UZH_0_X,CAEN_UZH_0_Y FROM dataframe_table WHERE RD53B_114_associateHit==1', # HARDCODED! I have no time to do it better right now.
				con = sqlite3.connect(self.path_to_directory_of_task('convert_tracks_root_file_to_easy_SQLite')/'tracks.sqlite'),
			)
			telescope_data.set_index(['n_event','n_track'], inplace=True)
			telescope_data.rename(columns={col: f'{col[-1].lower()} (m)' for col in telescope_data.columns}, inplace=True)
			telescope_data /= 1e3 # Convert to meters.
			
			tracks_multiplicity = telescope_data[telescope_data.columns[0]].groupby('n_event').count()
			tracks_multiplicity.name = 'tracks_multiplicity'
			
			telescope_data = telescope_data.loc[tracks_multiplicity[tracks_multiplicity==1].index] # Drop track multiplicity > 1
			
			for df in [CAENs_detected_a_hit, telescope_data]:
				df.reset_index(inplace=True, drop=False)
				df.set_index(['n_event'], inplace=True)
			
			original_telescope_n_event = telescope_data.index
			
			plot_path = task.path_to_directory_of_my_task/'hits.html'
			print(f'⚠️  Please look at the plot in {plot_path}, if you can distinguish the pixels of the DUTs enter "yes", otherwise just press enter to continue to a different value of offset.')
			n = 0
			while True:
				for sign in [-1,1] if n != 0 else [1]:
					offset_shift = n*sign
					shifted_telescope_n_event = original_telescope_n_event + offset_shift
					hits = CAENs_detected_a_hit.join(telescope_data.set_index(shifted_telescope_n_event), how='inner')
					
					N_POINTS_TO_PLOT = 9999
					if len(hits) > N_POINTS_TO_PLOT:
						hits = hits.sample(n=N_POINTS_TO_PLOT)
					fig = px.scatter(
						title = f'Telescope and CAEN events alignment<br><sup>{self.pseudopath}, n_event_telescope += {offset_shift}</sup>',
						data_frame = hits.reset_index().sort_values('CAEN_n_channel').astype(dict(CAEN_n_channel=str)),
						x = 'x (m)',
						y = 'y (m)',
						color = 'CAEN_n_channel',
						hover_data = ['n_event'],
					)
					fig.write_html(
						plot_path,
						include_plotlyjs = 'cdn',
					)
					if input(f'Offset = {offset_shift}, are you able to distinguish the pixels in the plot? ').lower() == 'yes':
						with open(task.path_to_directory_of_my_task/'offset.json','w') as ofile:
							json.dump(offset_shift, ofile)
						logging.info(f'Offset was determined for {self.pseudopath} ✅')
						return
				n += 1
			raise RuntimeError(f'Could not find the offset...')
	
	def load_hits_on_DUT(self, DUT_name:str, max_chi2ndof:float=None, apply_EUDAQ_offset_to_fix_bug:bool=True):
		"""Load all tracks from one EUDAQ run, which have a hit in the CROC 
		(i.e. that have `RD53B_114_associateHit==1`).
		
		Arguments
		---------
		DUT_name: str
			The name of the DUT for which to load the hits.
		apply_EUDAQ_offset_to_fix_bug: bool, default True
			Normally you would not use this, and keep it `True`. This will
			require that you previously run the `find_EUDAQ_offset` task in
			this `EUDAQ_run_dn`.
		max_chi2ndof: float, default None
			If a value is passed, only tracks with a chi^2/n_degrees_of_freedom
			lower than this value are returned. Also, a new data column
			to the returned data frame named `chi2ndof` is added.
		
		Returns
		-------
		hits_on_DUT: pandas.DataFrame
			A data frame like this one:
			```
								x (m)     y (m)
			n_event n_track                    
			45      0       -0.002738  0.001134
			79      0       -0.002727 -0.002765
			92      0       -0.003044  0.001085
			128     0       -0.002434  0.000308
			132     0       -0.002659 -0.000086
			...                   ...       ...
			58193   0       -0.001589  0.000682
			58233   0       -0.001420  0.001082
			58273   0       -0.001806 -0.000630
			58277   0       -0.002335  0.000634
			58291   0       -0.000444  0.000030

			[3280 rows x 2 columns]
			```
		"""
		TB_batch_dn = self.parent
		
		setup_config = TBBatch.DatanodeHandlerTBBatch.load_setup_configuration_info(TB_batch_dn)
		if DUT_name not in set(setup_config['DUT_name']):
			raise ValueError(f'DUT_name {repr(DUT_name)} not found among the DUT names available in batch {repr(str(TB_batch_dn.pseudopath))}. DUT names available are {set(setup_config["DUT_name"])}. ')
		
		load_this = setup_config.query(f'DUT_name=="{DUT_name}"')[['plane_number','CAEN_name']].drop_duplicates()
		if len(load_this) > 1:
			raise NotImplementedError(f'I was expecting that only one `plane_number` and only one `CAEN_name` would be assigned to DUT_name {repr(DUT_name)}, but this does not seem to be the case in run {repr(str(self.pseudopath))}. ')
		plane_number_to_load = load_this['plane_number'].values[0] # We already checked that it has len() = 0 in the previous line
		CAEN_name_to_load = load_this['CAEN_name'].values[0] # We already checked that it has len() = 0 in the previous line
		
		# Here I will hardcode some things, I am sorry for doing this. I am finishing my PhD and have no time to investigate more.
		plane_number_to_load += -1 # This seems to be an offset introduced in the indices at some point. If I add this, the contours of the DUTs as given by the tracks look sharper.
		data = pandas.read_sql(
			f'SELECT n_event,n_track,CAEN_{CAEN_name_to_load}_{plane_number_to_load}_X,CAEN_{CAEN_name_to_load}_{plane_number_to_load}_Y FROM dataframe_table WHERE RD53B_114_associateHit==1',
			con = sqlite3.connect(self.path_to_directory_of_task('convert_tracks_root_file_to_easy_SQLite')/'tracks.sqlite'),
		)
		if apply_EUDAQ_offset_to_fix_bug == True:
			with open(self.path_to_directory_of_task('find_EUDAQ_offset')/'offset.json','r') as ifile:
				EUDAQ_bug_offset = json.load(ifile)
			data['n_event'] += EUDAQ_bug_offset # I found this just trying numbers until the correlation between the telescope data and the CAENs data made sense. This works for batch 8. A quick test with batch 7 showed that +3 worked instead.
		data.set_index(['n_event','n_track'], inplace=True)
		data.rename(columns={col: f'{col[-1].lower()} (m)' for col in data.columns}, inplace=True)
		data /= 1e3 # Convert to meters.
		
		tracks_multiplicity = data[data.columns[0]].groupby('n_event').count()
		tracks_multiplicity.name = 'tracks_multiplicity'
		data = data.loc[tracks_multiplicity[tracks_multiplicity==1].index] # Drop track multiplicity > 1
		
		if max_chi2ndof is not None:
			chi2_data = pandas.read_sql(
				f'SELECT n_event,n_track,chi2/ndof as chi2ndof FROM dataframe_table WHERE chi2ndof < {max_chi2ndof}',
				con = sqlite3.connect(self.path_to_directory_of_task('extract_tracks_parameters_from_ROOT_to_nice_peoples_format')/'tracks_data.sqlite'),
			).set_index(['n_event','n_track'])
			data = chi2_data.join(data, how='inner') # Keep only those tracks that satisfy the chi2 condition.
		
		return data

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
	parser.add_argument(
		'--find_EUDAQ_offset',
		help = 'Executes the task to find the EUDAQ offset and deal with the bug.',
		required = False,
		dest = 'find_EUDAQ_offset',
		action = 'store_true'
	)
	args = parser.parse_args()
	
	dn = DatanodeHandlerEUDAQRun(args.datanode)
	if args.find_EUDAQ_offset:
		dn.find_EUDAQ_offset()

