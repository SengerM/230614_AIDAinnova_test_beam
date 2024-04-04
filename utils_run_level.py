from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import numpy
import pandas
import sqlite3
import logging
from progressreporting.TelegramProgressReporter import SafeTelegramReporter4Loops # https://github.com/SengerM/progressreporting
from contextlib import nullcontext
import parse_waveforms
import corry_stuff
import utils_batch_level
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

def find_EUDAQ_offset(EUDAQ_run_dn:DatanodeHandler):
	with EUDAQ_run_dn.handle_task(
		'find_EUDAQ_offset', 
		check_datanode_class = 'EUDAQ_run', 
		check_required_tasks = ['parse_waveforms','convert_tracks_root_file_to_easy_SQLite'],
	) as task:
		setup_config = utils_batch_level.load_setup_configuration_info(EUDAQ_run_dn.parent)
		CAENs_detected_a_hit = parse_waveforms.load_parsed_from_waveforms_from_EUDAQ_run(
			EUDAQ_run_dn = EUDAQ_run_dn, 
			where = '`Amplitude (V)` < -.01 AND `Time over 50% (s)` > 1e-9', # This is what we consider an active pixel.
			variables = 'Amplitude (V)',
		)
		telescope_data = pandas.read_sql(
			f'SELECT n_event,n_track,CAEN_UZH_0_X,CAEN_UZH_0_Y FROM dataframe_table WHERE RD53B_114_associateHit==1', # HARDCODED! I have no time to do it better right now.
			con = sqlite3.connect(EUDAQ_run_dn.path_to_directory_of_task('convert_tracks_root_file_to_easy_SQLite')/'tracks.sqlite'),
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
					title = f'Telescope and CAEN events alignment<br><sup>{EUDAQ_run_dn.pseudopath}, n_event_telescope += {offset_shift}</sup>',
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
					logging.info(f'Offset was determined for {EUDAQ_run_dn.pseudopath} ✅')
					return
			n += 1
		raise RuntimeError(f'Could not find the offset...')
