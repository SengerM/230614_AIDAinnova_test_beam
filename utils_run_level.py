from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import numpy
import pandas
import sqlite3
import logging
from progressreporting.TelegramProgressReporter import SafeTelegramReporter4Loops # https://github.com/SengerM/progressreporting
from contextlib import nullcontext

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

# ~ def load_tracks(TB_run:RunBureaucrat, only_multiplicity_one:bool=True):
	# ~ """Loads the tracks reconstructed by `corry_reconstruct_tracks_with_telescope`.
	
	# ~ Arguments
	# ~ ---------
	# ~ TB_run: RunBureaucrat
		# ~ A bureaucrat pointing to a test beam run.
	# ~ only_multiplicity_one: bool, default False
		# ~ If `True`, only tracks whose event has track multiplicity 1 will
		# ~ be loaded.
	# ~ """
	# ~ TB_run.check_these_tasks_were_run_successfully(['raw','corry_reconstruct_tracks_with_telescope'])
	
	# ~ SQL_query = 'SELECT * FROM dataframe_table'
	# ~ if only_multiplicity_one == True:
		# ~ SQL_query += ' GROUP BY n_event HAVING COUNT(n_track) = 1'
	# ~ tracks = pandas.read_sql(
		# ~ SQL_query,
		# ~ con = sqlite3.connect(TB_run.path_to_directory_of_task('corry_reconstruct_tracks_with_telescope')/'tracks.sqlite'),
	# ~ )
	
	# ~ tracks['n_event'] = tracks['n_event'] - 1 # Fix an offset that is present in the data, I think it has to do with the extra trigger sent by the TLU when the run starts, that was not sent to the CAENs.
	# ~ tracks[['Ax','Ay','Az','Bx','By']] *= 1e-3 # Convert millimeter to meter, it is more natural to work in SI units.
	# ~ tracks['chi2/ndof'] = tracks['chi2']/tracks['ndof']
	
	# ~ tracks.set_index(['n_event','n_track'], inplace=True)
	
	# ~ if only_multiplicity_one == True:
		# ~ # Check that the track multiplicity is indeed 1 for all events loaded:
		# ~ n_tracks_in_event = tracks['is_fitted'].groupby(['n_event']).count()
		# ~ n_tracks_in_event.name = 'n_tracks_in_event'
		# ~ if set(n_tracks_in_event) != {1} or len(tracks) == 0:
			# ~ raise RuntimeError(f'Failed to load tracks only from events with track multiplicity 1...')
	
	# ~ return tracks
