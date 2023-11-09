from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import numpy
import pandas
import sqlite3
import logging

def load_tracks(TB_run:RunBureaucrat, only_multiplicity_one:bool=True):
	"""Loads the tracks reconstructed by `corry_reconstruct_tracks_with_telescope`.
	
	Arguments
	---------
	TB_run: RunBureaucrat
		A bureaucrat pointing to a test beam run.
	only_multiplicity_one: bool, default False
		If `True`, only tracks whose event has track multiplicity 1 will
		be loaded.
	"""
	TB_run.check_these_tasks_were_run_successfully(['raw','corry_reconstruct_tracks_with_telescope'])
	
	SQL_query = 'SELECT * FROM dataframe_table'
	if only_multiplicity_one == True:
		SQL_query += ' GROUP BY n_event HAVING COUNT(n_track) = 1'
	tracks = pandas.read_sql(
		SQL_query,
		con = sqlite3.connect(TB_run.path_to_directory_of_task('corry_reconstruct_tracks_with_telescope')/'tracks.sqlite'),
	)
	
	tracks['n_event'] = tracks['n_event'] - 1 # Fix an offset that is present in the data, I think it has to do with the extra trigger sent by the TLU when the run starts, that was not sent to the CAENs.
	tracks[['Ax','Ay','Az','Bx','By']] *= 1e-3 # Convert millimeter to meter, it is more natural to work in SI units.
	tracks['chi2/ndof'] = tracks['chi2']/tracks['ndof']
	
	tracks.set_index(['n_event','n_track'], inplace=True)
	
	if only_multiplicity_one == True:
		# Check that the track multiplicity is indeed 1 for all events loaded:
		n_tracks_in_event = tracks['is_fitted'].groupby(['n_event']).count()
		n_tracks_in_event.name = 'n_tracks_in_event'
		if set(n_tracks_in_event) != {1} or len(tracks) == 0:
			raise RuntimeError(f'Failed to load tracks only from events with track multiplicity 1...')
	
	return tracks

def load_parsed_from_waveforms(TB_run:RunBureaucrat, where:str, variables:list=None):
	TB_run.check_these_tasks_were_run_successfully(['raw','parse_waveforms'])
	
	logging.info(f'Reading parsed from waveforms from run {TB_run.pseudopath}...')
	
	if variables is not None:
		variables = ',' + ','.join([f'`{_}`' for _ in variables])
	else:
		variables = ''
	data = pandas.read_sql(
		f'SELECT n_event,n_CAEN,CAEN_n_channel{variables} FROM dataframe_table WHERE {where}',
		con = sqlite3.connect(TB_run.path_to_directory_of_task('parse_waveforms')/f'{TB_run.run_name}.sqlite'),
	)
	data.set_index(['n_event','n_CAEN','CAEN_n_channel'], inplace=True)
	return data

