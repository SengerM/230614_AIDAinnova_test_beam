from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import numpy
import pandas
import utils_run_level
import logging

def load_setup_configuration_info(TB_batch:RunBureaucrat)->pandas.DataFrame:
	TB_batch.check_these_tasks_were_run_successfully(['runs','batch_info'])
	if not all([b.was_task_run_successfully('parse_waveforms') for b in TB_batch.list_subruns_of_task('runs')]):
		raise RuntimeError(f'To load the setup configuration it is needed that all of the runs of the batch have had the `parse_waveforms` task performed on them, but does not seem to be the case')
	
	match TB_batch.parent.run_name: # This is the test beam campaign, the parent of every batch.
		case '230614_June':
			planes = pandas.read_pickle(TB_batch.path_to_directory_of_task('batch_info')/'planes.pickle')
			signals_connections = pandas.read_pickle(TB_batch.path_to_directory_of_task('batch_info')/'signals.pickle')
		case '230830_August':
			planes = pandas.read_pickle(TB_batch.path_to_directory_of_task('batch_info')/'planes_definition.pickle')
			signals_connections = pandas.read_pickle(TB_batch.path_to_directory_of_task('batch_info')/'pixels_definition.pickle')
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
			raise RuntimeError(f'Cannot read setup information for run {TB_batch.run_name}')
	
	CAENs_names = []
	for run in TB_batch.list_subruns_of_task('runs'):
		n_run = int(run.run_name.split('_')[0].replace('run',''))
		df = pandas.read_pickle(run.path_to_directory_of_task('parse_waveforms')/f'{run.run_name}_CAENs_names.pickle')
		df = df.to_frame()
		df['n_run'] = n_run
		df.set_index('n_run',append=True,inplace=True)
		CAENs_names.append(df)
	CAENs_names = pandas.concat(CAENs_names)
	CAENs_names['CAEN_name'] = CAENs_names['CAEN_name'].apply(lambda x: x.replace('CAEN_',''))
	
	# Here we assume that the CAENs were not changed within a batch, which is reasonable.
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
		planes['DUT_name'],
		on = 'plane_number',
	)
	
	signals_connections['CAEN_trigger_group_n'] = signals_connections['CAEN_n_channel'].apply(lambda x: 0 if x in {0,1,2,3,4,5,6,7,16} else 1 if x in {8,9,10,11,12,13,14,15,17} else -1)
	
	signals_connections['rowcol'] = signals_connections[['row','col']].apply(lambda x: f'{x["row"]}{x["col"]}', axis=1)
	signals_connections['DUT_name_rowcol'] = signals_connections[['DUT_name','row','col']].apply(lambda x: f'{x["DUT_name"]} ({x["row"]},{x["col"]})', axis=1)
	
	return signals_connections

def load_parsed_from_waveforms(TB_batch:RunBureaucrat, load_this:dict, variables:list=None)->pandas.DataFrame:
	"""Load data parsed from the waveforms for all the runs within a batch.
	
	Arguments
	---------
	TB_batch: RunBureaucrat
		A bureaucrat pointing to the batch from which to load the data.
	load_this: dict
		A dictionary of the form
		```
		{
			DUT_name_rowcol: conditions,
		}
		```
		where `DUT_name_rowcol` is a string, e.g. `'TI123 (0,1)'` and 
		`conditions` is an SQL query with the cuts to apply to the different
		variables available, e.g.:
		```
		{
			'TI123 (0,1)': '`Amplitude (V)` < -5e-3 AND t_50 (s) > 50e-9',
			'TI222 (1,1)': '`Amplitude (V)` < -10e-3 AND t_50 (s) > 50e-9',
		}
		```
	variables: list of str
		A list of the variables to be loaded, e.g. `['Amplitude (V)','Collected charge (V s)']`.
	
	Returns
	-------
	parsed_from_waveforms: pandas.DataFrame
		A data frame of the form
		```
		                               Amplitude (V)  Collected charge (V s)
		n_run n_event DUT_name_rowcol                                       
		42    38      TI228 (0,0)          -0.005629           -4.103537e-12
			  49      TI228 (1,0)          -0.005816           -2.829203e-12
			  53      TI228 (1,0)          -0.070297           -1.066991e-10
			  66      TI228 (1,0)          -0.074181           -1.142252e-10
			  88      TI228 (0,0)          -0.005203           -2.491007e-12
		...                                      ...                     ...
		38    11695   TI228 (0,0)          -0.005421           -4.191143e-12
			  11697   TI228 (0,0)          -0.101138           -1.509368e-10
			  11703   TI228 (1,0)          -0.088648           -1.263468e-10
			  11732   TI228 (0,0)          -0.005097           -4.018176e-12
			  11782   TI228 (0,0)          -0.005678           -3.041788e-12

		[17854 rows x 2 columns]

		```
		
	"""
	TB_batch.check_these_tasks_were_run_successfully('runs')
	
	logging.info(f'Loading parsed from waveforms for {TB_batch.pseudopath} for {sorted(load_this)}...')
	
	setup_config = load_setup_configuration_info(TB_batch)
	
	SQL_query_where = []
	for DUT_name_rowcol, this_DUT_SQL_query in load_this.items():
		if DUT_name_rowcol not in set(setup_config['DUT_name_rowcol']):
			raise ValueError(f'Received DUT_name_rowcol {repr(DUT_name_rowcol)}, but only allowed possibilities for batch {TB_batch.pseudopath} are {sorted(set(setup_config["DUT_name_rowcol"]))}')
		setup_config_this_DUT = setup_config.query(f'DUT_name_rowcol == "{DUT_name_rowcol}"')
		if len(setup_config_this_DUT) != 1:
			raise ValueError('This should never have happened! Check! This means that there is more than one pixel with the same `DUT_name_rowcol` which is impossible.')
		setup_config_this_DUT = setup_config_this_DUT.iloc[0] # Convert to Series
		CAEN_n_channel = setup_config_this_DUT['CAEN_n_channel']
		n_CAEN = setup_config_this_DUT['n_CAEN']
		SQL_query_where.append(f'n_CAEN=={n_CAEN} AND CAEN_n_channel=={CAEN_n_channel} AND ({this_DUT_SQL_query})')
	SQL_query_where = ') or ('.join(SQL_query_where)
	SQL_query_where = f'({SQL_query_where})'
	
	parsed_from_waveforms = {}
	for run in TB_batch.list_subruns_of_task('runs'):
		n_run = int(run.run_name.split('_')[0].replace('run',''))
		parsed_from_waveforms[n_run] = utils_run_level.load_parsed_from_waveforms(
			TB_run = run,
			where = SQL_query_where,
			variables = variables,
		)
	parsed_from_waveforms = pandas.concat(parsed_from_waveforms, names=['n_run'])
	parsed_from_waveforms = parsed_from_waveforms.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])['DUT_name_rowcol'], on=['n_CAEN','CAEN_n_channel'])
	parsed_from_waveforms.reset_index(['n_CAEN','CAEN_n_channel'], drop=True, inplace=True)
	parsed_from_waveforms.set_index('DUT_name_rowcol', append=True, inplace=True)
	return parsed_from_waveforms

def load_hits(TB_batch:RunBureaucrat, DUTs_and_hit_criterions:dict)->pandas.DataFrame:
	"""Load the hits on a set of DUTs according to some criterions.
	
	Arguments
	---------
	TB_batch: RunBureaucrat
		A bureaucrat pointing to the batch from which to load the data.
	DUTs_and_hit_criterions: dict
		A dictionary of the form
		```
		{
			DUT_name_rowcol: conditions,
		}
		```
		where `DUT_name_rowcol` is a string, e.g. `'TI123 (0,1)'` and 
		`conditions` is an SQL query with the cuts to apply to the different
		variables available, which define what is considered as a hit, e.g.:
		```
		{
			'TI123 (0,1)': '`Amplitude (V)` < -5e-3 AND t_50 (s) > 50e-9',
			'TI222 (1,1)': '`Amplitude (V)` < -10e-3 AND t_50 (s) > 50e-9',
		}
		```
	
	Returns
	-------
	hits: pandas.DataFrame
		A data frame of the form
		```
		DUT_name_rowcol  TI228 (0,0)  TI228 (1,0)
		n_run n_event                            
		38    9                 True        False
			  52                True        False
			  134               True        False
			  158              False         True
			  290              False         True
		...                      ...          ...
		44    127353            True        False
			  127380           False         True
			  127390            True        False
			  127394            True        False
			  127402            True        False

		[17339 rows x 2 columns]
		```
		where `True` or `False` denote whether there was a hit or not,
		according to the criterion specified in `DUTs_and_hit_criterions`
		for each DUT.
	"""
	TB_batch.check_these_tasks_were_run_successfully('runs')
	
	logging.info(f'Loading hits from {TB_batch.pseudopath} for {sorted(DUTs_and_hit_criterions)}...')
	
	hits = load_parsed_from_waveforms(
		TB_batch = TB_batch,
		load_this = DUTs_and_hit_criterions,
		variables = None, # We don't need no variables.
	)
	hits['has_hit'] = True
	hits = hits.unstack('DUT_name_rowcol', fill_value=False)
	hits = hits['has_hit'] # Drop unnecessary level.
	return hits

def load_tracks(TB_batch:RunBureaucrat, only_multiplicity_one:bool=False, require_coincidence_with_DUTs:list=None):
	"""Loads the tracks reconstructed by `corry_reconstruct_tracks_with_telescope`
	from all the runs within a TB_batch.
	
	Arguments
	---------
	TB_batch: RunBureaucrat
		A bureaucrat pointing to a run.
	only_multiplicity_one: bool, default False
		If `True`, only tracks whose event has track multiplicity 1 will
		be loaded.
	require_coincidence_with_DUTs: list of dict, default None
		If `None`, this argument is ignored. Else, it has to be a list
		of dictionaries, each of the form
		```
		{
			'DUT_name': 'AC20',
			'row': 0,
			'col': 1,
			'hit_criterion': "100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<-5e-3",
		}
		```
		specifying additional pixels from the DUTs to be required to have
		a coincidence with the tracks.
	"""
	TB_batch.check_these_tasks_were_run_successfully('runs')
	
	logging.info(f'Reading tracks from {TB_batch.pseudopath}...')
	
	tracks = []
	for run in TB_batch.list_subruns_of_task('runs'):
		df = utils_run_level.load_tracks_from_run(
			run = run,
			only_multiplicity_one = only_multiplicity_one,
		)
		run_number = int(run.run_name.split('_')[0].replace('run',''))
		df = pandas.concat({run_number: df}, names=['n_run'])
		tracks.append(df)
	tracks = pandas.concat(tracks)
	
	if require_coincidence_with_DUTs is not None:
		TB_batch.check_these_tasks_were_run_successfully()
	
	return tracks

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
	
	hits = load_hits(
		TB_batch = RunBureaucrat('/media/msenger/230829_gray/AIDAinnova_test_beams/TB/campaigns/subruns/230830_August/batches/subruns/batch_1'),
		DUTs_and_hit_criterions = {
			'TI228 (0,0)': '`Amplitude (V)` < -5e-3 AND `t_50 (s)`>50e-9',
			'TI228 (1,0)': '`Amplitude (V)` < -5e-3 AND `t_50 (s)`>30e-9',
		},
	)
	print(hits)

# ~ def load_tracks(RSD_analysis:RunBureaucrat, DUT_z_position:float, use_DUTs_as_trigger:dict=None):
	# ~ """
	# ~ Arguments
	# ~ ---------
	# ~ RSD_analysis: RunBureaucrat
		# ~ A bureaucrat pointing to the respective RSD analysis for which to
		# ~ load the tracks.
	# ~ DUT_z_position: float
		# ~ The z coordinate of the DUT to project the tracks to its plane.
	# ~ use_DUTs_as_trigger: dict, dafault None
		# ~ A dictionary where the keys are the names of other DUTs that are 
		# ~ present in the same batch (i.e. in `RSD_analysis.parent`) that are added to the triggering
		# ~ by requesting a coincidence between the tracks and them, e.g. `"AC 11 (0,1)"`.
		# ~ The item following each key is the SQL query with the "DUT_hit_criterion",
		# ~ e.g. ``,
	# ~ """
	# ~ RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	# ~ batch = RSD_analysis.parent
	
	# ~ tracks = load_tracks_from_batch(batch, only_multiplicity_one=True)
	# ~ tracks = tracks.join(project_tracks(tracks=tracks, z_position=DUT_z_position))
	
	# ~ if use_DUTs_as_trigger is not None:
		# ~ use_DUTs_as_trigger_names = set([_.split(' (')[0] for _ in use_DUTs_as_trigger.keys()])
		# ~ trigger_DUTs_hits = []
		# ~ for DUT_name in use_DUTs_as_trigger_names:
			# ~ this_DUT_hits = load_hits(
				# ~ RSD_analysis = RunBureaucrat(RSD_analysis.path_to_run_directory.parent/DUT_name),
				# ~ DUT_hit_criterion = , # For the moment this is hardcoded.
			# ~ )
			# ~ trigger_DUTs_hits.append(this_DUT_hits)
		# ~ trigger_DUTs_hits = pandas.concat(
			# ~ trigger_DUTs_hits,
			# ~ join = 'outer',
			# ~ axis = 'columns',
		# ~ )
		# ~ trigger_DUTs_hits = trigger_DUTs_hits[use_DUTs_as_trigger]
		# ~ trigger_DUTs_hits['trigger_on_this_event'] = trigger_DUTs_hits.sum(axis=1)>0
		# ~ tracks = utils.select_by_multiindex(
			# ~ df = tracks,
			# ~ idx = trigger_DUTs_hits.query('trigger_on_this_event == True').index,
		# ~ )
	
	# ~ if max_chi2dof is not None:
		# ~ tracks = tracks.query(f'`chi2/ndof`<{max_chi2dof}')
		
	# ~ return tracks
