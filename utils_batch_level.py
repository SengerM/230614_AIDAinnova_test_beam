from pathlib import Path
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import numpy
import pandas
import utils
import logging
import plotly.graph_objects as go
import plotly.express as px
import dominate # https://github.com/Knio/dominate

def setup_batch_info(TB_batch_dn:DatanodeHandler):
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
	
	TB_campaign = TB_batch_dn.parent # The parent of the batch should be the TB campaign.
	if TB_campaign is None or TB_campaign.datanode_class != 'TB_campaign':
		raise RuntimeError(f'I was expecting that the parent of batch "{TB_batch_dn.pseudopath}" is a test beam campaign, but it is not. ')
	
	match TB_campaign.datanode_name:
		case '230614_June':
			setup_batch_info_June_test_beam(TB_batch_dn)
		case '230830_August':
			setup_batch_info_August_test_beam(TB_batch_dn)
		case '240212_DESY':
			setup_batch_info_240212_DESY_test_beam(TB_batch_dn)
		case _:
			raise RuntimeError(f'Cannot determine which test beam campaign {TB_batch_dn.pseudopath} belongs to...')
	logging.info(f'Setup info was set for batch "{TB_batch_dn.pseudopath}" ✅')

def load_setup_configuration_info(TB_batch_dn:DatanodeHandler)->pandas.DataFrame:
	TB_batch_dn.check_datanode_class('TB_batch')
	
	# Check that all the tasks we need were properly executed already:
	TB_batch_dn.check_these_tasks_were_run_successfully(['EUDAQ_runs','batch_info'])
	if not all([dn.was_task_run_successfully('parse_waveforms') for dn in TB_batch_dn.list_subdatanodes_of_task('EUDAQ_runs')]):
		raise RuntimeError(f'To load the setup configuration it is needed that all of the runs of the batch have had the `parse_waveforms` task performed on them, but does not seem to be the case')
	
	TB_campaign = TB_batch_dn.parent # The parent of the batch should be the TB campaign.
	if TB_campaign is None or TB_campaign.datanode_class != 'TB_campaign':
		raise RuntimeError(f'I was expecting that the parent of batch "{TB_batch_dn.pseudopath}" is a test beam campaign, but it is not. ')
	
	match TB_campaign.datanode_name: # This is the test beam campaign, the parent of every batch.
		case '230614_June':
			planes = pandas.read_pickle(TB_batch_dn.path_to_directory_of_task('batch_info')/'planes.pickle')
			signals_connections = pandas.read_pickle(TB_batch_dn.path_to_directory_of_task('batch_info')/'signals.pickle')
		case '230830_August':
			planes = pandas.read_pickle(TB_batch_dn.path_to_directory_of_task('batch_info')/'planes_definition.pickle')
			signals_connections = pandas.read_pickle(TB_batch_dn.path_to_directory_of_task('batch_info')/'pixels_definition.pickle')
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
			planes = pandas.read_pickle(TB_batch_dn.path_to_directory_of_task('batch_info')/'planes_definition.pickle')
			signals_connections = pandas.read_pickle(TB_batch_dn.path_to_directory_of_task('batch_info')/'pixels_definition.pickle')
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
			raise RuntimeError(f'Cannot read setup information for batch "{TB_batch_dn.datanode_name}". ')
	
	CAENs_names = []
	for run_dn in TB_batch_dn.list_subdatanodes_of_task('EUDAQ_runs'):
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

# ~ def load_hits(TB_batch:RunBureaucrat, DUTs_and_hit_criterions:dict)->pandas.DataFrame:
	# ~ """Load the hits on a set of DUTs according to some criterions.
	
	# ~ Arguments
	# ~ ---------
	# ~ TB_batch: RunBureaucrat
		# ~ A bureaucrat pointing to the batch from which to load the data.
	# ~ DUTs_and_hit_criterions: dict
		# ~ A dictionary of the form
		# ~ ```
		# ~ {
			# ~ DUT_name_rowcol: conditions,
		# ~ }
		# ~ ```
		# ~ where `DUT_name_rowcol` is a string, e.g. `'TI123 (0,1)'` and 
		# ~ `conditions` is an SQL query with the cuts to apply to the different
		# ~ variables available, which define what is considered as a hit, e.g.:
		# ~ ```
		# ~ {
			# ~ 'TI123 (0,1)': '`Amplitude (V)` < -5e-3 AND t_50 (s) > 50e-9',
			# ~ 'TI222 (1,1)': '`Amplitude (V)` < -10e-3 AND t_50 (s) > 50e-9',
		# ~ }
		# ~ ```
	
	# ~ Returns
	# ~ -------
	# ~ hits: pandas.DataFrame
		# ~ A data frame of the form
		# ~ ```
		# ~ DUT_name_rowcol  TI228 (0,0)  TI228 (1,0)
		# ~ n_run n_event                            
		# ~ 38    9                 True        False
			  # ~ 52                True        False
			  # ~ 134               True        False
			  # ~ 158              False         True
			  # ~ 290              False         True
		# ~ ...                      ...          ...
		# ~ 44    127353            True        False
			  # ~ 127380           False         True
			  # ~ 127390            True        False
			  # ~ 127394            True        False
			  # ~ 127402            True        False

		# ~ [17339 rows x 2 columns]
		# ~ ```
		# ~ where `True` or `False` denote whether there was a hit or not,
		# ~ according to the criterion specified in `DUTs_and_hit_criterions`
		# ~ for each DUT.
	# ~ """
	# ~ TB_batch.check_these_tasks_were_run_successfully('EUDAQ_runs')
	
	# ~ logging.info(f'Loading hits from {TB_batch.pseudopath} for {sorted(DUTs_and_hit_criterions)}...')
	
	# ~ hits = load_parsed_from_waveforms(
		# ~ TB_batch = TB_batch,
		# ~ load_this = DUTs_and_hit_criterions,
		# ~ variables = None, # We don't need no variables.
	# ~ )
	# ~ if len(hits) == 0: # No hits at all...
		# ~ return hits
	# ~ hits['has_hit'] = True
	# ~ hits = hits.unstack('DUT_name_rowcol', fill_value=False)
	# ~ hits = hits['has_hit'] # Drop unnecessary level.
	# ~ return hits

# ~ def load_tracks(TB_batch:RunBureaucrat, only_multiplicity_one:bool=True, trigger_on_DUTs:dict=None)->pandas.DataFrame:
	# ~ """Loads the tracks reconstructed by `corry_reconstruct_tracks_with_telescope`
	# ~ from all the runs within a TB_batch.
	
	# ~ Arguments
	# ~ ---------
	# ~ TB_batch: RunBureaucrat
		# ~ A bureaucrat pointing to a run.
	# ~ only_multiplicity_one: bool, default False
		# ~ If `True`, only tracks whose event has track multiplicity 1 will
		# ~ be loaded.
	# ~ trigger_on_DUTs: list of dict, default None
		# ~ If `None`, this argument is ignored. Else, a dictionary of the form
		# ~ ```
		# ~ {
			# ~ DUT_name_rowcol: conditions,
		# ~ }
		# ~ ```
		# ~ where `DUT_name_rowcol` is a string, e.g. `'TI123 (0,1)'` and 
		# ~ `conditions` is an SQL query with the cuts to apply to the different
		# ~ variables available, e.g.:
		# ~ ```
		# ~ {
			# ~ 'TI123 (0,1)': '`Amplitude (V)` < -5e-3 AND t_50 (s) > 50e-9',
			# ~ 'TI222 (1,1)': '`Amplitude (V)` < -10e-3 AND t_50 (s) > 50e-9',
		# ~ }
		# ~ ```
		# ~ specifying additional pixels from the DUTs to be required to have
		# ~ a coincidence with the tracks.
		# ~ Each of these DUTs is added as an `or` to the trigger line, i.e.
		# ~ if any of the DUTs in `trigger_on_DUTs` has a hit for some event,
		# ~ then this event is not discarded. If neither of the DUTs in `trigger_on_DUTs`
		# ~ has a hit, then the event is discarded.
	
	# ~ Returns
	# ~ -------
	# ~ tracks: pandas.DataFrame
		# ~ A data frame of the form
		# ~ ```
		                       # ~ is_fitted        chi2  ndof        Ax        Ay     Az        Bx        By    Bz  chi2/ndof
		# ~ n_run n_event n_track                                                                                             
		# ~ 42    -1      0                1    7.561065     8 -0.006860  0.000802  0.099 -0.006874  0.000815 -99.0   0.945133
			   # ~ 0      0                1  210.956625     8 -0.007270 -0.001034  0.099 -0.007262 -0.001069 -99.0  26.369578
			   # ~ 1      0                1    7.138928     8 -0.008476  0.000838  0.099 -0.008475  0.000855 -99.0   0.892366
			   # ~ 2      0                1   11.985300     8 -0.008407  0.000644  0.099 -0.008429  0.000664 -99.0   1.498162
			   # ~ 3      0                1    3.517900     8 -0.008494 -0.000040  0.099 -0.008526 -0.000042 -99.0   0.439737
		# ~ ...                          ...         ...   ...       ...       ...    ...       ...       ...   ...        ...
		# ~ 38     11855  0                1    4.209887     8 -0.008299 -0.001134  0.099 -0.008310 -0.001114 -99.0   0.526236
			   # ~ 11857  0                1    5.188543     8 -0.008874  0.000089  0.099 -0.008891  0.000105 -99.0   0.648568
			   # ~ 11858  0                1    3.497484     8 -0.008743 -0.001127  0.099 -0.008735 -0.001123 -99.0   0.437186
			   # ~ 11859  0                1    5.961973     8 -0.006770  0.000023  0.099 -0.006795  0.000027 -99.0   0.745247
			   # ~ 11864  0                1   17.142053     8 -0.008584 -0.001071  0.099 -0.008601 -0.001009 -99.0   2.142757

		# ~ [351935 rows x 10 columns]

		# ~ ```
	# ~ """
	# ~ TB_batch.check_these_tasks_were_run_successfully('EUDAQ_runs')
	
	# ~ logging.info(f'Reading tracks from {TB_batch.pseudopath}...')
	
	# ~ tracks = []
	# ~ for run in TB_batch.list_subruns_of_task('EUDAQ_runs'):
		# ~ df = utils_run_level.load_tracks(
			# ~ TB_run = run,
			# ~ only_multiplicity_one = only_multiplicity_one,
		# ~ )
		# ~ run_number = int(run.run_name.split('_')[0].replace('run',''))
		# ~ df = pandas.concat({run_number: df}, names=['n_run'])
		# ~ tracks.append(df)
	# ~ tracks = pandas.concat(tracks)
	
	# ~ if trigger_on_DUTs is not None:
		# ~ hits_on_trigger_DUTs = load_hits(
			# ~ TB_batch = TB_batch,
			# ~ DUTs_and_hit_criterions = trigger_on_DUTs,
		# ~ )
		# ~ tracks = utils.select_by_multiindex(tracks, hits_on_trigger_DUTs.index)
	# ~ return tracks

# ~ def plot_DUTs_hits(TB_batch:RunBureaucrat):
	# ~ """Produces a plot with a few tracks from each DUT that should have
	# ~ some hits, and puts them all together, making it easy to see which
	# ~ DUTs overlap in space and where they are located."""
	# ~ TB_batch.check_these_tasks_were_run_successfully('EUDAQ_runs')
	
	# ~ with TB_batch.handle_task('plot_DUTs_hits') as employee:
	
		# ~ tracks = load_tracks(
			# ~ TB_batch = TB_batch,
			# ~ only_multiplicity_one = True,
		# ~ )
		
		# ~ setup_config = load_setup_configuration_info(TB_batch)
		# ~ setup_config = setup_config.query('row>=0 and col>=-1') # Negative rows and cols denote fictitious devices, like the trigger line for the CAENs.
		# ~ setup_config = setup_config.sort_values('DUT_name_rowcol')
		
		# ~ fig = go.Figure()
		
		# ~ some_random_tracks = tracks.sample(999) if len(tracks)>999 else tracks
		# ~ fig.add_trace(
			# ~ go.Scatter(
				# ~ x = some_random_tracks['Ax'],
				# ~ y = some_random_tracks['Ay'],
				# ~ name = 'Random tracks',
				# ~ mode = 'markers',
			# ~ )
		# ~ )
		
		# ~ for DUT_name, this_DUT_config in setup_config.groupby('DUT_name'):
			# ~ DUT_hits = load_hits(
				# ~ TB_batch = TB_batch,
				# ~ DUTs_and_hit_criterions = {_:f'`SNR`>33  AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9' for _ in this_DUT_config['DUT_name_rowcol']},
			# ~ )
			# ~ DUT_hits = DUT_hits.sample(999) if len(DUT_hits)>999 else DUT_hits
			
			# ~ fig.add_trace(
				# ~ go.Scatter(
					# ~ x = utils.select_by_multiindex(tracks, DUT_hits.index)['Ax'] if len(DUT_hits)>0 else [float('NaN')],
					# ~ y = utils.select_by_multiindex(tracks, DUT_hits.index)['Ay'] if len(DUT_hits)>0 else [float('NaN')],
					# ~ name = DUT_name,
					# ~ mode = 'markers',
				# ~ )
			# ~ )
		
		# ~ fig.update_layout(
			# ~ title = f'DUTs hits preview<br><sup>{TB_batch.pseudopath}</sup>',
			# ~ xaxis_title = 'Ax (m)',
			# ~ yaxis_title = 'Ay (m)',
		# ~ )
		# ~ fig.update_yaxes(
			# ~ scaleanchor = "x",
			# ~ scaleratio = 1,
		# ~ )
		# ~ fig.write_html(
			# ~ employee.path_to_directory_of_my_task/f'DUTs_hits.html',
			# ~ include_plotlyjs = 'cdn',
		# ~ )

# ~ def plot_DUT_distributions(TB_batch:RunBureaucrat, max_events_to_plot:int=9999, distributions:bool=True, scatter_plots:bool=True):
	# ~ TB_batch.check_these_tasks_were_run_successfully(['EUDAQ_runs','batch_info'])
	
	# ~ with TB_batch.handle_task('plot_DUT_distributions') as employee:
		# ~ setup_config = load_setup_configuration_info(TB_batch)
		# ~ setup_config = setup_config.query('row>-1 and col>-1') # Negative row col means a fake device, such as a trigger line or so.
		# ~ setup_config = setup_config.sort_values('DUT_name_rowcol')
		
		# ~ VARIABLES_TO_PLOT_DISTRIBUTION = {'Amplitude (V)','Collected charge (V s)','t_50 (s)','Time over 50% (s)','Noise (V)','SNR'}
		# ~ if distributions:
			# ~ save_1D_distributions_here = employee.path_to_directory_of_my_task/'distributions'
			# ~ save_1D_distributions_here.mkdir()
			# ~ for DUT_name,DUT_config in setup_config.groupby('DUT_name'):
				# ~ DUT_config = DUT_config.set_index(['n_CAEN','CAEN_n_channel'])
				# ~ save_plots_here = save_1D_distributions_here/DUT_name
				# ~ save_plots_here.mkdir(parents=True)
				# ~ for variable in VARIABLES_TO_PLOT_DISTRIBUTION:
					# ~ logging.info(f'Producing distribution plot for {variable} for DUT {DUT_name} in {TB_batch.pseudopath}...')
					# ~ data = load_parsed_from_waveforms(
						# ~ TB_batch = TB_batch,
						# ~ load_this = {_:None for _ in DUT_config['DUT_name_rowcol']},
						# ~ variables = [variable],
					# ~ )
					# ~ data = data.unstack('DUT_name_rowcol')
					# ~ if len(data) > max_events_to_plot:
						# ~ data = data.sample(max_events_to_plot)
					# ~ data = data.stack('DUT_name_rowcol')
					
					# ~ fig = px.ecdf(
						# ~ data_frame = data.reset_index(drop=False).sort_values('DUT_name_rowcol'),
						# ~ x = variable,
						# ~ color = 'DUT_name_rowcol',
						# ~ title = f'{variable} for {DUT_name}<br><sup>{TB_batch.pseudopath}</sup>',
						# ~ marginal = 'histogram',
					# ~ )
					# ~ fig.write_html(
						# ~ save_plots_here/f'{variable}.html',
						# ~ include_plotlyjs = 'cdn',
					# ~ )
			# ~ for variable in VARIABLES_TO_PLOT_DISTRIBUTION:
				# ~ document_title = f'{variable} distribution for all DUTs in {TB_batch.pseudopath}'
				# ~ html_doc = dominate.document(title=document_title)
				# ~ with html_doc:
					# ~ dominate.tags.h1(document_title)
					# ~ for DUT_name in sorted(setup_config['DUT_name'].drop_duplicates()):
						# ~ dominate.tags.iframe(
							# ~ src = f'distributions/{DUT_name}/{variable}.html',
							# ~ style = f'height: 88vh; width: 100%; border-style: solid;',
						# ~ )
				# ~ with open(employee.path_to_directory_of_my_task/f'{variable}.html', 'w') as ofile:
					# ~ print(html_doc, file=ofile)
		
		# ~ PAIRS_OF_VARIABLES_FOR_SCATTER_PLOTS = {('t_50 (s)','Amplitude (V)'),('t_50 (s)','Collected charge (V s)'),('Time over 50% (s)','Amplitude (V)'),('Time over 50% (s)','Collected charge (V s)')}
		# ~ if scatter_plots:
			# ~ save_2D_scatter_plots_here = employee.path_to_directory_of_my_task/'scatter_plots'
			# ~ save_2D_scatter_plots_here.mkdir()
			# ~ for DUT_name,DUT_config in setup_config.groupby('DUT_name'):
				# ~ DUT_config = DUT_config.set_index(['n_CAEN','CAEN_n_channel'])
				# ~ save_plots_here = save_2D_scatter_plots_here/DUT_name
				# ~ save_plots_here.mkdir(parents=True)
				# ~ for x,y in PAIRS_OF_VARIABLES_FOR_SCATTER_PLOTS:
					# ~ logging.info(f'Producing 2D scatter plot for {y} vs {x} for DUT {DUT_name} in {TB_batch.pseudopath}...')
					# ~ data = load_parsed_from_waveforms(
						# ~ TB_batch = TB_batch,
						# ~ load_this = {_:None for _ in DUT_config['DUT_name_rowcol']},
						# ~ variables = [x,y],
					# ~ )
					# ~ data = data.unstack('DUT_name_rowcol')
					# ~ if len(data) > max_events_to_plot:
						# ~ data = data.sample(max_events_to_plot)
					# ~ data = data.stack('DUT_name_rowcol')
					
					# ~ fig = px.scatter(
						# ~ data_frame = data.reset_index(drop=False).sort_values('DUT_name_rowcol'),
						# ~ x = x,
						# ~ y = y,
						# ~ color = 'DUT_name_rowcol',
						# ~ title = f'{y} vs {x} for {DUT_name}<br><sup>{TB_batch.pseudopath}</sup>',
						# ~ hover_data = data.index.names,
					# ~ )
					# ~ fig.write_html(
						# ~ save_plots_here/f'{y} vs {x}.html',
						# ~ include_plotlyjs = 'cdn',
					# ~ )
			# ~ for x,y in PAIRS_OF_VARIABLES_FOR_SCATTER_PLOTS:
				# ~ document_title = f'{y} vs {x} scatter preview for all DUTs in {TB_batch.pseudopath}'
				# ~ html_doc = dominate.document(title=document_title)
				# ~ with html_doc:
					# ~ dominate.tags.h1(document_title)
					# ~ for DUT_name in sorted(setup_config['DUT_name'].drop_duplicates()):
						# ~ dominate.tags.iframe(
							# ~ src = f'scatter_plots/{DUT_name}/{y} vs {x}.html',
							# ~ style = f'height: 88vh; width: 100%; border-style: solid;',
						# ~ )
				# ~ with open(employee.path_to_directory_of_my_task/f'{y} vs {x}.html', 'w') as ofile:
					# ~ print(html_doc, file=ofile)

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
	parser.add_argument(
		'--plot_DUT_distributions',
		help = 'If this flag is passed, it will execute `plot_DUT_distributions`.',
		required = False,
		dest = 'plot_DUT_distributions',
		action = 'store_true'
	)
	parser.add_argument(
		'--plot_DUTs_hits',
		help = 'If this flag is passed, it will execute `plot_DUTs_hits`.',
		required = False,
		dest = 'plot_DUTs_hits',
		action = 'store_true'
	)
	args = parser.parse_args()
	
	dn = DatanodeHandler(args.datanode)
	if args.setup_batch_info:
		setup_batch_info(dn)
	if args.plot_DUT_distributions:
		plot_DUT_distributions(dn, max_events_to_plot=11111)
	if args.plot_DUTs_hits:
		plot_DUTs_hits(dn)
