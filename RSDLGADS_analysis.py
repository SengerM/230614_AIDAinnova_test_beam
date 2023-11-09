from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import utils
import logging
import pandas
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy
import plotly_utils
from corry_stuff import load_tracks_from_batch
import warnings
from parse_waveforms import read_parsed_from_waveforms_from_batch
from TILGADs_analysis import translate_and_then_rotate
import sys
sys.path.append('/home/msenger/code/AC-LGAD_scripts') # I am expecting to find this in here https://github.com/SengerM/AC-LGAD_scripts
import reconstructors # https://github.com/SengerM/AC-LGAD_scripts
import matplotlib.pyplot as plt
import json
import pickle

def load_RSD_analyses_config():
	logging.info(f'Reading analyses config from the cloud...')
	analyses = pandas.read_csv(
		'https://docs.google.com/spreadsheets/d/e/2PACX-1vTaR20eM5ZQxtizmZiaAtHooE7hWYfSixSgc1HD5sVNZT_RNxZKmhI09wCEtXEVepjM8NB1n8BUBZnc/pub?gid=1826054435&single=true&output=csv',
		index_col = ['test_beam_campaign','batch_name','DUT_name'],
	)
	analyses = analyses.query('DUT_type=="RSD-LGAD"')
	analyses.to_csv('../TB/RSD-LGAD_analyses_config.backup.csv')
	return analyses

def load_this_RSD_analysis_config(RSD_analysis:RunBureaucrat):
	TB_batch = RSD_analysis.parent
	TB_campaign = TB_batch.parent
	analysis_config = load_RSD_analyses_config()
	return analysis_config.loc[(TB_campaign.run_name,TB_batch.run_name,RSD_analysis.run_name)]

def load_tracks(RSD_analysis:RunBureaucrat, DUT_z_position:float, use_DUTs_as_trigger:list=None, max_chi2dof:float=None):
	"""
	Arguments
	---------
	RSD_analysis: RunBureaucrat
		A bureaucrat pointing to the respective RSD analysis for which to
		load the tracks.
	DUT_z_position: float
		The z coordinate of the DUT to project the tracks to its plane.
	use_DUTs_as_trigger: list of str, default None
		A list with the names of other DUTs that are present in the same batch
		(i.e. in `RSD_analysis.parent`) that are added to the triggering
		by requesting a coincidence between the tracks and them, e.g. `"AC 11 (0,1)"`.
	"""
	RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	batch = RSD_analysis.parent
	
	tracks = load_tracks_from_batch(batch, only_multiplicity_one=True)
	tracks = tracks.join(project_tracks(tracks=tracks, z_position=DUT_z_position))
	
	if use_DUTs_as_trigger is not None:
		use_DUTs_as_trigger_names = set([_.split(' (')[0] for _ in use_DUTs_as_trigger])
		trigger_DUTs_hits = []
		for DUT_name in use_DUTs_as_trigger_names:
			this_DUT_hits = load_hits(
				RSD_analysis = RunBureaucrat(RSD_analysis.path_to_run_directory.parent/DUT_name),
				DUT_hit_criterion = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<-5e-3", # For the moment this is hardcoded.
			)
			trigger_DUTs_hits.append(this_DUT_hits)
		trigger_DUTs_hits = pandas.concat(
			trigger_DUTs_hits,
			join = 'outer',
			axis = 'columns',
		)
		trigger_DUTs_hits = trigger_DUTs_hits[use_DUTs_as_trigger]
		trigger_DUTs_hits['trigger_on_this_event'] = trigger_DUTs_hits.sum(axis=1)>0
		tracks = utils.select_by_multiindex(
			df = tracks,
			idx = trigger_DUTs_hits.query('trigger_on_this_event == True').index,
		)
	
	tracks['chi2/ndof'] = tracks['chi2']/tracks['ndof']
	if max_chi2dof is not None:
		tracks = tracks.query(f'`chi2/ndof`<{max_chi2dof}')
		
	return tracks
	
def load_hits(RSD_analysis:RunBureaucrat, DUT_hit_criterion:str):
	RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	batch = RSD_analysis.parent
	
	DUT_hits = read_parsed_from_waveforms_from_batch(
		batch = batch,
		DUT_name = RSD_analysis.run_name,
		variables = [], # No need for variables, only need to know which ones are hits.
		additional_SQL_selection = DUT_hit_criterion,
	)
	
	setup_config = utils.load_setup_configuration_info(batch)
	
	DUT_hits = DUT_hits.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])['DUT_name_rowcol'])
	DUT_hits.reset_index(['n_CAEN','CAEN_n_channel'], drop=True, inplace=True)
	DUT_hits['has_hit'] = True
	DUT_hits.set_index('DUT_name_rowcol',append=True,inplace=True)
	DUT_hits = DUT_hits.unstack('DUT_name_rowcol', fill_value=False)
	DUT_hits = DUT_hits['has_hit']
	
	return DUT_hits

class TracksDataLoader:
	def __init__(self, RSD_analysis:RunBureaucrat, use_DUTs_as_trigger:list=None):
		RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
		self._RSD_analysis = RSD_analysis
		self._use_DUTs_as_trigger = use_DUTs_as_trigger
	
	def get_data(self):
		if not hasattr(self, '_tracks'):
			analysis_config = load_this_RSD_analysis_config(self._RSD_analysis)
			tracks = load_tracks(
				RSD_analysis = self._RSD_analysis,
				DUT_z_position = analysis_config['DUT_z_position'],
				use_DUTs_as_trigger = self._use_DUTs_as_trigger,
			)
			tracks[['Px','Py']] = translate_and_then_rotate(
				points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
				x_translation = analysis_config['x_translation'],
				y_translation = analysis_config['y_translation'],
				angle_rotation = analysis_config['rotation_around_z_deg']/180*numpy.pi,
			)
			self._tracks = tracks
		return self._tracks

class ParsedFromWaveformsDataLoader:
	def __init__(self, batch:RunBureaucrat, DUT_name:str, variables:list=['Amplitude (V)'], additional_SQL_selection:str=None, n_events:int=None):
		batch.check_these_tasks_were_run_successfully('runs')
		self._batch = batch
		self._DUT_name = DUT_name
		self._variables = variables
		self._additional_SQL_selection = additional_SQL_selection
		self._n_events = n_events
	
	def get_data(self):
		if not hasattr(self, '_data'):
			DUT_waveforms_data = read_parsed_from_waveforms_from_batch(
				batch = self._batch,
				DUT_name = self._DUT_name,
				variables = self._variables,
				additional_SQL_selection = self._additional_SQL_selection,
			)
			setup_config = utils.load_setup_configuration_info(self._batch)
			DUT_waveforms_data = DUT_waveforms_data.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])[['row','col']])
			DUT_waveforms_data.reset_index(['n_CAEN','CAEN_n_channel'], drop=True, inplace=True)
			DUT_waveforms_data.set_index(['row','col'], append=True, inplace=True)
			DUT_waveforms_data = DUT_waveforms_data.unstack(['row','col'])
			self._data = DUT_waveforms_data
		return self._data

def calculate_features(data:pandas.DataFrame):
	"""Calculate the different features used for position reconstruction.
	
	Arguments
	---------
	data: pandas.DataFrame
		A data frame with the data measured from the RSD sensors. Each row
		is one event, each column one pixel's data, example:
		```
		              Amplitude (V)                               Collected charge (V s)                                          
		row                       0                   1                                0                           1              
		col                       0         1         0         1                      0             1             0             1
		n_run n_event                                                                                                             
		95    0                 NaN       NaN -0.004031 -0.005146                    NaN           NaN -4.824668e-12 -8.137969e-12
			  6                 NaN       NaN -0.021686 -0.006714                    NaN           NaN -3.005961e-11 -1.079528e-11
			  14          -0.004135       NaN -0.004243       NaN          -5.658003e-12           NaN -7.099769e-12           NaN
			  15                NaN -0.007979       NaN       NaN                    NaN -1.311078e-11           NaN           NaN
			  18          -0.020107 -0.015712 -0.008512 -0.008586          -2.944584e-11 -2.822604e-11 -1.378297e-11 -1.435356e-11
		...                     ...       ...       ...       ...                    ...           ...           ...           ...
		100   25248             NaN       NaN -0.027735 -0.016106                    NaN           NaN -4.133678e-11 -2.733657e-11
			  25250             NaN -0.005829       NaN -0.009288                    NaN -9.100285e-12           NaN -1.310064e-11
			  25253             NaN -0.008117 -0.008315 -0.093955                    NaN -9.754022e-12 -9.191558e-12 -1.376223e-10
			  25258             NaN       NaN       NaN -0.008348                    NaN           NaN           NaN -1.077767e-11
			  25259       -0.009984 -0.029141       NaN -0.013372          -1.009468e-11 -4.516632e-11           NaN -1.375321e-11

		[20282 rows x 8 columns]
		```
	"""
	if list(data.index.names) != ['n_run','n_event']:
		raise ValueError(f'The index levels of `data` must be [n_run,n_event].')
	if len({'row','col'} - set(data.columns.names)) > 0:
		raise ValueError('Both "row" and "col" must be present in the columns levels of `data`.')
	total_amplitude = data['Amplitude (V)'].sum(axis=1, skipna=True)
	amplitude_shared_fraction = (data['Amplitude (V)'].stack(['row','col'])/total_amplitude).unstack(['row','col']).sort_index(axis=1)
	total_charge = data['Collected charge (V s)'].sum(axis=1, skipna=True)
	charge_shared_fraction = (data['Collected charge (V s)'].stack(['row','col'])/total_charge).unstack(['row','col']).sort_index(axis=1)
	amplitude_imbalance = pandas.DataFrame(
		{
			'x': amplitude_shared_fraction[(0,1)].fillna(0) + amplitude_shared_fraction[(1,1)].fillna(0) - amplitude_shared_fraction[(0,0)].fillna(0) - amplitude_shared_fraction[(1,0)].fillna(0),
			'y': amplitude_shared_fraction[(0,0)].fillna(0) + amplitude_shared_fraction[(0,1)].fillna(0) - amplitude_shared_fraction[(1,0)].fillna(0) - amplitude_shared_fraction[(1,1)].fillna(0),
		},
		index = amplitude_shared_fraction.index,
	)
	charge_imbalance = pandas.DataFrame(
		{
			'x': charge_shared_fraction[(0,1)].fillna(0) + charge_shared_fraction[(1,1)].fillna(0) - charge_shared_fraction[(0,0)].fillna(0) - charge_shared_fraction[(1,0)].fillna(0),
			'y': charge_shared_fraction[(0,0)].fillna(0) + charge_shared_fraction[(0,1)].fillna(0) - charge_shared_fraction[(1,0)].fillna(0) - charge_shared_fraction[(1,1)].fillna(0),
		},
		index = charge_shared_fraction.index,
	)
	_ = {}
	_['ASF'] = amplitude_shared_fraction
	_['CSF'] = charge_shared_fraction
	_['amplitude_imbalance'] = amplitude_imbalance
	_['charge_imbalance'] = charge_imbalance
	return _

# Tasks ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

def setup_RSD_LGAD_analysis_within_batch(batch:RunBureaucrat, DUT_name:str)->RunBureaucrat:
	"""Setup a directory structure to perform further analysis of an RSD-LGAD
	that is inside a batch pointed by `batch`. This should be the 
	first step before starting an RSD-LGAD analysis."""
	batch.check_these_tasks_were_run_successfully(['runs','batch_info'])
	
	with batch.handle_task('RSD-LGADs_analyses', drop_old_data=False) as employee:
		setup_configuration_info = utils.load_setup_configuration_info(batch)
		if DUT_name not in set(setup_configuration_info['DUT_name']):
			raise RuntimeError(f'DUT_name {repr(DUT_name)} not present within the set of DUTs in {batch.pseudopath}, which is {set(setup_configuration_info["DUT_name"])}')
		
		try:
			RSDLGAD_bureaucrat = employee.create_subrun(DUT_name)
			with RSDLGAD_bureaucrat.handle_task('this_is_an_RSD-LGAD_analysis'):
				pass
			logging.info(f'Directory structure for RSD-LGAD analysis "{RSDLGAD_bureaucrat.pseudopath}" was created.')
		except RuntimeError as e: # This will happen if the run already existed beforehand.
			if 'Cannot create run' in str(e):
				RSDLGAD_bureaucrat = [b for b in batch.list_subruns_of_task('TI-LGADs_analyses') if b.run_name==DUT_name][0] # Get the bureaucrat to return it.
			else:
				raise e
	return RSDLGAD_bureaucrat

def plot_distributions(RSD_analysis:RunBureaucrat, force:bool=False):
	"""Plot some raw distributions, useful to configure cuts and thresholds
	for further steps in the analysis."""
	MAXIMUM_NUMBER_OF_EVENTS = 9999
	TASK_NAME = 'plot_distributions'
	
	RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	
	if force==False and RSD_analysis.was_task_run_successfully(TASK_NAME):
		return
	
	with RSD_analysis.handle_task(TASK_NAME) as employee:
		setup_config = utils.load_setup_configuration_info(RSD_analysis.parent)
		
		save_distributions_plots_here = employee.path_to_directory_of_my_task/'distributions'
		save_distributions_plots_here.mkdir()
		for variable in ['Amplitude (V)','t_50 (s)','Noise (V)','Time over 50% (s)',]:
			logging.info(f'Plotting {variable} distribution...')
			data = read_parsed_from_waveforms_from_batch(
				batch = RSD_analysis.parent,
				DUT_name = RSD_analysis.run_name,
				variables = [variable],
				n_events = MAXIMUM_NUMBER_OF_EVENTS,
			)
			data = data.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])['DUT_name_rowcol'])
			fig = px.ecdf(
				data.sort_values('DUT_name_rowcol'),
				title = f'{variable} distribution<br><sup>{RSD_analysis.pseudopath}</sup>',
				x = variable,
				marginal = 'histogram',
				color = 'DUT_name_rowcol',
			)
			fig.write_html(
				save_distributions_plots_here/f'{variable}_ECDF.html',
				include_plotlyjs = 'cdn',
			)
		
		save_scatter_plots_here = employee.path_to_directory_of_my_task/'scatter_plots'
		save_scatter_plots_here.mkdir()
		for x,y in [('t_50 (s)','Amplitude (V)'), ('Time over 50% (s)','Amplitude (V)'),]:
			logging.info(f'Plotting {y} vs {x} scatter_plot...')
			data = read_parsed_from_waveforms_from_batch(
				batch = RSD_analysis.parent,
				DUT_name = RSD_analysis.run_name,
				variables = [x,y],
				n_events = MAXIMUM_NUMBER_OF_EVENTS,
			)
			data = data.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])['DUT_name_rowcol'])
			fig = px.scatter(
				data.sort_values('DUT_name_rowcol').reset_index(drop=False),
				title = f'{y} vs {x} scatter plot<br><sup>{RSD_analysis.pseudopath}</sup>',
				x = x,
				y = y,
				color = 'DUT_name_rowcol',
				hover_data = ['n_run','n_event'],
			)
			fig.write_html(
				save_scatter_plots_here/f'{y}_vs_{x}_scatter.html',
				include_plotlyjs = 'cdn',
			)

def project_tracks(tracks:pandas.DataFrame, z_position):
	logging.info('Projecting tracks onto DUT...')
	projected = utils.project_track_in_z(
		A = tracks[[f'A{_}' for _ in ['x','y','z']]].to_numpy().T,
		B = tracks[[f'B{_}' for _ in ['x','y','z']]].to_numpy().T,
		z = z_position,
	).T
	projected = pandas.DataFrame(
		projected,
		columns = ['Px','Py','Pz'],
		index = tracks.index,
	)
	return projected

def plot_tracks_and_hits(RSD_analysis:RunBureaucrat, do_3D_plot:bool=False, force:bool=False):
	RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	TASK_NAME = 'plot_tracks_and_hits'
	
	if force==False and RSD_analysis.was_task_run_successfully(TASK_NAME):
		return
	
	with RSD_analysis.handle_task(TASK_NAME) as employee:
		batch = RSD_analysis.parent
		
		analysis_config = load_this_RSD_analysis_config(RSD_analysis)
		
		tracks = load_tracks_from_batch(batch, only_multiplicity_one=True)
		tracks = tracks.join(project_tracks(tracks=tracks, z_position=analysis_config['DUT_z_position']))
		
		DUT_hits = read_parsed_from_waveforms_from_batch(
			batch = batch,
			DUT_name = RSD_analysis.run_name,
			variables = [], # No need for variables, only need to know which ones are hits.
			additional_SQL_selection = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{analysis_config['Amplitude threshold (V)']}",
		)
		
		setup_config = utils.load_setup_configuration_info(batch)
		
		DUT_hits = DUT_hits.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])['DUT_name_rowcol'])
		
		tracks = tracks.join(DUT_hits['DUT_name_rowcol'])
		
		tracks['DUT_name_rowcol'] = tracks['DUT_name_rowcol'].fillna('no hit')
		tracks = tracks.sort_values('DUT_name_rowcol', ascending=False)
		
		logging.info('Plotting tracks and hits on DUT...')
		fig = px.scatter(
			tracks.reset_index(),
			title = f'Tracks projected on the DUT<br><sup>{RSD_analysis.pseudopath}</sup>',
			x = 'Px',
			y = 'Py',
			color = 'DUT_name_rowcol',
			hover_data = ['n_run','n_event'],
			labels = {
				'Px': 'x (m)',
				'Py': 'y (m)',
			},
		)
		fig.update_yaxes(
			scaleanchor = "x",
			scaleratio = 1,
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'tracks_projected_on_DUT.html',
			include_plotlyjs = 'cdn',
		)
		
		if do_3D_plot:
			logging.info('Doing 3D tracks plot...')
			tracks_for_plotly = []
			for AB in ['A','B']:
				_ = tracks[[f'{AB}{coord}' for coord in ['x','y','z']]]
				_.columns = ['x','y','z']
				tracks_for_plotly.append(_)
			tracks_for_plotly = pandas.concat(tracks_for_plotly)
			tracks_for_plotly = tracks_for_plotly.join(tracks['DUT_name_rowcol'])
			
			fig = px.line_3d(
				tracks_for_plotly.reset_index(drop=False).query('DUT_name_rowcol != "no hit"'),
				title = f'Tracks<br><sup>{RSD_analysis.pseudopath}</sup>',
				x = 'x',
				y = 'y',
				z = 'z',
				color = 'DUT_name_rowcol',
				line_group = 'n_event',
				labels = {
					'x': 'x (m)',
					'y': 'y (m)',
					'z': 'z (m)',
				},
			)
			fig.update_layout(hovermode=False)
			# ~ fig.layout.scene.camera.projection.type = "orthographic"
			fig.write_html(
				employee.path_to_directory_of_my_task/f'tracks_3D.html',
				include_plotlyjs = 'cdn',
			)

def transformation_for_centering_and_leveling(RSD_analysis:RunBureaucrat, draw_square:bool=True, force:bool=False):
	RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	
	if force==False and RSD_analysis.was_task_run_successfully('transformation_for_centering_and_leveling'):
		return
	
	with RSD_analysis.handle_task('transformation_for_centering_and_leveling') as employee:
		batch = RSD_analysis.parent
		
		analysis_config = load_this_RSD_analysis_config(RSD_analysis)
		__ = {'x_translation','y_translation','rotation_around_z_deg'}
		if any([numpy.isnan(analysis_config[_]) for _ in __]):
			raise RuntimeError(f'One (or more) of {__} is `NaN`, check the spreadsheet.')
		
		tracks = load_tracks(
			RSD_analysis = RSD_analysis,
			DUT_z_position = analysis_config['DUT_z_position'],
		)
		hits = load_hits(
			RSD_analysis = RSD_analysis,
			DUT_hit_criterion = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{analysis_config['Amplitude threshold (V)']}",
		)
		
		cluster_size = hits.sum(axis=1)
		cluster_size.name = 'cluster_size'
		
		tracks = tracks.join(cluster_size)
		tracks['cluster_size'].fillna(0, inplace=True)
		tracks['cluster_size'] = tracks['cluster_size'].astype(int)
		
		tracks[['Px_transformed','Py_transformed']] = translate_and_then_rotate(
			points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
			x_translation = analysis_config['x_translation'],
			y_translation = analysis_config['y_translation'],
			angle_rotation = analysis_config['rotation_around_z_deg']/180*numpy.pi,
		)
		
		logging.info('Plotting tracks and hits on DUT...')
		fig = px.scatter(
			data_frame = tracks.reset_index().sort_values('cluster_size').astype({'cluster_size':str}),
			title = f'Cluster size<br><sup>{RSD_analysis.pseudopath}</sup>',
			x = 'Px_transformed',
			y = 'Py_transformed',
			color = 'cluster_size',
			hover_data = ['n_run','n_event'],
			labels = {
				'Px_transformed': 'x (m)',
				'Py_transformed': 'y (m)',
			},
		)
		for xy,method in dict(x=fig.add_vline, y=fig.add_hline).items():
			method(0)
		if draw_square:
			fig.add_shape(
				type = "rect",
				x0 = -analysis_config['DUT pitch (m)']/2,
				y0 = -analysis_config['DUT pitch (m)']/2, 
				x1 = analysis_config['DUT pitch (m)']/2, 
				y1 = analysis_config['DUT pitch (m)']/2,
			)
		fig.update_yaxes(
			scaleanchor = "x",
			scaleratio = 1,
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'tracks_projected_on_DUT.html',
			include_plotlyjs = 'cdn',
		)

def plot_cluster_size(RSD_analysis:RunBureaucrat, force:bool=False, use_DUTs_as_trigger:list=None, DUT_ROI_margin:float=None, draw_square:bool=True):
	"""
	Arguments
	---------
	use_DUTs_as_trigger: list of str, default None
		A list with the names of other DUTs that are present in the same batch
		(i.e. in `RSD_analysis.parent`) that are added to the triggering
		by requesting a coincidence between the tracks and them, e.g. `"AC 11 (0,1)"`.
	DUT_ROI_margin: float, default None
		If `None`, this argumet is ignored. If a float number, this defines
		a margin to consider for the ROI of the DUT which is defined
		by its pitch. For example, if the pitch is 500 µm and `DUT_ROI_margin=0`,
		then only the tracks inside a square of 500 µm side are used. If,
		instead, `DUT_ROI_margin=50e-6` then the square is reduced by 50 µm
		in each side, i.e. margins of 50 µm are added to it. Negative values
		are allowed, which increase the ROI.
	"""
	RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	TASK_NAME = 'plot_cluster_size'
	
	if force==False and RSD_analysis.was_task_run_successfully(TASK_NAME):
		return
	
	with RSD_analysis.handle_task(TASK_NAME) as employee:
		analysis_config = load_this_RSD_analysis_config(RSD_analysis)
		
		tracks = load_tracks(
			RSD_analysis = RSD_analysis,
			DUT_z_position = analysis_config['DUT_z_position'],
			use_DUTs_as_trigger = use_DUTs_as_trigger,
		)
		hits = load_hits(
			RSD_analysis = RSD_analysis,
			DUT_hit_criterion = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{analysis_config['Amplitude threshold (V)']}",
		)
		
		cluster_size = hits.sum(axis=1)
		cluster_size.name = 'cluster_size'
		
		tracks = tracks.join(cluster_size)
		tracks['cluster_size'].fillna(0, inplace=True)
		tracks['cluster_size'] = tracks['cluster_size'].astype(int)
		
		tracks[['Px','Py']] = translate_and_then_rotate(
			points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
			x_translation = analysis_config['x_translation'],
			y_translation = analysis_config['y_translation'],
			angle_rotation = analysis_config['rotation_around_z_deg']/180*numpy.pi,
		)
		
		if DUT_ROI_margin is not None:
			ROI_size = analysis_config['DUT pitch (m)']
			if numpy.isnan(ROI_size):
				raise RuntimeError(f'The `ROI_size` is NaN, probably it is not configured in the analyses spreadsheet...')
			tracks = tracks.query(f'Px>{-ROI_size/2+DUT_ROI_margin}')
			tracks = tracks.query(f'Px<{ROI_size/2-DUT_ROI_margin}')
			tracks = tracks.query(f'Py>{-ROI_size/2+DUT_ROI_margin}')
			tracks = tracks.query(f'Py<{ROI_size/2-DUT_ROI_margin}')
		
		fig = px.scatter(
			data_frame = tracks.reset_index().sort_values('cluster_size').astype({'cluster_size':str}),
			title = f'Cluster size<br><sup>{RSD_analysis.pseudopath}</sup>',
			x = 'Px',
			y = 'Py',
			color = 'cluster_size',
			hover_data = ['n_run','n_event'],
			labels = {
				'Px': 'x (m)',
				'Py': 'y (m)',
			},
		)
		fig.update_yaxes(
			scaleanchor = "x",
			scaleratio = 1,
		)
		if draw_square:
			fig.add_shape(
				type = "rect",
				x0 = -analysis_config['DUT pitch (m)']/2,
				y0 = -analysis_config['DUT pitch (m)']/2, 
				x1 = analysis_config['DUT pitch (m)']/2, 
				y1 = analysis_config['DUT pitch (m)']/2,
			)
		fig.write_html(
			employee.path_to_directory_of_my_task/'cluster_size.html',
			include_plotlyjs = 'cdn',
		)
		
		fig = px.histogram(
			tracks,
			x =  'cluster_size',
			title = f'Cluster size<br><sup>{RSD_analysis.pseudopath}</sup>',
			text_auto = True,
		)
		fig.update_yaxes(type="log")
		fig.write_html(
			employee.path_to_directory_of_my_task/'cluster_size_histogram.html',
			include_plotlyjs = 'cdn',
		)

def plot_features(RSD_analysis:RunBureaucrat, force:bool=False, use_DUTs_as_trigger:list=None, DUT_ROI_margin:float=None, draw_square:bool=True):
	"""
	Arguments
	---------
	use_DUTs_as_trigger: list of str, default None
		A list with the names of other DUTs that are present in the same batch
		(i.e. in `RSD_analysis.parent`) that are added to the triggering
		by requesting a coincidence between the tracks and them, e.g. `"AC 11 (0,1)"`.
	DUT_ROI_margin: float, default None
		If `None`, this argumet is ignored. If a float number, this defines
		a margin to consider for the ROI of the DUT which is defined
		by its pitch. For example, if the pitch is 500 µm and `DUT_ROI_margin=0`,
		then only the tracks inside a square of 500 µm side are used. If,
		instead, `DUT_ROI_margin=50e-6` then the square is reduced by 50 µm
		in each side, i.e. margins of 50 µm are added to it. Negative values
		are allowed, which increase the ROI.
	"""
	RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	TASK_NAME = 'plot_features'
	
	if force==False and RSD_analysis.was_task_run_successfully(TASK_NAME):
		return
	
	with RSD_analysis.handle_task(TASK_NAME) as employee:
		analysis_config = load_this_RSD_analysis_config(RSD_analysis)
		
		tracks = load_tracks(
			RSD_analysis = RSD_analysis,
			DUT_z_position = analysis_config['DUT_z_position'],
			use_DUTs_as_trigger = use_DUTs_as_trigger,
		)
		tracks.reset_index('n_track', inplace=True)
		DUT_waveforms_data = read_parsed_from_waveforms_from_batch(
			batch = RSD_analysis.parent,
			DUT_name = RSD_analysis.run_name,
			variables = ['Amplitude (V)','Collected charge (V s)'],
			additional_SQL_selection = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{analysis_config['Amplitude threshold (V)']}",
		)
		setup_config = utils.load_setup_configuration_info(RSD_analysis.parent)
		DUT_waveforms_data = DUT_waveforms_data.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])[['row','col']])
		DUT_waveforms_data.reset_index(['n_CAEN','CAEN_n_channel'], drop=True, inplace=True)
		DUT_waveforms_data.set_index(['row','col'], append=True, inplace=True)
		DUT_waveforms_data = DUT_waveforms_data.unstack(['row','col'])
		DUT_features = calculate_features(DUT_waveforms_data)
		
		tracks[['Px','Py']] = translate_and_then_rotate(
			points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
			x_translation = analysis_config['x_translation'],
			y_translation = analysis_config['y_translation'],
			angle_rotation = analysis_config['rotation_around_z_deg']/180*numpy.pi,
		)
		
		if DUT_ROI_margin is not None:
			ROI_size = analysis_config['DUT pitch (m)']
			if numpy.isnan(ROI_size):
				raise RuntimeError(f'The `ROI_size` is NaN, probably it is not configured in the analyses spreadsheet...')
			tracks = tracks.query(f'Px>{-ROI_size/2+DUT_ROI_margin}')
			tracks = tracks.query(f'Px<{ROI_size/2-DUT_ROI_margin}')
			tracks = tracks.query(f'Py>{-ROI_size/2+DUT_ROI_margin}')
			tracks = tracks.query(f'Py<{ROI_size/2-DUT_ROI_margin}')
		
		for feature_name in ['ASF','CSF']:
			feature_data = DUT_features[feature_name]
			feature_data.fillna(0, inplace=True)
			feature_data = feature_data.stack(['row','col'])
			feature_data.name = feature_name
			feature_data = feature_data.to_frame()
			feature_data.reset_index(['row','col'], drop=False, inplace=True)
			feature_data = feature_data.join(tracks[['Px','Py']])
			
			fig = px.scatter(
				data_frame = feature_data.reset_index().sort_values(['row','col']),
				title = f'{feature_name}<br><sup>{RSD_analysis.pseudopath}</sup>',
				x = 'Px',
				y = 'Py',
				color = feature_name,
				hover_data = ['n_run','n_event'],
				labels = {
					'Px': 'x (m)',
					'Py': 'y (m)',
				},
				facet_col = 'col',
				facet_row = 'row',
			)
			fig.update_yaxes(
				scaleanchor = "x",
				scaleratio = 1,
			)
			fig.update_layout(coloraxis_colorbar_title_side="right")
			if draw_square:
				for row in [0,1]:
					for col in [0,1]:
						fig.add_shape(
							type = "rect",
							x0 = -analysis_config['DUT pitch (m)']/2,
							y0 = -analysis_config['DUT pitch (m)']/2, 
							x1 = analysis_config['DUT pitch (m)']/2, 
							y1 = analysis_config['DUT pitch (m)']/2,
							row = row+1,
							col = col+1,
						)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{feature_name}.html',
				include_plotlyjs = 'cdn',
			)
		
		for feature_name in ['amplitude_imbalance','charge_imbalance']:
			feature_data = DUT_features[feature_name]
			feature_data.columns.names = ['direction']
			feature_data = feature_data.stack('direction')
			feature_data.name = feature_name
			feature_data = feature_data.to_frame()
			feature_data = feature_data.join(tracks[['Px','Py']])
			
			fig = px.scatter(
				data_frame = feature_data.reset_index().sort_values('direction'),
				title = f'{feature_name}<br><sup>{RSD_analysis.pseudopath}</sup>',
				x = 'Px',
				y = 'Py',
				color = feature_name,
				hover_data = ['n_run','n_event'],
				labels = {
					'Px': 'x (m)',
					'Py': 'y (m)',
				},
				facet_col = 'direction',
			)
			fig.update_yaxes(
				scaleanchor = "x",
				scaleratio = 1,
			)
			fig.update_layout(coloraxis_colorbar_title_side="right")
			if draw_square:
				for col in [0,1]:
					fig.add_shape(
						type = "rect",
						x0 = -analysis_config['DUT pitch (m)']/2,
						y0 = -analysis_config['DUT pitch (m)']/2, 
						x1 = analysis_config['DUT pitch (m)']/2, 
						y1 = analysis_config['DUT pitch (m)']/2,
						col = col+1,
						row = 1,
					)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{feature_name}.html',
				include_plotlyjs = 'cdn',
			)

def plot_waveforms_distributions_inside_ROI(RSD_analysis:RunBureaucrat, force:bool=False, DUT_ROI_margin:float=None, draw_square:bool=True):
	"""
	Arguments
	---------
	use_DUTs_as_trigger: list of str, default None
		A list with the names of other DUTs that are present in the same batch
		(i.e. in `RSD_analysis.parent`) that are added to the triggering
		by requesting a coincidence between the tracks and them, e.g. `"AC 11 (0,1)"`.
	DUT_ROI_margin: float, default None
		If `None`, this argumet is ignored. If a float number, this defines
		a margin to consider for the ROI of the DUT which is defined
		by its pitch. For example, if the pitch is 500 µm and `DUT_ROI_margin=0`,
		then only the tracks inside a square of 500 µm side are used. If,
		instead, `DUT_ROI_margin=50e-6` then the square is reduced by 50 µm
		in each side, i.e. margins of 50 µm are added to it. Negative values
		are allowed, which increase the ROI.
	"""
	RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	TASK_NAME = 'plot_waveforms_distributions_inside_ROI'
	
	if force==False and RSD_analysis.was_task_run_successfully(TASK_NAME):
		return
	
	with RSD_analysis.handle_task(TASK_NAME) as employee:
		analysis_config = load_this_RSD_analysis_config(RSD_analysis)
		
		tracks = load_tracks(
			RSD_analysis = RSD_analysis,
			DUT_z_position = analysis_config['DUT_z_position'],
		)
		tracks.reset_index('n_track', inplace=True)
		
		tracks[['Px','Py']] = translate_and_then_rotate(
			points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
			x_translation = analysis_config['x_translation'],
			y_translation = analysis_config['y_translation'],
			angle_rotation = analysis_config['rotation_around_z_deg']/180*numpy.pi,
		)
		
		if DUT_ROI_margin is not None:
			ROI_size = analysis_config['DUT pitch (m)']
			if numpy.isnan(ROI_size):
				raise RuntimeError(f'The `ROI_size` is NaN, probably it is not configured in the analyses spreadsheet...')
			tracks = tracks.query(f'Px>{-ROI_size/2+DUT_ROI_margin}')
			tracks = tracks.query(f'Px<{ROI_size/2-DUT_ROI_margin}')
			tracks = tracks.query(f'Py>{-ROI_size/2+DUT_ROI_margin}')
			tracks = tracks.query(f'Py<{ROI_size/2-DUT_ROI_margin}')
		
		DUT_waveforms_data = read_parsed_from_waveforms_from_batch(
			batch = RSD_analysis.parent,
			DUT_name = RSD_analysis.run_name,
			variables = ['Amplitude (V)','Collected charge (V s)','t_50 (s)','Time over 20% (s)'],
		)
		setup_config = utils.load_setup_configuration_info(RSD_analysis.parent)
		DUT_waveforms_data = DUT_waveforms_data.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])['DUT_name_rowcol'])
		DUT_waveforms_data.reset_index(['n_CAEN','CAEN_n_channel'], drop=True, inplace=True)
		
		DUT_waveforms_data = utils.select_by_multiindex(DUT_waveforms_data, tracks.index)
		DUT_waveforms_data.sort_values('DUT_name_rowcol', inplace=True)
		DUT_waveforms_data.set_index('DUT_name_rowcol', inplace=True)
		
		save_plots_here = employee.path_to_directory_of_my_task/'histograms'
		save_plots_here.mkdir()
		for col in DUT_waveforms_data.columns:
			fig = px.ecdf(
				data_frame = DUT_waveforms_data.reset_index(),
				title = f'{col} distribution<br><sup>{RSD_analysis.pseudopath}</sup>',
				x = col,
				color = 'DUT_name_rowcol',
				marginal = 'histogram',
			)
			fig.write_html(
				save_plots_here/f'{col}_distribution.html',
				include_plotlyjs = 'cdn',
			)
		
		save_plots_here = employee.path_to_directory_of_my_task/'scatters'
		save_plots_here.mkdir()
		for x,y in [('t_50 (s)','Amplitude (V)'),('Time over 20% (s)','Amplitude (V)')]:
			fig = px.scatter(
				data_frame = DUT_waveforms_data.reset_index(),
				title = f'{y} vs {x} distribution<br><sup>{RSD_analysis.pseudopath}</sup>',
				x = x,
				y = y,
				color = 'DUT_name_rowcol',
			)
			fig.write_html(
				save_plots_here/f'{y}_vs_{x}_distribution.html',
				include_plotlyjs = 'cdn',
			)

def position_reconstruction_with_charge_imbalance(RSD_analysis:RunBureaucrat, force:bool=False, use_DUTs_as_trigger:list=None, DUT_ROI_margin:float=None, draw_square:bool=True):
	"""
	Arguments
	---------
	use_DUTs_as_trigger: list of str, default None
		A list with the names of other DUTs that are present in the same batch
		(i.e. in `RSD_analysis.parent`) that are added to the triggering
		by requesting a coincidence between the tracks and them, e.g. `"AC 11 (0,1)"`.
	DUT_ROI_margin: float, default None
		If `None`, this argumet is ignored. If a float number, this defines
		a margin to consider for the ROI of the DUT which is defined
		by its pitch. For example, if the pitch is 500 µm and `DUT_ROI_margin=0`,
		then only the tracks inside a square of 500 µm side are used. If,
		instead, `DUT_ROI_margin=50e-6` then the square is reduced by 50 µm
		in each side, i.e. margins of 50 µm are added to it. Negative values
		are allowed, which increase the ROI.
	"""
	RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	TASK_NAME = 'position_reconstruction_with_charge_imbalance'
	
	if force==False and RSD_analysis.was_task_run_successfully(TASK_NAME):
		return
	
	with RSD_analysis.handle_task(TASK_NAME) as employee:
		analysis_config = load_this_RSD_analysis_config(RSD_analysis)
		
		if numpy.isnan(analysis_config['DUT pitch (m)']):
			raise RuntimeError(f"`analysis_config['DUT pitch (m)']` is NaN, please fill it in the spreadsheet for {RSD_analysis.pseudopath}.")
		
		tracks = load_tracks(
			RSD_analysis = RSD_analysis,
			DUT_z_position = analysis_config['DUT_z_position'],
			use_DUTs_as_trigger = use_DUTs_as_trigger,
			max_chi2dof = 2,
		)
		tracks.reset_index('n_track', inplace=True)
		DUT_waveforms_data = read_parsed_from_waveforms_from_batch(
			batch = RSD_analysis.parent,
			DUT_name = RSD_analysis.run_name,
			variables = ['Amplitude (V)','Collected charge (V s)'],
			additional_SQL_selection = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{analysis_config['Amplitude threshold (V)']}",
		)
		setup_config = utils.load_setup_configuration_info(RSD_analysis.parent)
		DUT_waveforms_data = DUT_waveforms_data.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])[['row','col']])
		DUT_waveforms_data.reset_index(['n_CAEN','CAEN_n_channel'], drop=True, inplace=True)
		DUT_waveforms_data.set_index(['row','col'], append=True, inplace=True)
		DUT_waveforms_data = DUT_waveforms_data.unstack(['row','col'])
		DUT_features = calculate_features(DUT_waveforms_data)
		
		tracks[['Px','Py']] = translate_and_then_rotate(
			points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
			x_translation = analysis_config['x_translation'],
			y_translation = analysis_config['y_translation'],
			angle_rotation = analysis_config['rotation_around_z_deg']/180*numpy.pi,
		)
		
		if DUT_ROI_margin is not None:
			ROI_size = analysis_config['DUT pitch (m)']
			if numpy.isnan(ROI_size):
				raise RuntimeError(f'The `ROI_size` is NaN, probably it is not configured in the analyses spreadsheet...')
			tracks = tracks.query(f'Px>{-ROI_size/2+DUT_ROI_margin}')
			tracks = tracks.query(f'Px<{ROI_size/2-DUT_ROI_margin}')
			tracks = tracks.query(f'Py>{-ROI_size/2+DUT_ROI_margin}')
			tracks = tracks.query(f'Py<{ROI_size/2-DUT_ROI_margin}')
		
		reconstructed = {}
		for feature_name in ['amplitude_imbalance','charge_imbalance']:
			reco = DUT_features[feature_name]*analysis_config['DUT pitch (m)']/2
			reco.rename(columns={'x':'x (m)','y':'y (m)'}, inplace=True)
			reconstructed[feature_name] = reco
		
		telescope_positions = tracks[['Px','Py','chi2/ndof']]
		telescope_positions.rename(columns={'Px':'x (m)','Py':'y (m)'}, inplace=True)
		
		fig = px.ecdf(
			data_frame = telescope_positions.reset_index(drop=False),
			title = f'Telescope tracks quality<br><sup>{RSD_analysis.pseudopath}</sup>',
			x = 'chi2/ndof',
			marginal = 'histogram',
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/f'telescope_chi2_over_ndof.html',
			include_plotlyjs = 'cdn',
		)
		
		for feature_name, reco in reconstructed.items():
			reco = utils.select_by_multiindex(reco, telescope_positions.index)
			reco_error = reco - telescope_positions
			reco_error.rename(columns={'x (m)':'Reconstruction error x (m)','y (m)':'Reconstruction error y (m)'}, inplace=True)
			reco_error['Reconstruction error (m)'] = sum([reco_error[f'Reconstruction error {_} (m)']**2 for _ in {'x','y'}])**.5
			reco_error = reco_error.join(telescope_positions[['x (m)','y (m)']])
			
			fig = px.ecdf(
				data_frame = reco_error.reset_index(drop=False),
				title = f'Reconstruction error using {feature_name}<br><sup>{RSD_analysis.pseudopath}</sup>',
				x = 'Reconstruction error (m)',
				marginal = 'histogram',
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'reconstruction_error_using_{feature_name}_histogram.html',
				include_plotlyjs = 'cdn',
			)
			
			quantile = reco_error['Reconstruction error (m)'].quantile(.95)
			fig = px.scatter(
				data_frame = reco_error.sort_index().reset_index(drop=False).query(f'`Reconstruction error (m)`<{quantile}'),
				title = f'Reconstruction error using {feature_name}<br><sup>{RSD_analysis.pseudopath}</sup>',
				x = 'x (m)',
				y = 'y (m)',
				color = 'Reconstruction error (m)',
				hover_data = ['n_run','n_event'],
			)
			fig.update_yaxes(
				scaleanchor = "x",
				scaleratio = 1,
			)
			fig.update_layout(coloraxis_colorbar_title_side="right")
			if draw_square:
				fig.add_shape(
					type = "rect",
					x0 = -analysis_config['DUT pitch (m)']/2,
					y0 = -analysis_config['DUT pitch (m)']/2, 
					x1 = analysis_config['DUT pitch (m)']/2, 
					y1 = analysis_config['DUT pitch (m)']/2,
				)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'reconstruction_error_using_{feature_name}_scatter.html',
				include_plotlyjs = 'cdn',
			)

def create_positions_grid_from_random_positions(positions, nx:int, ny:int, x_limits:tuple, y_limits:tuple):
	"""Create a discret xy grid of positions starting from random positions.
	
	Arguments
	---------
	positions: pandas.DataFrame
		A data frame of the form 
		```
		                      x         y
		n_run n_event                    
		95    18      -0.000061  0.000069
			  111     -0.000023 -0.000128
			  157     -0.000053  0.000206
			  442     -0.000067  0.000054
			  483      0.000246 -0.000162
		...                 ...       ...
		100   24829    0.000140  0.000227
			  24887   -0.000030  0.000086
			  25060   -0.000085 -0.000066
			  25141   -0.000181  0.000212
			  25199   -0.000073  0.000181

		[1095 rows x 2 columns]
		```
	nx, ny: int
		Number of discrete bins in x,y.
	x_limits, y_limits: tuple
		A tuple with 2 floats, giving the maximum extension of the grid
		in x,y. 
	
	Returns
	-------
	positions: pandas.DataFrame
		The new positions with discrete values, example:
		```
		               n_x  n_y  n_position         x         y
		n_run n_event                                          
		95    18         1    2           6 -0.000083  0.000083
			  111        1    1           5 -0.000083 -0.000083
			  157        1    3           7 -0.000083  0.000250
			  442        1    2           6 -0.000083  0.000083
			  483        3    1          13  0.000250 -0.000083
		...            ...  ...         ...       ...       ...
		100   24829      2    3          11  0.000083  0.000250
			  24887      1    2           6 -0.000083  0.000083
			  25060      1    1           5 -0.000083 -0.000083
			  25141      0    3           3 -0.000250  0.000250
			  25199      1    3           7 -0.000083  0.000250

		[1095 rows x 5 columns]
		```
	new_positions_table: pandas.DataFrame
		A table with the grid created, example:
		```
		         n_position         x         y
		n_x n_y                                
		0   0             0 -0.000250 -0.000250
			1             1 -0.000250 -0.000083
			2             2 -0.000250  0.000083
			3             3 -0.000250  0.000250
		1   0             4 -0.000083 -0.000250
			1             5 -0.000083 -0.000083
			2             6 -0.000083  0.000083
			3             7 -0.000083  0.000250
		2   0             8  0.000083 -0.000250
			1             9  0.000083 -0.000083
			2            10  0.000083  0.000083
			3            11  0.000083  0.000250
		3   0            12  0.000250 -0.000250
			1            13  0.000250 -0.000083
			2            14  0.000250  0.000083
			3            15  0.000250  0.000250
		```
	"""
	positions = positions.copy() # Don't want to touch the original.
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		x = pandas.Series(numpy.linspace(x_limits[0], x_limits[1], nx), name='x')
		y = pandas.Series(numpy.linspace(y_limits[0], y_limits[1], ny), name='y')
		x.index.rename('n_x', inplace=True)
		y.index.rename('n_y', inplace=True)
		
		x_for_table = x.to_frame()
		y_for_table = y.to_frame()
		for xy in [x_for_table,y_for_table]:
			xy.reset_index(drop=False, inplace=True)
		new_positions_table = x_for_table.merge(y_for_table, how='cross')
		new_positions_table.index.rename('n_position', inplace=True)
		new_positions_table.reset_index(drop=False, inplace=True)
		new_positions_table.set_index(['n_x','n_y'], inplace=True)
		
		for xy_label, xy in {'x':x, 'y':y}.items():
			positions[f'n_{xy_label}'] = 0
			for n_xy,_xy in xy[1:].items():
				positions.loc[positions[xy_label]>(_xy-(xy[n_xy]-xy[n_xy-1])/2),f'n_{xy_label}'] += 1
		positions = positions.drop(columns=['x','y'])
		index_cols = positions.index.names
		positions.reset_index(inplace=True, drop=False)
		positions.set_index(new_positions_table.index.names, inplace=True)
		positions = positions.join(new_positions_table)
		positions.reset_index(inplace=True, drop=False)
		positions.set_index(index_cols, inplace=True)
		positions.sort_index(inplace=True)
	
	return positions, new_positions_table

def new_position_reconstruction(RSD_analysis:RunBureaucrat, reconstructor:reconstructors.RSDPositionReconstructor, features, positions, reconstructor_name:str, position_grid_parameters:dict, metadata:dict=None, reconstructor_fit_kwargs:dict=None, do_quiver_plot:bool=True):
	"""Creates and trains (fit) a new position reconstructor for RSD."""
	RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	
	if len(features) != len(positions):
		raise ValueError(f'`features` and `positions` have different lengths.')
	
	with RSD_analysis.handle_task('position_reconstructors', drop_old_data=False) as employee:
		reconstructor_bureaucrat = employee.create_subrun(reconstructor_name, if_exists='skip')
		with reconstructor_bureaucrat.handle_task('train') as employee_train:
			with open(employee_train.path_to_directory_of_my_task/'train_parameters.json', 'w') as ofile:
				json.dump(
					dict(
						position_grid_parameters = position_grid_parameters,
						reconstructor_type = str(type(reconstructor)),
						metadata = metadata,
						number_of_events_in_train_dataset = len(features),
					), 
					ofile,
					indent = '\t',
				)
			
			positions_discretized, positions_grid = create_positions_grid_from_random_positions(
				positions = positions,
				nx = position_grid_parameters['n_x'], 
				ny = position_grid_parameters['n_y'],
				x_limits = (position_grid_parameters['x_min'], position_grid_parameters['x_max']),
				y_limits = (position_grid_parameters['y_min'], position_grid_parameters['y_max']),
			)
			
			if do_quiver_plot:
				logging.info('Producing grid plot...')
				fig, ax = plt.subplots()
				ax.quiver(
					positions['x'],
					positions['y'],
					(positions_discretized['x']-positions['x']),
					(positions_discretized['y']-positions['y']),
					angles = 'xy',
					scale_units = 'xy',
					scale = 1,
				)
				ax.scatter(
					x = positions_grid['x'],
					y = positions_grid['y'],
					c = 'red',
				)
				ax.set_aspect('equal')
				ax.set_xlabel('x (m)')
				ax.set_ylabel('y (m)')
				plt.title(f'Position grid discretization\n{reconstructor_bureaucrat.pseudopath}')
				for fmt in {'png','pdf'}:
					plt.savefig(employee_train.path_to_directory_of_my_task/f'positions_grid.{fmt}')
			
			logging.info(f'Fitting reconstructor...')
			if reconstructor_fit_kwargs is None:
				reconstructor_fit_kwargs = dict()
			
			reconstructor.fit(
				positions = positions_discretized.set_index('n_position', append=True)[['x','y']],
				features = features.join(positions_discretized['n_position']),
				**reconstructor_fit_kwargs,
			)
			with open(employee_train.path_to_directory_of_my_task/'reconstructor.pickle', 'wb') as ofile:
				pickle.dump(reconstructor, ofile, pickle.HIGHEST_PROTOCOL)

# ~ def use_reconstructor(reconstructor_bureaucrat:RunBureaucrat, reconstruction_name:str, features, reconstructor_kwargs:dict=None, reconstruct_data_from_RSD_analysis:RunBureaucrat, use_DUTs_as_trigger:list=None, DUT_ROI_margin:float=None, draw_square:bool=True):
	# ~ reconstructor_bureaucrat.check_these_tasks_were_run_successfully('train')
	# ~ reconstructor_bureaucrat.parent.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	# ~ reconstruct_data_from_RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	
	
	# ~ analysis_config = load_this_RSD_analysis_config(RSD_analysis)
	
	# ~ tracks = load_tracks(
		# ~ RSD_analysis = RSD_analysis,
		# ~ DUT_z_position = analysis_config['DUT_z_position'],
		# ~ use_DUTs_as_trigger = use_DUTs_as_trigger,
	# ~ )
	# ~ tracks.reset_index('n_track', inplace=True)
	
	# ~ DUT_features = calculate_features(DUT_waveforms_data)
	
	# ~ tracks[['Px','Py']] = translate_and_then_rotate(
		# ~ points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
		# ~ x_translation = analysis_config['x_translation'],
		# ~ y_translation = analysis_config['y_translation'],
		# ~ angle_rotation = analysis_config['rotation_around_z_deg']/180*numpy.pi,
	# ~ )
	
	# ~ if DUT_ROI_margin is not None:
		# ~ ROI_size = analysis_config['DUT pitch (m)']
		# ~ if numpy.isnan(ROI_size):
			# ~ raise RuntimeError(f'The `ROI_size` is NaN, probably it is not configured in the analyses spreadsheet...')
		# ~ tracks = tracks.query(f'Px>{-ROI_size/2+DUT_ROI_margin}')
		# ~ tracks = tracks.query(f'Px<{ROI_size/2-DUT_ROI_margin}')
		# ~ tracks = tracks.query(f'Py>{-ROI_size/2+DUT_ROI_margin}')
		# ~ tracks = tracks.query(f'Py<{ROI_size/2-DUT_ROI_margin}')
	
	

def train_reconstructors(RSD_analysis:RunBureaucrat, force:bool=False, use_DUTs_as_trigger:list=None, DUT_ROI_margin:float=None, draw_square:bool=True):
	"""
	Arguments
	---------
	use_DUTs_as_trigger: list of str, default None
		A list with the names of other DUTs that are present in the same batch
		(i.e. in `RSD_analysis.parent`) that are added to the triggering
		by requesting a coincidence between the tracks and them, e.g. `"AC 11 (0,1)"`.
	DUT_ROI_margin: float, default None
		If `None`, this argumet is ignored. If a float number, this defines
		a margin to consider for the ROI of the DUT which is defined
		by its pitch. For example, if the pitch is 500 µm and `DUT_ROI_margin=0`,
		then only the tracks inside a square of 500 µm side are used. If,
		instead, `DUT_ROI_margin=50e-6` then the square is reduced by 50 µm
		in each side, i.e. margins of 50 µm are added to it. Negative values
		are allowed, which increase the ROI.
	"""
	RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	TASK_NAME = 'deleteme'
	
	if force==False and RSD_analysis.was_task_run_successfully(TASK_NAME):
		return
	
	analysis_config = load_this_RSD_analysis_config(RSD_analysis)
	
	tracks_loader = TracksDataLoader(
		RSD_analysis = RSD_analysis,
		use_DUTs_as_trigger = use_DUTs_as_trigger,
	)
	tracks.reset_index('n_track', inplace=True)
	DUT_waveforms_data_loader = ParsedFromWaveformsDataLoader(
		batch = RSD_analysis.parent,
		DUT_name = RSD_analysis.run_name,
		variables = ['Amplitude (V)','Collected charge (V s)'],
		additional_SQL_selection = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{analysis_config['Amplitude threshold (V)']}",
	)
	
	DUT_features = calculate_features(DUT_waveforms_data)
	
	
	
	if DUT_ROI_margin is not None:
		ROI_size = analysis_config['DUT pitch (m)']
		if numpy.isnan(ROI_size):
			raise RuntimeError(f'The `ROI_size` is NaN, probably it is not configured in the analyses spreadsheet...')
		tracks = tracks.query(f'Px>{-ROI_size/2+DUT_ROI_margin}')
		tracks = tracks.query(f'Px<{ROI_size/2-DUT_ROI_margin}')
		tracks = tracks.query(f'Py>{-ROI_size/2+DUT_ROI_margin}')
		tracks = tracks.query(f'Py<{ROI_size/2-DUT_ROI_margin}')
	
	################################################################
	# First of all, convert the current data format to the format expected by the reconstructors.
	
	positions = tracks[['Px','Py']]
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		positions.rename(columns={'Px':'x','Py':'y'}, inplace=True)
	
	data_for_reconstructors = {}
	
	for feature_name in ['ASF','CSF']:
		data = DUT_features[feature_name]
		data.columns = [f'{feature_name} ({row},{col})' for row,col in data.columns]
		data = data.fillna(0)
		# ~ data = data.merge(positions, how='inner', left_index=True, right_index=True)
		data = utils.select_by_multiindex(data, positions.index)
		data_for_reconstructors[feature_name] = data
	
	for feature_name in ['amplitude_imbalance','charge_imbalance']:
		data = DUT_features[feature_name]
		data.columns = [f'{feature_name} {direction}' for direction in data.columns]
		data = utils.select_by_multiindex(data, positions.index)
		data_for_reconstructors[feature_name] = data
	
	for n_grid in [2,3,4,5,7,11,22]:
		new_position_reconstruction(
			RSD_analysis = RSD_analysis,
			reconstructor = reconstructors.LookupTablePositionReconstructor(),
			features = data_for_reconstructors['ASF'],
			positions = positions,
			reconstructor_name = f'lookup_table_{n_grid}x{n_grid}', 
			position_grid_parameters = dict(
				n_x = n_grid,
				n_y = n_grid,
				x_max = positions['x'].max(),
				x_min = positions['x'].min(),
				y_max = positions['y'].max(),
				y_min = positions['y'].min(),
			),
			metadata = dict(
				use_DUTs_as_trigger = use_DUTs_as_trigger,
				DUT_ROI_margin = DUT_ROI_margin,
			),
			do_quiver_plot = True,
		)
	
	a

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
		'--force',
		help = 'If this flag is passed, it will force whatever has to be done, meaning old data will be deleted.',
		required = False,
		dest = 'force',
		action = 'store_true'
	)
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = Path,
	)
	parser.add_argument('--setup_analysis_for_DUT',
		metavar = 'DUT_name', 
		help = 'Name of the DUT name for which to setup a new analysis.',
		required = False,
		dest = 'setup_analysis_for_DUT',
		type = str,
		default = 'None',
	)
	parser.add_argument(
		'--plot_distributions',
		help = 'Pass this flag to run `plot_distributions`.',
		required = False,
		dest = 'plot_distributions',
		action = 'store_true'
	)
	parser.add_argument(
		'--plot_tracks_and_hits',
		help = 'Pass this flag to run `plot_tracks_and_hits`.',
		required = False,
		dest = 'plot_tracks_and_hits',
		action = 'store_true'
	)
	parser.add_argument(
		'--transformation_for_centering_and_leveling',
		help = 'Pass this flag to run `plot_cluster_size`.',
		required = False,
		dest = 'transformation_for_centering_and_leveling',
		action = 'store_true'
	)
	parser.add_argument(
		'--plot_cluster_size',
		help = 'Pass this flag to run `plot_cluster_size`.',
		required = False,
		dest = 'plot_cluster_size',
		action = 'store_true'
	)
	parser.add_argument(
		'--plot_features',
		help = 'Pass this flag to run `plot_features`.',
		required = False,
		dest = 'plot_features',
		action = 'store_true'
	)
	parser.add_argument(
		'--plot_waveforms_distributions_inside_ROI',
		help = 'Pass this flag to run `plot_waveforms_distributions_inside_ROI`.',
		required = False,
		dest = 'plot_waveforms_distributions_inside_ROI',
		action = 'store_true'
	)
	parser.add_argument(
		'--position_reconstruction_with_charge_imbalance',
		help = 'Pass this flag to run `position_reconstruction_with_charge_imbalance`.',
		required = False,
		dest = 'position_reconstruction_with_charge_imbalance',
		action = 'store_true'
	)
	parser.add_argument(
		'--train_reconstructors',
		help = 'Pass this flag to run `train_reconstructors`.',
		required = False,
		dest = 'train_reconstructors',
		action = 'store_true'
	)
	parser.add_argument(
		'--trigger_DUTs',
		help = 'A list of DUT names and pixels, present in the same batch, to add to the trigger line. For example `AC\ 11\ (0,0)`',
		required = False,
		dest = 'trigger_DUTs',
		nargs = '+', # So it parses a list of things.
		default = None,
	)
	parser.add_argument(
		'--DUT_ROI_margin',
		help = 'A float number (in meters) to be used as the margin for the ROI size.',
		required = False,
		dest = 'DUT_ROI_margin',
		default = None,
		type = float,
	)
	args = parser.parse_args()
	
	_bureaucrat = RunBureaucrat(args.directory)
	if _bureaucrat.was_task_run_successfully('this_is_an_RSD-LGAD_analysis'):
		if args.plot_distributions == True:
			plot_distributions(_bureaucrat, force=args.force)
		if args.plot_tracks_and_hits == True:
			plot_tracks_and_hits(_bureaucrat, force=args.force)
		if args.transformation_for_centering_and_leveling == True:
			transformation_for_centering_and_leveling(_bureaucrat, force=args.force)
		if args.plot_cluster_size == True:
			plot_cluster_size(_bureaucrat, force=args.force, use_DUTs_as_trigger=args.trigger_DUTs, DUT_ROI_margin=args.DUT_ROI_margin)
		if args.plot_features == True:
			plot_features(_bureaucrat, force=args.force, use_DUTs_as_trigger=args.trigger_DUTs, DUT_ROI_margin=args.DUT_ROI_margin)
		if args.position_reconstruction_with_charge_imbalance == True:
			position_reconstruction_with_charge_imbalance(_bureaucrat, force=args.force, use_DUTs_as_trigger=args.trigger_DUTs, DUT_ROI_margin=args.DUT_ROI_margin)
		if args.plot_waveforms_distributions_inside_ROI == True:
			plot_waveforms_distributions_inside_ROI(_bureaucrat, force=args.force, DUT_ROI_margin=args.DUT_ROI_margin)
		if args.train_reconstructors == True:
			train_reconstructors(_bureaucrat, force=args.force, use_DUTs_as_trigger=args.trigger_DUTs, DUT_ROI_margin=args.DUT_ROI_margin)
	elif _bureaucrat.was_task_run_successfully('batch_info') and args.setup_analysis_for_DUT is not None:
		setup_RSD_LGAD_analysis_within_batch(batch=_bureaucrat, DUT_name=args.setup_analysis_for_DUT)
	else:
		raise RuntimeError(f"Don't know what to do in {_bureaucrat.path_to_run_directory}... Please read script help or source code.")
	
