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

def load_tracks(RSD_analysis:RunBureaucrat, DUT_z_position:float, use_DUTs_as_trigger:list=None):
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

def calculate_features(data:pandas.DataFrame):
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
	_ = {}
	_['ASF'] = amplitude_shared_fraction
	_['CSF'] = charge_shared_fraction
	_['amplitude_imbalance'] = amplitude_imbalance
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
		
		for feature_name in ['amplitude_imbalance']:
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
	elif _bureaucrat.was_task_run_successfully('batch_info') and args.setup_analysis_for_DUT is not None:
		setup_RSD_LGAD_analysis_within_batch(batch=_bureaucrat, DUT_name=args.setup_analysis_for_DUT)
	else:
		raise RuntimeError(f"Don't know what to do in {_bureaucrat.path_to_run_directory}... Please read script help or source code.")
	
