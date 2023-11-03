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
import json
import warnings
from parse_waveforms import read_parsed_from_waveforms_from_batch
import multiprocessing
from uncertainties import ufloat
from scipy.interpolate import interp1d

def load_analyses_config():
	logging.info(f'Reading analyses config from the cloud...')
	analyses = pandas.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vTaR20eM5ZQxtizmZiaAtHooE7hWYfSixSgc1HD5sVNZT_RNxZKmhI09wCEtXEVepjM8NB1n8BUBZnc/pub?gid=0&single=true&output=csv').set_index(['test_beam_campaign','batch_name','DUT_name']).query('DUT_type=="TI-LGAD"')
	return analyses

def load_this_TILGAD_analysis_config(TI_LGAD_analysis:RunBureaucrat):
	TB_batch = TI_LGAD_analysis.parent
	TB_campaign = TB_batch.parent
	analysis_config = load_analyses_config()
	return analysis_config.loc[(TB_campaign.run_name,TB_batch.run_name,TI_LGAD_analysis.run_name)]

def project_tracks(tracks:pandas.DataFrame, z:float):
	projected = utils.project_track_in_z(
		A = tracks[[f'A{_}' for _ in ['x','y','z']]].to_numpy().T,
		B = tracks[[f'B{_}' for _ in ['x','y','z']]].to_numpy().T,
		z = z,
	).T
	return pandas.DataFrame(
		projected,
		columns = ['Px','Py','Pz'],
		index = tracks.index,
	)

def load_tracks(TI_LGAD_analysis:RunBureaucrat, DUT_z_position:float):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully('this_is_a_TI-LGAD_analysis')
	batch = TI_LGAD_analysis.parent
	
	tracks = load_tracks_from_batch(batch, only_multiplicity_one=True)
	tracks = tracks.join(project_tracks(tracks=tracks, z=DUT_z_position))
	
	return tracks
	
def load_hits(TI_LGAD_analysis:RunBureaucrat, DUT_hit_criterion:str):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully('this_is_a_TI-LGAD_analysis')
	batch = TI_LGAD_analysis.parent
	
	DUT_hits = read_parsed_from_waveforms_from_batch(
		batch = batch,
		DUT_name = TI_LGAD_analysis.run_name,
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

# Tasks ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

def setup_TI_LGAD_analysis_within_batch(batch:RunBureaucrat, DUT_name:str)->RunBureaucrat:
	"""Setup a directory structure to perform further analysis of a TI-LGAD
	that is inside a batch pointed by `batch`. This should be the 
	first step before starting a TI-LGAD analysis."""
	batch.check_these_tasks_were_run_successfully(['runs','batch_info'])
	
	with batch.handle_task('TI-LGADs_analyses', drop_old_data=False) as employee:
		setup_configuration_info = utils.load_setup_configuration_info(batch)
		if DUT_name not in set(setup_configuration_info['DUT_name']):
			raise RuntimeError(f'DUT_name {repr(DUT_name)} not present within the set of DUTs in {batch.pseudopath}, which is {set(setup_configuration_info["DUT_name"])}')
		
		try:
			TILGAD_bureaucrat = employee.create_subrun(DUT_name)
			with TILGAD_bureaucrat.handle_task('this_is_a_TI-LGAD_analysis'):
				pass
			logging.info(f'Directory structure for TI-LGAD analysis was created in {TILGAD_bureaucrat.pseudopath} ')
		except RuntimeError as e: # This will happen if the run already existed beforehand.
			if 'Cannot create run' in str(e):
				TILGAD_bureaucrat = [b for b in batch.list_subruns_of_task('TI-LGADs_analyses') if b.run_name==DUT_name][0]
		
	return TILGAD_bureaucrat

def plot_DUT_distributions(TI_LGAD_analysis_bureaucrat:RunBureaucrat, force:bool=False):
	"""Plot some raw distributions, useful to configure cuts and thresholds
	for further steps in the analysis."""
	MAXIMUM_NUMBER_OF_EVENTS = 9999
	TI_LGAD_analysis_bureaucrat.check_these_tasks_were_run_successfully('this_is_a_TI-LGAD_analysis') # To be sure we are inside what is supposed to be a TI-LGAD analysis.
	
	if force==False and TI_LGAD_analysis_bureaucrat.was_task_run_successfully('plot_distributions'):
		return
	
	with TI_LGAD_analysis_bureaucrat.handle_task('plot_distributions') as employee:
		setup_config = utils.load_setup_configuration_info(TI_LGAD_analysis_bureaucrat.parent)
		
		save_distributions_plots_here = employee.path_to_directory_of_my_task/'distributions'
		save_distributions_plots_here.mkdir()
		for variable in ['Amplitude (V)','t_50 (s)','Noise (V)','Time over 50% (s)',]:
			logging.info(f'Plotting {variable} distribution...')
			data = read_parsed_from_waveforms_from_batch(
				batch = TI_LGAD_analysis_bureaucrat.parent,
				DUT_name = TI_LGAD_analysis_bureaucrat.run_name,
				variables = [variable],
				n_events = MAXIMUM_NUMBER_OF_EVENTS,
			)
			data = data.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])['DUT_name_rowcol'])
			fig = px.ecdf(
				data.sort_values('DUT_name_rowcol'),
				title = f'{variable} distribution<br><sup>{TI_LGAD_analysis_bureaucrat.pseudopath}</sup>',
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
				batch = TI_LGAD_analysis_bureaucrat.parent,
				DUT_name = TI_LGAD_analysis_bureaucrat.run_name,
				variables = [x,y],
				n_events = MAXIMUM_NUMBER_OF_EVENTS,
			)
			data = data.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])['DUT_name_rowcol'])
			fig = px.scatter(
				data.sort_values('DUT_name_rowcol').reset_index(drop=False),
				title = f'{y} vs {x} scatter plot<br><sup>{TI_LGAD_analysis_bureaucrat.pseudopath}</sup>',
				x = x,
				y = y,
				color = 'DUT_name_rowcol',
				hover_data = ['n_run','n_event'],
			)
			fig.write_html(
				save_scatter_plots_here/f'{y}_vs_{x}_scatter.html',
				include_plotlyjs = 'cdn',
			)

def plot_tracks_and_hits(TI_LGAD_analysis:RunBureaucrat, do_3D_plot:bool=True, force:bool=False):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully('this_is_a_TI-LGAD_analysis')
	
	if force==False and TI_LGAD_analysis.was_task_run_successfully('plot_tracks_and_hits'):
		return
	
	with TI_LGAD_analysis.handle_task('plot_tracks_and_hits') as employee:
		batch = TI_LGAD_analysis.parent
		
		analysis_config = load_this_TILGAD_analysis_config(TI_LGAD_analysis)
		
		tracks = load_tracks_from_batch(batch, only_multiplicity_one=True)
		
		logging.info('Projecting tracks onto DUT...')
		projected = utils.project_track_in_z(
			A = tracks[[f'A{_}' for _ in ['x','y','z']]].to_numpy().T,
			B = tracks[[f'B{_}' for _ in ['x','y','z']]].to_numpy().T,
			z = analysis_config['DUT_z_position'],
		).T
		projected = pandas.DataFrame(
			projected,
			columns = ['Px','Py','Pz'],
			index = tracks.index,
		)
		tracks = tracks.join(projected)
		
		DUT_hits = read_parsed_from_waveforms_from_batch(
			batch = batch,
			DUT_name = TI_LGAD_analysis.run_name,
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
			title = f'Tracks projected on the DUT<br><sup>{TI_LGAD_analysis.pseudopath}</sup>',
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
				title = f'Tracks<br><sup>{TI_LGAD_analysis.pseudopath}</sup>',
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

def plot_cluster_size(TI_LGAD_analysis:RunBureaucrat, force:bool=False):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully('this_is_a_TI-LGAD_analysis')
	TASK_NAME = 'plot_cluster_size'
	
	if force==False and TI_LGAD_analysis.was_task_run_successfully(TASK_NAME):
		return
	
	with TI_LGAD_analysis.handle_task(TASK_NAME) as employee:
		analysis_config = load_this_TILGAD_analysis_config(TI_LGAD_analysis)
		
		tracks = load_tracks(
			TI_LGAD_analysis = TI_LGAD_analysis,
			DUT_z_position = analysis_config['DUT_z_position'],
		)
		hits = load_hits(
			TI_LGAD_analysis = TI_LGAD_analysis,
			DUT_hit_criterion = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{analysis_config['Amplitude threshold (V)']}",
		)
		
		cluster_size = hits.sum(axis=1)
		cluster_size.name = 'cluster_size'
		
		tracks = tracks.join(cluster_size)
		tracks['cluster_size'].fillna(0, inplace=True)
		tracks['cluster_size'] = tracks['cluster_size'].astype(int)
		
		fig = px.scatter(
			data_frame = tracks.reset_index().sort_values('cluster_size').astype({'cluster_size':str}),
			title = f'Cluster size<br><sup>{TI_LGAD_analysis.pseudopath}</sup>',
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
		fig.write_html(
			employee.path_to_directory_of_my_task/'cluster_size.html',
			include_plotlyjs = 'cdn',
		)
		
		fig = px.histogram(
			tracks,
			x =  'cluster_size',
			title = f'Cluster size<br><sup>{TI_LGAD_analysis.pseudopath}</sup>',
			text_auto = True,
		)
		fig.update_yaxes(type="log")
		fig.write_html(
			employee.path_to_directory_of_my_task/'cluster_size_histogram.html',
			include_plotlyjs = 'cdn',
		)
		a

def translate_and_then_rotate(points:pandas.DataFrame, x_translation:float, y_translation:float, angle_rotation:float):
	"""Apply a translation followed by a rotation to the points.
	
	Arguments
	---------
	points: pandas.DataFrame
		A data frame of the form 
	"""
	for xy,translation in {'x':x_translation,'y':y_translation}.items():
			points[xy] += translation
	r = (points['x']**2 + points['y']**2)**.5
	phi = numpy.arctan2(points['y'], points['x'])
	points['x'], points['y'] = r*numpy.cos(phi+angle_rotation), r*numpy.sin(phi+angle_rotation)
	return points

def transformation_for_centering_and_leveling(TI_LGAD_analysis:RunBureaucrat, draw_square:bool=True, force:bool=False):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully('this_is_a_TI-LGAD_analysis')
	
	if force==False and TI_LGAD_analysis.was_task_run_successfully('transformation_for_centering_and_leveling'):
		return
	
	with TI_LGAD_analysis.handle_task('transformation_for_centering_and_leveling') as employee:
		batch = TI_LGAD_analysis.parent
		
		analysis_config = load_this_TILGAD_analysis_config(TI_LGAD_analysis)
		__ = {'x_translation','y_translation','rotation_around_z_deg'}
		if any([numpy.isnan(analysis_config[_]) for _ in __]):
			raise RuntimeError(f'One (or more) of {__} is `NaN`, check the spreadsheet.')
		
		tracks = load_tracks_from_batch(batch, only_multiplicity_one=True)
		
		logging.info('Projecting tracks onto DUT...')
		projected = project_tracks(tracks, z=analysis_config['DUT_z_position'])
		tracks = tracks.join(projected)
		
		DUT_hits = read_parsed_from_waveforms_from_batch(
			batch = batch,
			DUT_name = TI_LGAD_analysis.run_name,
			variables = [], # No need for variables, only need to know which ones are hits.
			additional_SQL_selection = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{analysis_config['Amplitude threshold (V)']}",
		)
		
		setup_config = utils.load_setup_configuration_info(batch)
		
		DUT_hits = DUT_hits.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])['DUT_name_rowcol'])
		
		tracks = tracks.join(DUT_hits['DUT_name_rowcol'])
		tracks['DUT_name_rowcol'] = tracks['DUT_name_rowcol'].fillna('no hit')
		tracks = tracks.sort_values('DUT_name_rowcol', ascending=False)
		
		logging.info('Applying transformation to tracks to center and align DUT...')
		tracks[['Px','Py']] = translate_and_then_rotate(
			points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
			x_translation = analysis_config['x_translation'],
			y_translation = analysis_config['y_translation'],
			angle_rotation = analysis_config['rotation_around_z_deg']/180*numpy.pi,
		)
		
		logging.info('Plotting tracks and hits on DUT...')
		fig = px.scatter(
			tracks.reset_index(),
			title = f'Tracks projected on the DUT after transformation<br><sup>{TI_LGAD_analysis.pseudopath}</sup>',
			x = 'Px',
			y = 'Py',
			color = 'DUT_name_rowcol',
			hover_data = ['n_run','n_event'],
			labels = {
				'Px': 'x (m)',
				'Py': 'y (m)',
			},
		)
		for xy,method in dict(x=fig.add_vline, y=fig.add_hline).items():
			method(0)
		if draw_square:
			fig.add_shape(
				type = "rect",
				x0 = -250e-6, 
				y0 = -250e-6, 
				x1 = 250e-6, 
				y1 = 250e-6,
			)
		fig.update_yaxes(
			scaleanchor = "x",
			scaleratio = 1,
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'tracks_projected_on_DUT.html',
			include_plotlyjs = 'cdn',
		)

def estimate_fraction_of_misreconstructed_tracks(TI_LGAD_analysis:RunBureaucrat, force:bool=False):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully('this_is_a_TI-LGAD_analysis')
	
	if force==False and TI_LGAD_analysis.was_task_run_successfully('estimate_fraction_of_misreconstructed_tracks'):
		return
	
	with TI_LGAD_analysis.handle_task('estimate_fraction_of_misreconstructed_tracks') as employee:
		batch = TI_LGAD_analysis.parent
		
		analysis_config = load_this_TILGAD_analysis_config(TI_LGAD_analysis)
		
		tracks = load_tracks_from_batch(batch, only_multiplicity_one=True)
		
		logging.info('Projecting tracks onto DUT...')
		projected = project_tracks(tracks, z=analysis_config['DUT_z_position'])
		tracks = tracks.join(projected)
		
		DUT_hits = read_parsed_from_waveforms_from_batch(
			batch = batch,
			DUT_name = TI_LGAD_analysis.run_name,
			variables = [], # No need for variables, only need to know which ones are hits.
			additional_SQL_selection = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{analysis_config['Amplitude threshold (V)']}",
		)
		DUT_hits['DUT_hit'] = True
		
		setup_config = utils.load_setup_configuration_info(batch)
		
		tracks = tracks.join(DUT_hits['DUT_hit'])
		tracks['DUT_hit'] = tracks['DUT_hit'].fillna(False)
		
		logging.info('Applying transformation to tracks to center and align DUT...')
		tracks[['Px_transformed','Py_transformed']] = translate_and_then_rotate(
			points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
			x_translation = analysis_config['x_translation'],
			y_translation = analysis_config['y_translation'],
			angle_rotation = analysis_config['rotation_around_z_deg']/180*numpy.pi,
		)
		
		# To estimate the total area, I use the tracks data before the transformation (rotation and translation) in which the total area should be a rectangle aligned with x and y, and then apply the transformation wherever needed.
		total_area_corners = pandas.DataFrame(
			{
				'x': [tracks['Px'].min(), tracks['Px'].max(), tracks['Px'].max(), tracks['Px'].min()],
				'y': [tracks['Py'].min(), tracks['Py'].min(), tracks['Py'].max(), tracks['Py'].max()],
				'which_corner': ['bottom_left','bottom_right','top_right','top_left'],
			},
		).set_index('which_corner')
		total_area_corners[['x_transformed','y_transformed']] = translate_and_then_rotate(
			points = total_area_corners,
			x_translation = analysis_config['x_translation'],
			y_translation = analysis_config['y_translation'],
			angle_rotation = analysis_config['rotation_around_z_deg']/180*numpy.pi/4,
		)
		
		logging.info(f'Calculating probability that corry fails...')
		data = []
		for DUT_ROI_size in numpy.linspace(111e-6,2222e-6,33):
			tracks_for_which_DUT_has_a_signal = tracks.query('DUT_hit==True')
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore")
				# The previous is to get rid of the annoying warning "A value is trying to be set on a copy of a slice from a DataFrame."
				tracks_for_which_DUT_has_a_signal['is_inside_DUT_ROI'] = False # Initialize.
				tracks_for_which_DUT_has_a_signal.loc[(tracks_for_which_DUT_has_a_signal['Px_transformed']>=-DUT_ROI_size/2) & (tracks_for_which_DUT_has_a_signal['Px_transformed']<DUT_ROI_size/2) & (tracks_for_which_DUT_has_a_signal['Py_transformed']>=-DUT_ROI_size/2) & (tracks_for_which_DUT_has_a_signal['Py_transformed']<DUT_ROI_size/2), 'is_inside_DUT_ROI'] = True
			n_tracks_outside_DUT = len(tracks_for_which_DUT_has_a_signal.query('is_inside_DUT_ROI==False'))
			n_tracks = len(tracks_for_which_DUT_has_a_signal)
			n_tracks_within_DUT = len(tracks_for_which_DUT_has_a_signal.query('is_inside_DUT_ROI==True'))
			total_area = (total_area_corners.loc['bottom_right','x']-total_area_corners.loc['bottom_left','x'])*(total_area_corners.loc['top_left','y']-total_area_corners.loc['bottom_left','y'])
			DUT_area = DUT_ROI_size**2
			
			probability_corry_fails = ((ufloat(n_tracks_within_DUT,n_tracks_within_DUT**.5)/ufloat(n_tracks_outside_DUT,n_tracks_within_DUT**.5) - DUT_area/(total_area-DUT_area))*(total_area-DUT_area)/total_area+1)**-1
			
			data.append(
				{
					'DUT_ROI_size (m)': DUT_ROI_size,
					'probability_corry_fails': probability_corry_fails.nominal_value,
					'probability_corry_fails error': probability_corry_fails.std_dev,
				}
			)
			
			############################################################
			############################################################
			if True: # This `if True` is simply so I can fold the code block.
				save_these_plots_here = employee.path_to_directory_of_my_task/'tracks'
				save_these_plots_here.mkdir(exist_ok = True)
				df = tracks_for_which_DUT_has_a_signal.sort_values('is_inside_DUT_ROI')
				graph_dimensions = dict(
					color = 'is_inside_DUT_ROI',
				)
				labels = {
					'Px_transformed': 'x (m)',
					'Py_transformed': 'y (m)',
				}
				fig = px.scatter(
					df.reset_index(drop=False),
					title = f'Tracks that hit the DUT<br><sup>{TI_LGAD_analysis.pseudopath}, ROI size = {DUT_ROI_size*1e6:.0f} µm</sup>',
					x = 'Px_transformed',
					y = 'Py_transformed',
					hover_data = ['n_run','n_event'],
					labels = labels,
					**graph_dimensions
				)
				# ~ plotly_utils.add_grouped_legend(fig=fig, data_frame=df, x='Px_transformed', graph_dimensions=graph_dimensions, labels=labels)
				fig.update_yaxes(
					scaleanchor = "x",
					scaleratio = 1,
				)
				for xy,method in dict(x=fig.add_vline, y=fig.add_hline).items():
					method(0)
				fig.add_shape(
					type = "rect",
					x0 = -250e-6, 
					y0 = -250e-6, 
					x1 = 250e-6, 
					y1 = 250e-6,
				)
				fig.add_shape(
					type = "rect",
					x0 = -DUT_ROI_size/2,
					y0 = -DUT_ROI_size/2,
					x1 = DUT_ROI_size/2,
					y1 = DUT_ROI_size/2,
					line=dict(
						dash = "dash",
					),
				)
				fig.add_trace(
					go.Scatter(
						x = [total_area_corners.loc[corner,'x_transformed'] for corner in total_area_corners.index.get_level_values('which_corner')] + [total_area_corners.loc[total_area_corners.index.get_level_values('which_corner')[0],'x_transformed']],
						y = [total_area_corners.loc[corner,'y_transformed'] for corner in total_area_corners.index.get_level_values('which_corner')] + [total_area_corners.loc[total_area_corners.index.get_level_values('which_corner')[0],'y_transformed']],
						mode = 'lines',
						line = dict(
							color = 'black',
						),
						showlegend = False,
						hoverinfo = 'skip',
					)
				)
				fig.write_html(
					save_these_plots_here/f'DUT_hits_when_ROI_{DUT_ROI_size*1e6:.0f}um.html',
					include_plotlyjs = 'cdn',
				)
			############################################################
			############################################################
			
		data = pandas.DataFrame.from_records(data)
		
		dydx = data['probability_corry_fails'].diff()/data['DUT_ROI_size (m)'].diff()
		probability_corry_fails_final_value = numpy.mean([ufloat(_['probability_corry_fails'],_['probability_corry_fails error']) for i,_ in data.loc[dydx**2<22**2].iterrows()])
		
		utils.save_dataframe(
			df = pandas.Series(
				dict(
					probability_corry_fails = probability_corry_fails_final_value.nominal_value,
					probability_corry_fails_error = probability_corry_fails_final_value.std_dev,
				),
				name = 'probability',
			),
			name = 'probability_that_corry_fails',
			location = employee.path_to_directory_of_my_task,
		)
		
		fig = px.line(
			data_frame = data,
			title = f'Estimation of probability track reconstruction error<br><sup>{TI_LGAD_analysis.pseudopath}</sup>',
			x = 'DUT_ROI_size (m)',
			y = 'probability_corry_fails',
			error_y = 'probability_corry_fails error',
			markers = True,
			labels = {
				'probability_corry_fails': 'Probability that corry fails',
				'DUT_ROI_size (m)': 'DUT ROI size (m)',
			},
		)
		fig.add_hline(
			probability_corry_fails_final_value.nominal_value,
			annotation_text = f'Probability that corry fails = {probability_corry_fails_final_value}',
		)
		fig.add_hrect(
			y0 = probability_corry_fails_final_value.nominal_value - probability_corry_fails_final_value.std_dev,
			y1 = probability_corry_fails_final_value.nominal_value + probability_corry_fails_final_value.std_dev,
			fillcolor = "black",
			opacity = 0.1,
			line_width = 0,
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'probability_of_failure_vs_ROI_size.html',
			include_plotlyjs = 'cdn',
		)

def efficiency_vs_1D_distance_rolling(tracks:pandas.DataFrame, DUT_hits, project_on:str, distances:numpy.array, window_size:float, number_of_noHitTrack_that_are_fake_per_unit_area:float):
	if set(tracks.index.names) != set(DUT_hits.names):
		raise ValueError('The index levels of `tracks` and `DUT_hits` must be the same, they are not.')
	if set(tracks.columns) != {'x','y'}:
		raise ValueError('The columns of `tracks` must be "x" and "y"')
	if project_on not in {'x','y'}:
		raise ValueError('`project_on` must be "x" or "y"')
	
	if project_on == 'x':
		window_thickness = tracks['y'].max() - tracks['y'].min()
	else:
		window_thickness = tracks['x'].max() - tracks['x'].min()
	window_area = window_size*window_thickness
	number_of_noHitTrack_that_are_fake_per_window = window_area*number_of_noHitTrack_that_are_fake_per_unit_area
	
	total_tracks_count = distances*0
	DUT_hits_count = distances*0
	for i,d in enumerate(distances):
		total_tracks_count[i] = len(tracks.query(f'{d-window_size/2}<={project_on} and {project_on}<{d+window_size/2}'))
		DUT_hits_count[i] = len(utils.select_by_multiindex(tracks, DUT_hits).query(f'{d-window_size/2}<={project_on} and {project_on}<{d+window_size/2}'))
	return DUT_hits_count/(total_tracks_count-number_of_noHitTrack_that_are_fake_per_window)

def efficiency_vs_1D_distance_rolling_error_estimation(tracks:pandas.DataFrame, DUT_hits, project_on:str, distances:numpy.array, window_size:float, number_of_noHitTrack_that_are_fake_per_unit_area:float, number_of_noHitTrack_that_are_fake_per_unit_area_uncertainty:float, n_bootstraps:int=99, confidence_level:float=.68):
	replicas = []
	for n_bootstrap in range(n_bootstraps):
		efficiency = efficiency_vs_1D_distance_rolling(
			tracks = tracks.sample(frac=1, replace=True),
			DUT_hits = DUT_hits,
			project_on = project_on,
			distances = distances,
			window_size = window_size,
			number_of_noHitTrack_that_are_fake_per_unit_area = number_of_noHitTrack_that_are_fake_per_unit_area + numpy.random.randn()*number_of_noHitTrack_that_are_fake_per_unit_area_uncertainty,
		)
		replicas.append(efficiency)
	replicas = numpy.array(replicas)
	value = numpy.quantile(replicas, q=.5, axis=0, method='interpolated_inverted_cdf')
	error_up = numpy.quantile(replicas, q=.5+confidence_level/2, axis=0, method='interpolated_inverted_cdf') - value
	error_down = value - numpy.quantile(replicas, q=.5-confidence_level/2, axis=0, method='interpolated_inverted_cdf')
	
	return error_down, error_up

def efficiency_vs_distance_calculation(TI_LGAD_analysis:RunBureaucrat, force:bool=False):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully(['this_is_a_TI-LGAD_analysis','estimate_fraction_of_misreconstructed_tracks'])
	
	TASK_NAME = 'efficiency_vs_distance_calculation'
	
	if force == False and TI_LGAD_analysis.was_task_run_successfully(TASK_NAME):
		return
	
	with TI_LGAD_analysis.handle_task(TASK_NAME) as employee:
		batch = TI_LGAD_analysis.parent
		
		# Read data ---
		analysis_config = load_this_TILGAD_analysis_config(TI_LGAD_analysis)
		setup_config = utils.load_setup_configuration_info(batch)
		
		tracks = load_tracks_from_batch(batch, only_multiplicity_one=True)
		tracks.reset_index('n_track', inplace=True) # We are loading with multiplicity one, so this is not needed anymore.
		
		DUT_hits = read_parsed_from_waveforms_from_batch(
			batch = batch,
			DUT_name = TI_LGAD_analysis.run_name,
			variables = [], # No need for variables, only need to know which ones are hits.
			additional_SQL_selection = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{analysis_config['Amplitude threshold (V)']}",
		)
		
		probability_corry_fails = pandas.read_pickle(TI_LGAD_analysis.path_to_directory_of_task('estimate_fraction_of_misreconstructed_tracks')/'probability_that_corry_fails.pickle')
		probability_corry_fails = ufloat(probability_corry_fails['probability_corry_fails'],probability_corry_fails['probability_corry_fails_error'])
		
		# Now do some pre processing ---
		logging.info('Projecting tracks onto DUT...')
		projected = utils.project_track_in_z(
			A = tracks[[f'A{_}' for _ in ['x','y','z']]].to_numpy().T,
			B = tracks[[f'B{_}' for _ in ['x','y','z']]].to_numpy().T,
			z = analysis_config['DUT_z_position'],
		).T
		projected = pandas.DataFrame(
			projected,
			columns = ['Px','Py','Pz'],
			index = tracks.index,
		)
		tracks = tracks.join(projected)
		tracks = tracks[['Px','Py','Pz']] # Keep only relevant columns from now on.
		
		DUT_hits = DUT_hits.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])[['DUT_name_rowcol']])
		DUT_hits.reset_index(['n_CAEN','CAEN_n_channel'], inplace=True, drop=True) # Not used anymore.
		DUT_hits['hit'] = True
		DUT_hits = DUT_hits.set_index('DUT_name_rowcol', append=True).unstack('DUT_name_rowcol', fill_value=False)
		DUT_hits = DUT_hits.droplevel(0, axis=1) # Not used anymore.
		
		logging.info('Applying transformation to tracks to center and align DUT...')
		tracks[['Px_transformed','Py_transformed']] = translate_and_then_rotate(
			points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
			x_translation = analysis_config['x_translation'],
			y_translation = analysis_config['y_translation'],
			angle_rotation = analysis_config['rotation_around_z_deg']/180*numpy.pi,
		)
		
		# To estimate the total area, I use the tracks data before the transformation (rotation and translation) in which the total area should be a rectangle aligned with x and y, and then apply the transformation wherever needed.
		total_area_corners = pandas.DataFrame(
			{
				'x': [tracks['Px'].min(), tracks['Px'].max(), tracks['Px'].max(), tracks['Px'].min()],
				'y': [tracks['Py'].min(), tracks['Py'].min(), tracks['Py'].max(), tracks['Py'].max()],
				'which_corner': ['bottom_left','bottom_right','top_right','top_left'],
			},
		).set_index('which_corner')
		total_area = (total_area_corners.loc['bottom_right','x']-total_area_corners.loc['bottom_left','x'])*(total_area_corners.loc['top_left','y']-total_area_corners.loc['bottom_left','y'])
		
		tracks = tracks.join(pandas.DataFrame({'DUT_hit': [True for _ in range(len(DUT_hits))]}, index=DUT_hits.index))
		tracks['DUT_hit'] = tracks['DUT_hit'].fillna(False)
		
		number_of_noHitTrack_that_are_fake_per_unit_area = (len(tracks.query('DUT_hit == False'))*probability_corry_fails/total_area)
		
		################################################################
		# Some hardcoded stuff #########################################
		################################################################
		PIXEL_SIZE = 250e-6
		ROI_DISTANCE_OFFSET = 88e-6
		ROI_WIDTH = PIXEL_SIZE/3
		PIXEL_DEPENDENT_SETTINGS = {
			'top_row': dict(
				project_on = 'x',
				ROI = dict(
					x_min = -PIXEL_SIZE-ROI_DISTANCE_OFFSET,
					x_max = PIXEL_SIZE+ROI_DISTANCE_OFFSET,
					y_min = PIXEL_SIZE/2-ROI_WIDTH/2,
					y_max = PIXEL_SIZE/2+ROI_WIDTH/2,
				),
				left_pixel_rowcol = [0,0],
				right_pixel_rowcol = [0,1],
			),
			'bottom_row': dict(
				project_on = 'x',
				ROI = dict(
					x_min = -PIXEL_SIZE-ROI_DISTANCE_OFFSET,
					x_max = PIXEL_SIZE+ROI_DISTANCE_OFFSET,
					y_min = -PIXEL_SIZE/2-ROI_WIDTH/2,
					y_max = -PIXEL_SIZE/2+ROI_WIDTH/2,
					
				),
				left_pixel_rowcol = [1,0],
				right_pixel_rowcol = [1,1],
			),
			'left_column': dict(
				project_on = 'y',
				ROI = dict(
					y_min = -PIXEL_SIZE-ROI_DISTANCE_OFFSET,
					y_max = PIXEL_SIZE+ROI_DISTANCE_OFFSET,
					x_min = -PIXEL_SIZE/2-ROI_WIDTH/2,
					x_max = -PIXEL_SIZE/2+ROI_WIDTH/2,
				),
				left_pixel_rowcol = [1,0],
				right_pixel_rowcol = [0,0],
			),
			'right_column': dict(
				project_on = 'y',
				ROI = dict(
					y_min = -PIXEL_SIZE-ROI_DISTANCE_OFFSET,
					y_max = PIXEL_SIZE+ROI_DISTANCE_OFFSET,
					x_min = PIXEL_SIZE/2-ROI_WIDTH/2,
					x_max = PIXEL_SIZE/2+ROI_WIDTH/2,
				),
				left_pixel_rowcol = [1,1],
				right_pixel_rowcol = [0,1],
			),
		}
		CALCULATION_STEP = 11e-6
		ROLLING_WINDOW_SIZE = 44e-6
		################################################################
		################################################################
		################################################################
		
		for which_pixels in PIXEL_DEPENDENT_SETTINGS.keys():
			if analysis_config[which_pixels] == False:
				continue
			which_pixels_analysis = employee.create_subrun(which_pixels)
			with which_pixels_analysis.handle_task('efficiency_vs_distance') as which_pixels_employee:
				logging.info(f'Proceeding with {which_pixels_analysis.pseudopath}...')
				xmin = PIXEL_DEPENDENT_SETTINGS[which_pixels]['ROI']['x_min']
				xmax = PIXEL_DEPENDENT_SETTINGS[which_pixels]['ROI']['x_max']
				ymin = PIXEL_DEPENDENT_SETTINGS[which_pixels]['ROI']['y_min']
				ymax = PIXEL_DEPENDENT_SETTINGS[which_pixels]['ROI']['y_max']
				project_on = PIXEL_DEPENDENT_SETTINGS[which_pixels]['project_on']
				
				tracks_for_efficiency_calculation = tracks.query(f'{xmin}<Px_transformed and Px_transformed<{xmax} and {ymin}<Py_transformed and Py_transformed<{ymax}')
				
				tracks_for_efficiency_calculation = tracks_for_efficiency_calculation.join(DUT_hits).fillna(False)
				tracks_for_efficiency_calculation['have_hit'] = tracks_for_efficiency_calculation[DUT_hits.columns].apply(lambda row: ' '.join(row[row==True].keys()), axis=1)
				tracks_for_efficiency_calculation.loc[tracks_for_efficiency_calculation['have_hit']=='','have_hit'] = 'no hit'
				
				if True:
					logging.info('Plotting tracks used for efficiency calculation...')
					fig = px.scatter(
						tracks_for_efficiency_calculation.reset_index().sort_values('have_hit'),
						title = f'Tracks projected on the DUT after transformation<br><sup>{which_pixels_analysis.pseudopath}</sup>',
						x = 'Px_transformed',
						y = 'Py_transformed',
						color = 'have_hit',
						hover_data = ['n_run','n_event'],
						labels = {
							'Px_transformeds': 'x (m)',
							'Py_transformed': 'y (m)',
						},
					)
					fig.add_shape(
						type = "rect",
						x0 = xmin, 
						y0 = ymin, 
						x1 = xmax, 
						y1 = ymax,
						line=dict(color="black"),
					)
					fig.update_yaxes(
						scaleanchor = "x",
						scaleratio = 1,
					)
					fig.write_html(
						which_pixels_employee.path_to_directory_of_my_task/'tracks_projected_on_DUT.html',
						include_plotlyjs = 'cdn',
					)
				
				logging.info('Calculating efficiency vs distance...')
				tracks_for_efficiency_calculation.sort_values(f'P{project_on}_transformed', inplace=True)
				distance_axis = numpy.arange(
					start = tracks_for_efficiency_calculation[f'P{project_on}_transformed'].min(),
					stop = tracks_for_efficiency_calculation[f'P{project_on}_transformed'].max(),
					step = CALCULATION_STEP,
				)
				efficiency_data = []
				pixel_names = {leftright: f'{TI_LGAD_analysis.run_name} ({PIXEL_DEPENDENT_SETTINGS[which_pixels][f"{leftright}_pixel_rowcol"][0]},{PIXEL_DEPENDENT_SETTINGS[which_pixels][f"{leftright}_pixel_rowcol"][1]})' for leftright in ['left','right']}
				for leftright in ['left','right','both']:
					if leftright in {'left','right'}:
						DUT_hits_for_efficiency = DUT_hits.query(f'`{pixel_names[leftright]}` == True')
					elif leftright == 'both':
						DUT_hits_for_efficiency = DUT_hits.query(f'`{pixel_names["left"]}` == True or `{pixel_names["right"]}` == True')
					else:
						raise RuntimeError('Check this, should never happen!')
					
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						efficiency_calculation_args = dict(
							tracks = tracks_for_efficiency_calculation.rename(columns={'Px_transformed':'x', 'Py_transformed':'y'})[['x','y']],
							DUT_hits = DUT_hits_for_efficiency.index,
							project_on = project_on,
							distances = distance_axis,
							window_size = ROLLING_WINDOW_SIZE,
							number_of_noHitTrack_that_are_fake_per_unit_area = number_of_noHitTrack_that_are_fake_per_unit_area.nominal_value,
						)
						error_minus, error_plus = efficiency_vs_1D_distance_rolling_error_estimation(
							**efficiency_calculation_args,
							number_of_noHitTrack_that_are_fake_per_unit_area_uncertainty = number_of_noHitTrack_that_are_fake_per_unit_area.std_dev,
							n_bootstraps = 33,
						)
					df = pandas.DataFrame(
						{
							'Distance (m)': distance_axis,
							'Efficiency': efficiency_vs_1D_distance_rolling(**efficiency_calculation_args),
							'Efficiency error_-': error_minus,
							'Efficiency error_+': error_plus,
						}
					)
					df['Pixel'] = leftright
					efficiency_data.append(df)
				efficiency_data = pandas.concat(efficiency_data)
				efficiency_data.set_index(['Pixel','Distance (m)'], inplace=True)
				
				utils.save_dataframe(
					df = efficiency_data,
					name = 'efficiency_vs_distance',
					location = which_pixels_employee.path_to_directory_of_my_task,
				)
				
				logging.info('Plotting efficiency vs distance...')
				fig = plotly_utils.line(
					data_frame = efficiency_data.sort_index().reset_index(drop=False),
					title = f'Efficiency vs distance<br><sup>{which_pixels_analysis.pseudopath}</sup>',
					x = 'Distance (m)',
					y = 'Efficiency',
					error_y = 'Efficiency error_+',
					error_y_minus = 'Efficiency error_-',
					color = 'Pixel',
					error_y_mode = 'bands',
				)
				fig.write_html(
					which_pixels_employee.path_to_directory_of_my_task/'efficiency_vs_distance.html',
					include_plotlyjs = 'cdn',
				)

def efficiency_vs_distance_calculation_usin_coincidence_with_other_sensor(TI_LGAD_analysis:RunBureaucrat, control_DUT_name:str, control_DUT_pixels:list, control_DUT_amplitude_thresohld:float, force:bool=False):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully(['this_is_a_TI-LGAD_analysis'])
	
	TASK_NAME = 'efficiency_vs_distance_calculation_using_coincidence_with_another'
	
	if force == False and TI_LGAD_analysis.was_task_run_successfully(TASK_NAME):
		return
	
	with TI_LGAD_analysis.handle_task(TASK_NAME) as employee:
		batch = TI_LGAD_analysis.parent
		
		# Read data ---
		analysis_config = load_this_TILGAD_analysis_config(TI_LGAD_analysis)
		setup_config = utils.load_setup_configuration_info(batch)
		
		tracks = load_tracks_from_batch(batch, only_multiplicity_one=True)
		tracks.reset_index('n_track', inplace=True) # We are loading with multiplicity one, so this is not needed anymore.
		
		DUT_hits = read_parsed_from_waveforms_from_batch(
			batch = batch,
			DUT_name = TI_LGAD_analysis.run_name,
			variables = [], # No need for variables, only need to know which ones are hits.
			additional_SQL_selection = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{analysis_config['Amplitude threshold (V)']}",
		)
		
		control_DUT_hits = read_parsed_from_waveforms_from_batch(
			batch = batch,
			DUT_name = control_DUT_name,
			variables = [], # No need for variables, only need to know which ones are hits.
			additional_SQL_selection = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{-abs(control_DUT_amplitude_thresohld)}",
		)
		
		# Now do some pre processing ---
		logging.info('Projecting tracks onto DUT...')
		projected = utils.project_track_in_z(
			A = tracks[[f'A{_}' for _ in ['x','y','z']]].to_numpy().T,
			B = tracks[[f'B{_}' for _ in ['x','y','z']]].to_numpy().T,
			z = analysis_config['DUT_z_position'],
		).T
		projected = pandas.DataFrame(
			projected,
			columns = ['Px','Py','Pz'],
			index = tracks.index,
		)
		tracks = tracks.join(projected)
		tracks = tracks[['Px','Py','Pz']] # Keep only relevant columns from now on.
		
		DUT_hits = DUT_hits.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])[['DUT_name_rowcol']])
		DUT_hits.reset_index(['n_CAEN','CAEN_n_channel'], inplace=True, drop=True) # Not used anymore.
		DUT_hits['hit'] = True
		DUT_hits = DUT_hits.set_index('DUT_name_rowcol', append=True).unstack('DUT_name_rowcol', fill_value=False)
		DUT_hits = DUT_hits.droplevel(0, axis=1) # Not used anymore.
		
		control_DUT_hits = control_DUT_hits.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])[['DUT_name_rowcol']])
		control_DUT_hits.reset_index(['n_CAEN','CAEN_n_channel'], inplace=True, drop=True) # Not used anymore.
		control_DUT_hits['hit'] = True
		control_DUT_hits = control_DUT_hits.set_index('DUT_name_rowcol', append=True).unstack('DUT_name_rowcol', fill_value=False)
		control_DUT_hits = control_DUT_hits.droplevel(0, axis=1) # Not used anymore.
		
		control_DUT_hits = control_DUT_hits[[f'{control_DUT_name} ({_[0]},{_[1]})' for _ in control_DUT_pixels]] # Keep only the columns we need.
		control_DUT_hits['has_hit'] = control_DUT_hits.sum(axis=1)
		control_DUT_hits = control_DUT_hits.query('has_hit >= 1') # Drop all events with no hit in control DUT.
		
		# Require a coincidence between the tracks and the control DUT:
		tracks = utils.select_by_multiindex(
			df = tracks,
			idx = control_DUT_hits.index,
		)
		
		logging.info('Applying transformation to tracks to center and align DUT...')
		tracks[['Px_transformed','Py_transformed']] = translate_and_then_rotate(
			points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
			x_translation = analysis_config['x_translation'],
			y_translation = analysis_config['y_translation'],
			angle_rotation = analysis_config['rotation_around_z_deg']/180*numpy.pi,
		)
		
		# To estimate the total area, I use the tracks data before the transformation (rotation and translation) in which the total area should be a rectangle aligned with x and y, and then apply the transformation wherever needed.
		total_area_corners = pandas.DataFrame(
			{
				'x': [tracks['Px'].min(), tracks['Px'].max(), tracks['Px'].max(), tracks['Px'].min()],
				'y': [tracks['Py'].min(), tracks['Py'].min(), tracks['Py'].max(), tracks['Py'].max()],
				'which_corner': ['bottom_left','bottom_right','top_right','top_left'],
			},
		).set_index('which_corner')
		total_area = (total_area_corners.loc['bottom_right','x']-total_area_corners.loc['bottom_left','x'])*(total_area_corners.loc['top_left','y']-total_area_corners.loc['bottom_left','y'])
		
		tracks = tracks.join(pandas.DataFrame({'DUT_hit': [True for _ in range(len(DUT_hits))]}, index=DUT_hits.index))
		tracks['DUT_hit'] = tracks['DUT_hit'].fillna(False)
		
		probability_corry_fails = ufloat(0,0) # Here I force it to 0, so I assume 100 % of all tracks are good ones. Basically I am disabling my background correction mechanism.
		number_of_noHitTrack_that_are_fake_per_unit_area = (len(tracks.query('DUT_hit == False'))*probability_corry_fails/total_area)
		
		################################################################
		# Some hardcoded stuff #########################################
		################################################################
		PIXEL_SIZE = 250e-6
		ROI_DISTANCE_OFFSET = 88e-6
		ROI_WIDTH = PIXEL_SIZE/3
		PIXEL_DEPENDENT_SETTINGS = {
			'top_row': dict(
				project_on = 'x',
				ROI = dict(
					x_min = -PIXEL_SIZE-ROI_DISTANCE_OFFSET,
					x_max = PIXEL_SIZE+ROI_DISTANCE_OFFSET,
					y_min = PIXEL_SIZE/2-ROI_WIDTH/2,
					y_max = PIXEL_SIZE/2+ROI_WIDTH/2,
				),
				left_pixel_rowcol = [0,0],
				right_pixel_rowcol = [0,1],
			),
			'bottom_row': dict(
				project_on = 'x',
				ROI = dict(
					x_min = -PIXEL_SIZE-ROI_DISTANCE_OFFSET,
					x_max = PIXEL_SIZE+ROI_DISTANCE_OFFSET,
					y_min = -PIXEL_SIZE/2-ROI_WIDTH/2,
					y_max = -PIXEL_SIZE/2+ROI_WIDTH/2,
					
				),
				left_pixel_rowcol = [1,0],
				right_pixel_rowcol = [1,1],
			),
			'left_column': dict(
				project_on = 'y',
				ROI = dict(
					y_min = -PIXEL_SIZE-ROI_DISTANCE_OFFSET,
					y_max = PIXEL_SIZE+ROI_DISTANCE_OFFSET,
					x_min = -PIXEL_SIZE/2-ROI_WIDTH/2,
					x_max = -PIXEL_SIZE/2+ROI_WIDTH/2,
				),
				left_pixel_rowcol = [1,0],
				right_pixel_rowcol = [0,0],
			),
			'right_column': dict(
				project_on = 'y',
				ROI = dict(
					y_min = -PIXEL_SIZE-ROI_DISTANCE_OFFSET,
					y_max = PIXEL_SIZE+ROI_DISTANCE_OFFSET,
					x_min = PIXEL_SIZE/2-ROI_WIDTH/2,
					x_max = PIXEL_SIZE/2+ROI_WIDTH/2,
				),
				left_pixel_rowcol = [1,1],
				right_pixel_rowcol = [0,1],
			),
		}
		CALCULATION_STEP = 11e-6
		ROLLING_WINDOW_SIZE = 44e-6
		################################################################
		################################################################
		################################################################
		
		for which_pixels in PIXEL_DEPENDENT_SETTINGS.keys():
			if analysis_config[which_pixels] == False:
				continue
			which_pixels_analysis = employee.create_subrun(which_pixels)
			with which_pixels_analysis.handle_task('efficiency_vs_distance') as which_pixels_employee:
				logging.info(f'Proceeding with {which_pixels_analysis.pseudopath}...')
				# Because this is a very artisanal analysis, different in every single case, I check this here:
				if TI_LGAD_analysis.run_name == 'TI116' and control_DUT_name == 'TI122' and which_pixels == 'left_column':
					xmin = PIXEL_DEPENDENT_SETTINGS[which_pixels]['ROI']['x_min']
					xmax = -50e-6
					ymin = PIXEL_DEPENDENT_SETTINGS[which_pixels]['ROI']['y_min']
					ymax = PIXEL_DEPENDENT_SETTINGS[which_pixels]['ROI']['y_max']
					project_on = PIXEL_DEPENDENT_SETTINGS[which_pixels]['project_on']
				if TI_LGAD_analysis.run_name == 'TI122' and control_DUT_name == 'TI116' and which_pixels == 'left_column':
					# ~ xmin = PIXEL_DEPENDENT_SETTINGS[which_pixels]['ROI']['x_min']
					# ~ xmax = PIXEL_DEPENDENT_SETTINGS[which_pixels]['ROI']['x_max']
					xmin = -200e-6
					xmax = 0
					ymin = PIXEL_DEPENDENT_SETTINGS[which_pixels]['ROI']['y_min']
					ymax = PIXEL_DEPENDENT_SETTINGS[which_pixels]['ROI']['y_max']
					project_on = PIXEL_DEPENDENT_SETTINGS[which_pixels]['project_on']
				else:
					raise RuntimeError(f'This analysis not implemented for the devices you want to analyze.')
				
				tracks_for_efficiency_calculation = tracks.query(f'{xmin}<Px_transformed and Px_transformed<{xmax} and {ymin}<Py_transformed and Py_transformed<{ymax}')
				
				tracks_for_efficiency_calculation = tracks_for_efficiency_calculation.join(DUT_hits).fillna(False)
				tracks_for_efficiency_calculation['have_hit'] = tracks_for_efficiency_calculation[DUT_hits.columns].apply(lambda row: ' '.join(row[row==True].keys()), axis=1)
				tracks_for_efficiency_calculation.loc[tracks_for_efficiency_calculation['have_hit']=='','have_hit'] = 'no hit'
				
				if True:
					logging.info('Plotting tracks used for efficiency calculation...')
					fig = px.scatter(
						tracks_for_efficiency_calculation.reset_index().sort_values('have_hit'),
						title = f'Tracks projected on the DUT after transformation<br><sup>{which_pixels_analysis.pseudopath}</sup>',
						x = 'Px_transformed',
						y = 'Py_transformed',
						color = 'have_hit',
						hover_data = ['n_run','n_event'],
						labels = {
							'Px_transformeds': 'x (m)',
							'Py_transformed': 'y (m)',
						},
					)
					fig.add_shape(
						type = "rect",
						x0 = xmin, 
						y0 = ymin, 
						x1 = xmax, 
						y1 = ymax,
						line=dict(color="black"),
					)
					fig.update_yaxes(
						scaleanchor = "x",
						scaleratio = 1,
					)
					fig.write_html(
						which_pixels_employee.path_to_directory_of_my_task/'tracks_projected_on_DUT.html',
						include_plotlyjs = 'cdn',
					)
				
				logging.info('Calculating efficiency vs distance...')
				tracks_for_efficiency_calculation.sort_values(f'P{project_on}_transformed', inplace=True)
				distance_axis = numpy.arange(
					start = tracks_for_efficiency_calculation[f'P{project_on}_transformed'].min(),
					stop = tracks_for_efficiency_calculation[f'P{project_on}_transformed'].max(),
					step = CALCULATION_STEP,
				)
				efficiency_data = []
				pixel_names = {leftright: f'{TI_LGAD_analysis.run_name} ({PIXEL_DEPENDENT_SETTINGS[which_pixels][f"{leftright}_pixel_rowcol"][0]},{PIXEL_DEPENDENT_SETTINGS[which_pixels][f"{leftright}_pixel_rowcol"][1]})' for leftright in ['left','right']}
				for leftright in ['left','right','both']:
					if leftright in {'left','right'}:
						DUT_hits_for_efficiency = DUT_hits.query(f'`{pixel_names[leftright]}` == True')
					elif leftright == 'both':
						DUT_hits_for_efficiency = DUT_hits.query(f'`{pixel_names["left"]}` == True or `{pixel_names["right"]}` == True')
					else:
						raise RuntimeError('Check this, should never happen!')
					
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						efficiency_calculation_args = dict(
							tracks = tracks_for_efficiency_calculation.rename(columns={'Px_transformed':'x', 'Py_transformed':'y'})[['x','y']],
							DUT_hits = DUT_hits_for_efficiency.index,
							project_on = project_on,
							distances = distance_axis,
							window_size = ROLLING_WINDOW_SIZE,
							number_of_noHitTrack_that_are_fake_per_unit_area = number_of_noHitTrack_that_are_fake_per_unit_area.nominal_value,
						)
						error_minus, error_plus = efficiency_vs_1D_distance_rolling_error_estimation(
							**efficiency_calculation_args,
							number_of_noHitTrack_that_are_fake_per_unit_area_uncertainty = number_of_noHitTrack_that_are_fake_per_unit_area.std_dev,
							n_bootstraps = 33,
						)
					df = pandas.DataFrame(
						{
							'Distance (m)': distance_axis,
							'Efficiency': efficiency_vs_1D_distance_rolling(**efficiency_calculation_args),
							'Efficiency error_-': error_minus,
							'Efficiency error_+': error_plus,
						}
					)
					df['Pixel'] = leftright
					efficiency_data.append(df)
				efficiency_data = pandas.concat(efficiency_data)
				efficiency_data.set_index(['Pixel','Distance (m)'], inplace=True)
				
				utils.save_dataframe(
					df = efficiency_data,
					name = 'efficiency_vs_distance',
					location = which_pixels_employee.path_to_directory_of_my_task,
				)
				
				logging.info('Plotting efficiency vs distance...')
				fig = plotly_utils.line(
					data_frame = efficiency_data.sort_index().reset_index(drop=False),
					title = f'Efficiency vs distance<br><sup>{which_pixels_analysis.pseudopath}</sup>',
					x = 'Distance (m)',
					y = 'Efficiency',
					error_y = 'Efficiency error_+',
					error_y_minus = 'Efficiency error_-',
					color = 'Pixel',
					error_y_mode = 'bands',
				)
				fig.write_html(
					which_pixels_employee.path_to_directory_of_my_task/'efficiency_vs_distance.html',
					include_plotlyjs = 'cdn',
				)

def calculate_interpixel_distance(efficiency_data:pandas.DataFrame, IPD_window:float=66e-6, measure_at_efficiency:float=.5):
	"""Calculate the inter-pixel distance given the efficiency data.
	
	Arguments
	---------
	efficiency_data: pandas.DataFrame
		A data frame of the form
		```
		                    Efficiency  Efficiency error_-  Efficiency error_+
		Pixel Distance (m)                                                    
		left  -0.000337       0.000000            0.000000            0.000000
			  -0.000326       0.000000            0.000000            0.000000
			  -0.000315       0.005809            0.005824            0.005907
			  -0.000304       0.018390            0.006069            0.010397
			  -0.000293       0.025456            0.009532            0.017805
								   ...                 ...                 ...
		both   0.000289       0.087905            0.022698            0.026788
			   0.000300       0.044990            0.017955            0.025709
			   0.000311       0.036991            0.019803            0.023058
			   0.000322       0.013673            0.012875            0.014932
			   0.000333       0.021675            0.021303            0.024336

		```
	"""
	data = efficiency_data.query(f'{-IPD_window/2} < `Distance (m)` < {IPD_window/2}')
	
	data = data.sort_index()
	interpolated = {}
	for pixel in {'left','right'}:
		interpolated[pixel] = {}
		interpolated[pixel]['val'] = interp1d(
			y = data.query(f'Pixel == "{pixel}"').index.get_level_values('Distance (m)'),
			x = data.query(f'Pixel == "{pixel}"')['Efficiency'],
		)
		for error_sign in {'+','-'}:
			interpolated[pixel][error_sign] = interp1d(
				y = data.query(f'Pixel == "{pixel}"').index.get_level_values('Distance (m)'),
				x = data.query(f'Pixel == "{pixel}"')['Efficiency'] + int(f'{error_sign}1')*data.query(f'Pixel == "{pixel}"')[f'Efficiency error_{error_sign}'],
			)
	
	IPD = interpolated['right']['val'](measure_at_efficiency) - interpolated['left']['val'](measure_at_efficiency)
	IPD_largest = interpolated['right']['-'](measure_at_efficiency) - interpolated['left']['-'](measure_at_efficiency)
	IPD_smallest = interpolated['right']['+'](measure_at_efficiency) - interpolated['left']['+'](measure_at_efficiency)
	
	IPD_error_up = IPD_largest - IPD
	IPD_error_down = IPD - IPD_smallest
	
	return IPD, IPD_error_up, IPD_error_down

def interpixel_distance(TI_LGAD_analysis:RunBureaucrat, force:bool=False):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully(['this_is_a_TI-LGAD_analysis','efficiency_vs_distance_calculation'])
	
	if force==False and TI_LGAD_analysis.was_task_run_successfully('interpixel_distance'):
		return
	
	with TI_LGAD_analysis.handle_task('interpixel_distance') as employee:
		logging.info(f'Calculating IPD of {TI_LGAD_analysis.pseudopath}...')
		efficiency_data = []
		for efficiency_analysis in TI_LGAD_analysis.list_subruns_of_task('efficiency_vs_distance_calculation'):
			efficiency_analysis.check_these_tasks_were_run_successfully('efficiency_vs_distance') # This should always be the case, but just to be sure...
			_ = pandas.read_pickle(efficiency_analysis.path_to_directory_of_task('efficiency_vs_distance')/'efficiency_vs_distance.pickle')
			_ = pandas.concat({efficiency_analysis.run_name: _}, names=['pixel_group'])
			efficiency_data.append(_)
		efficiency_data = pandas.concat(efficiency_data)
		
		IPD = []
		for pixel_group, data in efficiency_data.groupby('pixel_group'):
			_IPD, IPD_error_up, IPD_error_down = calculate_interpixel_distance(
				efficiency_data = data, 
				measure_at_efficiency = .5,
			)
			IPD.append(
				{
					'pixel_group': pixel_group,
					'IPD (m)': _IPD,
					'IPD (m) error -': IPD_error_down,
					'IPD (m) error +': IPD_error_up,
				}
			)
		IPD = pandas.DataFrame.from_records(IPD).set_index('pixel_group')
		
		IPD_final_value = numpy.mean([ufloat(row['IPD (m)'], max(row['IPD (m) error -'],row['IPD (m) error +'])) for idx,row in IPD.iterrows()])
		
		utils.save_dataframe(
			IPD,
			'IPD_values',
			employee.path_to_directory_of_my_task,
		)
		utils.save_dataframe(
			pandas.Series(
				{
					'IPD (m)': IPD_final_value.nominal_value,
					'IPD (m) error': IPD_final_value.std_dev,
				}
			),
			'IPD_final_value',
			employee.path_to_directory_of_my_task,
		)
		
		fig = px.scatter(
			title = f'Inter-pixel distance<br><sup>{employee.pseudopath}</sup>',
			data_frame = IPD.reset_index(drop=False).sort_values('pixel_group'),
			x = 'pixel_group',
			y = 'IPD (m)',
			error_y = 'IPD (m) error +',
			error_y_minus = 'IPD (m) error -',
		)
		fig.add_hline(
			y = IPD_final_value.nominal_value,
			annotation_text = f'IPD = {IPD_final_value*1e6} µm'.replace('+/-','±'),
		)
		fig.add_hrect(
			y0 = IPD_final_value.nominal_value - IPD_final_value.std_dev,
			y1 = IPD_final_value.nominal_value + IPD_final_value.std_dev,
			fillcolor = "black",
			opacity = 0.1,
			line_width = 0,
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'IPD.html',
			include_plotlyjs = 'cdn',
		)

def run_all_analyses_in_a_TILGAD(TI_LGAD_analysis:RunBureaucrat, force:bool=False):
	plot_DUT_distributions(TI_LGAD_analysis, force=force)
	plot_tracks_and_hits(TI_LGAD_analysis, do_3D_plot=False, force=force)
	transformation_for_centering_and_leveling(TI_LGAD_analysis, force=force)
	estimate_fraction_of_misreconstructed_tracks(TI_LGAD_analysis, force=force)
	efficiency_vs_distance_calculation(TI_LGAD_analysis, force=force)
	interpixel_distance(TI_LGAD_analysis, force=force)
	plot_cluster_size(TI_LGAD_analysis, force=force)

def execute_all_analyses():
	TB_bureaucrat = RunBureaucrat(Path('/media/msenger/230829_gray/AIDAinnova_test_beams/TB'))
	
	analyses_config = load_analyses_config()
	
	# First of all, create directories in which to perform the data analysis if they don't exist already...
	for (campaign_name,batch_name,DUT_name),_ in analyses_config.iterrows():
		TI_LGAD_analysis = RunBureaucrat(TB_bureaucrat.path_to_run_directory/'campaigns/subruns'/campaign_name/'batches/subruns'/batch_name/'TI-LGADs_analyses/subruns'/DUT_name) # This is ugly, but currently I have no other way of getting it...
		
		if TI_LGAD_analysis.exists() == False and any([analyses_config.loc[(campaign_name,batch_name,DUT_name),_]==True for _ in {'top_row','bottom_row','left_column','right_column'}]):
			setup_TI_LGAD_analysis_within_batch(TI_LGAD_analysis.parent, TI_LGAD_analysis.run_name)
	
	for campaign in TB_bureaucrat.list_subruns_of_task('campaigns'):
		if 'june' in  campaign.run_name.lower():
			continue
		for batch in campaign.list_subruns_of_task('batches'):
			with multiprocessing.Pool(5) as p:
				p.map(
					run_all_analyses_in_a_TILGAD, 
					[TI_LGAD_analysis for TI_LGAD_analysis in batch.list_subruns_of_task('TI-LGADs_analyses')],
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
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
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
		'--plot_DUT_distributions',
		help = 'Pass this flag to run `plot_DUT_distributions`.',
		required = False,
		dest = 'plot_DUT_distributions',
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
		help = 'Pass this flag to run `transformation_for_centering_and_leveling`.',
		required = False,
		dest = 'transformation_for_centering_and_leveling',
		action = 'store_true'
	)
	parser.add_argument(
		'--efficiency_vs_distance_calculation',
		help = 'Pass this flag to run `efficiency_vs_distance_calculation`.',
		required = False,
		dest = 'efficiency_vs_distance_calculation',
		action = 'store_true'
	)
	parser.add_argument(
		'--estimate_fraction_of_misreconstructed_tracks',
		help = 'Pass this flag to run `estimate_fraction_of_misreconstructed_tracks`.',
		required = False,
		dest = 'estimate_fraction_of_misreconstructed_tracks',
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
		'--IPD',
		help = 'Pass this flag to run `interpixel_distance` analysis.',
		required = False,
		dest = 'interpixel_distance',
		action = 'store_true'
	)
	parser.add_argument(
		'--enable_3D_tracks_plot',
		help = 'Pass this flag to enable the 3D plot with the tracks, which can take longer to execute.',
		required = False,
		dest = 'enable_3D_tracks_plot',
		action = 'store_true'
	)
	parser.add_argument(
		'--force',
		help = 'If this flag is passed, it will force whatever has to be done, meaning old data will be deleted.',
		required = False,
		dest = 'force',
		action = 'store_true'
	)
	parser.add_argument(
		'--run_all_analyses',
		help = 'If this flag is passed, it will execute `run_all_analyses_in_a_TILGAD`.',
		required = False,
		dest = 'run_all_analyses_in_a_TILGAD',
		action = 'store_true'
	)
	args = parser.parse_args()
	
	bureaucrat = RunBureaucrat(Path(args.directory))
	
	# ~ efficiency_vs_distance_calculation_usin_coincidence_with_other_sensor(
		# ~ TI_LGAD_analysis = RunBureaucrat(Path('/media/msenger/230829_gray/AIDAinnova_test_beams/TB/campaigns/subruns/230614_June/batches/subruns/batch_2_230V/TI-LGADs_analyses/subruns/TI116')),
		# ~ control_DUT_name = 'TI122',
		# ~ control_DUT_amplitude_thresohld = -7e-3, 
		# ~ control_DUT_pixels = [(0,0),(1,0)],
		# ~ force = True,
	# ~ )
	efficiency_vs_distance_calculation_usin_coincidence_with_other_sensor(
		TI_LGAD_analysis = RunBureaucrat(Path('/media/msenger/230829_gray/AIDAinnova_test_beams/TB/campaigns/subruns/230614_June/batches/subruns/batch_2_230V/TI-LGADs_analyses/subruns/TI122')),
		control_DUT_name = 'TI116',
		control_DUT_amplitude_thresohld = -7e-3, 
		control_DUT_pixels = [(0,0),(1,0)],
		force = True,
	)
	
	AAAAAAAAAAAAAAAAAAAA
	AAAAAAAAAAAAAAAAAAAA
	AAAAAAAAAAAAAAAAAAAA
	AAAAAAAAAAAAAAAAAAAA
	AAAAAAAAAAAAAAAAAAAA
	AAAAAAAAAAAAAAAAAAAA
	
	if bureaucrat.was_task_run_successfully('this_is_a_TI-LGAD_analysis'):
		if args.plot_DUT_distributions == True:
			plot_DUT_distributions(bureaucrat, force=args.force)
		if args.plot_tracks_and_hits == True:
			plot_tracks_and_hits(bureaucrat, do_3D_plot=args.enable_3D_tracks_plot, force=args.force)
		if args.transformation_for_centering_and_leveling == True:
			transformation_for_centering_and_leveling(bureaucrat, draw_square=True, force=args.force)
		if args.efficiency_vs_distance_calculation == True:
			efficiency_vs_distance_calculation(bureaucrat, force=args.force)
		if args.estimate_fraction_of_misreconstructed_tracks == True:
			estimate_fraction_of_misreconstructed_tracks(bureaucrat, force=args.force)
		if args.interpixel_distance == True:
			interpixel_distance(bureaucrat, force=args.force)
		if args.plot_cluster_size == True:
			plot_cluster_size(bureaucrat, force=args.force)
		if args.run_all_analyses_in_a_TILGAD:
			run_all_analyses_in_a_TILGAD(bureaucrat, force=args.force)
	elif args.setup_analysis_for_DUT != 'None':
		setup_TI_LGAD_analysis_within_batch(
			bureaucrat, 
			DUT_name = args.setup_analysis_for_DUT,
		)
	else:
		raise RuntimeError(f"Don't know what to do in {bureaucrat.path_to_run_directory}... Please read script help or source code.")
