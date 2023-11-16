from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import utils
import utils_batch_level
import tracks_utils
import logging
import pandas
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy
import plotly_utils
import json
import warnings
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

def draw_pixel(fig, x0, y0, x1, y1, **kwargs):
	for line_color, line_width in [('black',2.5),('white',1)]:
		fig.add_shape(
			x0 = x0,
			y0 = y0,
			x1 = x1,
			y1 = y1,
			type = "rect",
			line = dict(
				color = line_color,
				width = line_width,
			),
			**kwargs,
		)

def draw_2x2_DUT(fig, pixel_size, **kwargs):
	for i in [0,1]:
		for j in [0,1]:
			draw_pixel(
				fig = fig,
				x0 = pixel_size*j - pixel_size,
				y0 = pixel_size*i - pixel_size,
				x1 = pixel_size*j,
				y1 = pixel_size*i,
				**kwargs,
			)
	draw_pixel(
		fig = fig,
		x0 = - pixel_size,
		y0 = - pixel_size,
		x1 = pixel_size,
		y1 = pixel_size,
		**kwargs,
	)

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

def plot_DUT_distributions(TI_LGAD_analysis:RunBureaucrat, force:bool=False, maximum_number_of_events_per_plot=9999):
	"""Plot some raw distributions, useful to configure cuts and thresholds
	for further steps in the analysis."""
	TI_LGAD_analysis.check_these_tasks_were_run_successfully('this_is_a_TI-LGAD_analysis') # To be sure we are inside what is supposed to be a TI-LGAD analysis.
	
	if force==False and TI_LGAD_analysis.was_task_run_successfully('plot_distributions'):
		return
	
	with TI_LGAD_analysis.handle_task('plot_distributions') as employee:
		setup_config = utils_batch_level.load_setup_configuration_info(TI_LGAD_analysis.parent)
		
		save_distributions_plots_here = employee.path_to_directory_of_my_task/'distributions'
		save_distributions_plots_here.mkdir()
		for variable in ['Amplitude (V)','t_50 (s)','Noise (V)','Time over 50% (s)',]:
			logging.info(f'Plotting {variable} distribution...')
			data = utils_batch_level.load_parsed_from_waveforms(
				TB_batch = TI_LGAD_analysis.parent,
				load_this = {DUT_name_rowcol: None for DUT_name_rowcol in set(setup_config.query(f'DUT_name == "{TI_LGAD_analysis.run_name}"')['DUT_name_rowcol'])},
				variables = [variable],
			)
			data = data.sample(maximum_number_of_events_per_plot) if len(data) > maximum_number_of_events_per_plot else data
			fig = px.ecdf(
				data.sort_values('DUT_name_rowcol').reset_index(drop=False),
				title = f'{variable} distribution<br><sup>{TI_LGAD_analysis.pseudopath}</sup>',
				x = variable,
				marginal = 'histogram',
				color = 'DUT_name_rowcol',
				labels = utils.PLOTS_LABELS,
			)
			fig.write_html(
				save_distributions_plots_here/f'{variable}_ECDF.html',
				include_plotlyjs = 'cdn',
			)
		
		save_scatter_plots_here = employee.path_to_directory_of_my_task/'scatter_plots'
		save_scatter_plots_here.mkdir()
		for x,y in [('t_50 (s)','Amplitude (V)'), ('Time over 50% (s)','Amplitude (V)'),]:
			logging.info(f'Plotting {y} vs {x} scatter_plot...')
			data = utils_batch_level.load_parsed_from_waveforms(
				TB_batch = TI_LGAD_analysis.parent,
				load_this = {DUT_name_rowcol: None for DUT_name_rowcol in set(setup_config.query(f'DUT_name == "{TI_LGAD_analysis.run_name}"')['DUT_name_rowcol'])},
				variables = [x,y],
			)
			data = data.sample(maximum_number_of_events_per_plot) if len(data) > maximum_number_of_events_per_plot else data
			fig = px.scatter(
				data.sort_values('DUT_name_rowcol').reset_index(drop=False),
				title = f'{y} vs {x} scatter plot<br><sup>{TI_LGAD_analysis.pseudopath}</sup>',
				x = x,
				y = y,
				color = 'DUT_name_rowcol',
				hover_data = ['n_run','n_event'],
				labels = utils.PLOTS_LABELS,
			)
			fig.write_html(
				save_scatter_plots_here/f'{y}_vs_{x}_scatter.html',
				include_plotlyjs = 'cdn',
			)

def plot_tracks_and_hits(TI_LGAD_analysis:RunBureaucrat, force:bool=False):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully('this_is_a_TI-LGAD_analysis')
	
	if force==False and TI_LGAD_analysis.was_task_run_successfully('plot_tracks_and_hits'):
		return
	
	with TI_LGAD_analysis.handle_task('plot_tracks_and_hits') as employee:
		analysis_config = load_this_TILGAD_analysis_config(TI_LGAD_analysis)
		setup_config = utils_batch_level.load_setup_configuration_info(TI_LGAD_analysis.parent)
		
		tracks = utils_batch_level.load_tracks(TI_LGAD_analysis.parent, only_multiplicity_one=True)
		tracks = tracks.join(tracks_utils.project_tracks(tracks, z=analysis_config['DUT_z_position']))
		
		hit_criterion = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{analysis_config['Amplitude threshold (V)']}"
		DUT_hits = utils_batch_level.load_hits(
			TB_batch = TI_LGAD_analysis.parent,
			DUTs_and_hit_criterions = {DUT_name_rowcol:hit_criterion for DUT_name_rowcol in set(setup_config.query(f'DUT_name=="{TI_LGAD_analysis.run_name}"')['DUT_name_rowcol'])},
		)
		
		tracks = tracks_utils.tag_tracks_with_DUT_hits(tracks, DUT_hits)
		tracks = tracks.sort_values('DUT_name_rowcol', ascending=False)
		
		logging.info('Plotting tracks and hits...')
		fig = px.scatter(
			tracks.reset_index(),
			title = f'Tracks and hits projected on the DUT<br><sup>{TI_LGAD_analysis.pseudopath}, amplitude < {analysis_config["Amplitude threshold (V)"]*1e3:.1f} mV</sup>',
			x = 'Px',
			y = 'Py',
			color = 'DUT_name_rowcol',
			hover_data = ['n_run','n_event'],
			labels = utils.PLOTS_LABELS,
		)
		fig.update_yaxes(
			scaleanchor = "x",
			scaleratio = 1,
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'tracks_and_hits.html',
			include_plotlyjs = 'cdn',
		)

def plot_cluster_size(TI_LGAD_analysis:RunBureaucrat, force:bool=False):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully('this_is_a_TI-LGAD_analysis')
	TASK_NAME = 'plot_cluster_size'
	
	if force==False and TI_LGAD_analysis.was_task_run_successfully(TASK_NAME):
		return
	
	with TI_LGAD_analysis.handle_task(TASK_NAME) as employee:
		analysis_config = load_this_TILGAD_analysis_config(TI_LGAD_analysis)
		setup_config = utils_batch_level.load_setup_configuration_info(TI_LGAD_analysis.parent)
		
		tracks = utils_batch_level.load_tracks(TI_LGAD_analysis.parent, only_multiplicity_one=True)
		tracks = tracks.join(tracks_utils.project_tracks(tracks, z=analysis_config['DUT_z_position']))
		
		hit_criterion = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{analysis_config['Amplitude threshold (V)']}"
		hits = utils_batch_level.load_hits(
			TB_batch = TI_LGAD_analysis.parent,
			DUTs_and_hit_criterions = {DUT_name_rowcol:hit_criterion for DUT_name_rowcol in set(setup_config.query(f'DUT_name=="{TI_LGAD_analysis.run_name}"')['DUT_name_rowcol'])},
		)
		
		cluster_size = hits.sum(axis=1)
		cluster_size.name = 'cluster_size'
		
		tracks = tracks.join(cluster_size)
		tracks['cluster_size'].fillna(0, inplace=True)
		tracks['cluster_size'] = tracks['cluster_size'].astype(int)
		
		fig = px.scatter(
			data_frame = tracks.reset_index().sort_values('cluster_size').astype({'cluster_size':str}),
			title = f'Cluster size<br><sup>{TI_LGAD_analysis.pseudopath}, amplitude < {analysis_config["Amplitude threshold (V)"]*1e3:.1f} mV</sup>',
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
			title = f'Cluster size<br><sup>{TI_LGAD_analysis.pseudopath}, amplitude < {analysis_config["Amplitude threshold (V)"]*1e3:.1f} mV</sup>',
			text_auto = True,
		)
		fig.update_yaxes(type="log")
		fig.write_html(
			employee.path_to_directory_of_my_task/'cluster_size_histogram.html',
			include_plotlyjs = 'cdn',
		)

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
		analysis_config = load_this_TILGAD_analysis_config(TI_LGAD_analysis)
		__ = {'x_translation','y_translation','rotation_around_z_deg'}
		if any([numpy.isnan(analysis_config[_]) for _ in __]):
			raise RuntimeError(f'One (or more) of {__} is `NaN`, check the spreadsheet.')
		
		setup_config = utils_batch_level.load_setup_configuration_info(TI_LGAD_analysis.parent)
		
		tracks = utils_batch_level.load_tracks(TI_LGAD_analysis.parent, only_multiplicity_one=True)
		tracks = tracks.join(tracks_utils.project_tracks(tracks, z=analysis_config['DUT_z_position']))
		
		hit_criterion = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{analysis_config['Amplitude threshold (V)']}"
		DUT_hits = utils_batch_level.load_hits(
			TB_batch = TI_LGAD_analysis.parent,
			DUTs_and_hit_criterions = {DUT_name_rowcol:hit_criterion for DUT_name_rowcol in set(setup_config.query(f'DUT_name=="{TI_LGAD_analysis.run_name}"')['DUT_name_rowcol'])},
		)
		
		tracks = tracks_utils.tag_tracks_with_DUT_hits(tracks, DUT_hits)
		tracks = tracks.sort_values('DUT_name_rowcol', ascending=False)
		
		logging.info('Applying transformation to tracks to center and align DUT...')
		tracks[['Px','Py']] = translate_and_then_rotate(
			points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
			x_translation = analysis_config['x_translation'],
			y_translation = analysis_config['y_translation'],
			angle_rotation = analysis_config['rotation_around_z_deg']/180*numpy.pi,
		)
		
		with open(employee.path_to_directory_of_my_task/'transformation_parameters.json', 'w') as ofile:
			json.dump(
				dict(
					x_translation = analysis_config['x_translation'],
					y_translation = analysis_config['y_translation'],
					rotation_around_z_deg = analysis_config['rotation_around_z_deg'],
				),
				ofile,
				indent = '\t',
			)
		
		logging.info('Plotting tracks and hits on DUT...')
		fig = px.scatter(
			tracks.reset_index(),
			title = f'Tracks projected on the DUT after transformation<br><sup>{TI_LGAD_analysis.pseudopath}, amplitude < {analysis_config["Amplitude threshold (V)"]*1e3:.1f} mV</sup>',
			x = 'Px',
			y = 'Py',
			color = 'DUT_name_rowcol',
			hover_data = ['n_run','n_event'],
			labels = {
				'Px': 'x (m)',
				'Py': 'y (m)',
			},
		)
		# ~ for xy,method in dict(x=fig.add_vline, y=fig.add_hline).items():
			# ~ method(0)
		# ~ if draw_square:
			# ~ fig.add_shape(
				# ~ type = "rect",
				# ~ x0 = -250e-6, 
				# ~ y0 = -250e-6, 
				# ~ x1 = 250e-6, 
				# ~ y1 = 250e-6,
			# ~ )
		draw_2x2_DUT(fig, pixel_size=250e-6)
		fig.update_yaxes(
			scaleanchor = "x",
			scaleratio = 1,
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'tracks_after_applying_transformation.html',
			include_plotlyjs = 'cdn',
		)

def get_transformation_parameters(analysis:RunBureaucrat):
	analysis.check_these_tasks_were_run_successfully('transformation_for_centering_and_leveling')
	with open(analysis.path_to_directory_of_task('transformation_for_centering_and_leveling')/'transformation_parameters.json', 'r') as ifile:
		return json.load(ifile)

def estimate_fraction_of_misreconstructed_tracks(TI_LGAD_analysis:RunBureaucrat, force:bool=False):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully('this_is_a_TI-LGAD_analysis')
	
	if force==False and TI_LGAD_analysis.was_task_run_successfully('estimate_fraction_of_misreconstructed_tracks'):
		return
	
	with TI_LGAD_analysis.handle_task('estimate_fraction_of_misreconstructed_tracks') as employee:
		batch = TI_LGAD_analysis.parent
		
		analysis_config = load_this_TILGAD_analysis_config(TI_LGAD_analysis)
		setup_config = utils_batch_level.load_setup_configuration_info(batch)
		
		tracks = utils_batch_level.load_tracks(TI_LGAD_analysis.parent, only_multiplicity_one=True)
		tracks = tracks.join(tracks_utils.project_tracks(tracks, z=analysis_config['DUT_z_position']))
		
		hit_criterion = f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{analysis_config['Amplitude threshold (V)']}"
		DUT_hits = utils_batch_level.load_hits(
			TB_batch = TI_LGAD_analysis.parent,
			DUTs_and_hit_criterions = {DUT_name_rowcol:hit_criterion for DUT_name_rowcol in set(setup_config.query(f'DUT_name=="{TI_LGAD_analysis.run_name}"')['DUT_name_rowcol'])},
		)
		hit_multiplicity = DUT_hits.sum(axis=1)
		
		logging.info('Applying transformation to tracks to center and align DUT...')
		transformation_parameters = get_transformation_parameters(TI_LGAD_analysis)
		tracks[['Px_transformed','Py_transformed']] = translate_and_then_rotate(
			points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
			x_translation = transformation_parameters['x_translation'],
			y_translation = transformation_parameters['y_translation'],
			angle_rotation = transformation_parameters['rotation_around_z_deg']/180*numpy.pi,
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
			x_translation = transformation_parameters['x_translation'],
			y_translation = transformation_parameters['y_translation'],
			angle_rotation = transformation_parameters['rotation_around_z_deg']/180*numpy.pi/4,
		)
		
		logging.info(f'Calculating probability that corry fails...')
		data = []
		for DUT_ROI_size in numpy.linspace(111e-6,2222e-6,33):
			tracks_for_which_DUT_has_a_signal = utils.select_by_multiindex(tracks, hit_multiplicity[hit_multiplicity>0].index)
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
		probability_corry_fails_final_value = ufloat(
			numpy.mean([_['probability_corry_fails'] for i,_ in data.loc[dydx**2<22**2].iterrows()]),
			numpy.mean([_['probability_corry_fails error'] for i,_ in data.loc[dydx**2<22**2].iterrows()]),
		)
		
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
				'probability_corry_fails': 'Probability',
				'DUT_ROI_size (m)': 'DUT ROI size (m)',
			},
		)
		fig.add_hline(
			probability_corry_fails_final_value.nominal_value,
			annotation_text = f'Reconstruction error probability = {probability_corry_fails_final_value}',
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

def efficiency_vs_distance_left_right(DUT_analysis:RunBureaucrat, analysis_name:str, force:bool=False):
	"""Calculates the efficiency vs distance using a left and a right pixel,
	by sweeping a window along the transversal axis to the pixels. 
	Expects to find a file named `efficiency_vs_distance_left_right.config.json`
	in the run directory with a content like this:
	```
	{
		"TopRow": {
			"DUT_left_pixel": [0,0],
			"DUT_right_pixel": [0,1],
			"DUT_hit_criterion": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9",
			"use_estimation_of_misreconstructed_tracks": true,
			"pixel_size": 250e-6,
			"ROI_distance_offset_from_pixel_border": 88e-6,
			"ROI_width": 99e-6,
			"calculation_step": 11e-6,
			"bin_size": 44e-6
		}
	}
	```
	where the top level dictionary specifies different analyses with the
	key being the `analysis_name` and the items the analysis config.
	"""
	THIS_FUNCTION_PLOTS_LABELS = {
		'pixel_hit': 'Pixel hit',
	}
	_colors = px.colors.qualitative.Plotly
	THIS_FUNCTION_COLOR_DISCRETE_MAP_FOR_PIXEL_HIT = {
		'none': _colors[0],
		'left': _colors[1],
		'right': _colors[2],
		'both': _colors[3],
	}
	TASK_NAME = 'efficiency_vs_distance_left_right'
	
	# The checking mechanism becomes a bit tricky because this function actually operates on a subrun...
	_this_analysis_bureaucrat = [_ for _ in DUT_analysis.list_subruns_of_task(TASK_NAME) if _.run_name==analysis_name]
	if len(_this_analysis_bureaucrat) > 1: # If it is 0, it was not run yet, if it is 1 it was.
		raise RuntimeError(f'This should have never happen, check!')
	if force == False and len(_this_analysis_bureaucrat)==1 and _this_analysis_bureaucrat[0].was_task_run_successfully(TASK_NAME):
		return
	
	# Read analysis config:
	path_to_analysis_config_file = DUT_analysis.path_to_run_directory/'efficiency_vs_distance_left_right.config.json'
	if not path_to_analysis_config_file.is_file():
		raise RuntimeError(f'Cannot find analysis config file for {DUT_analysis.pseudopath}. You have to create a json file named {path_to_analysis_config_file.name} in the run directory of {DUT_analysis.pseudopath}. ')
	with open(path_to_analysis_config_file, 'r') as ifile:
		analysis_config = json.load(ifile)
		_analyses_names_in_config_file = analysis_config.keys()
		analysis_config = analysis_config.get(analysis_name)
		if analysis_config is None:
			raise RuntimeError(f'No analysis named "{analysis_name}" found in analysis config file for {DUT_analysis.pseudopath}. Analyses names found {sorted(_analyses_names_in_config_file)}')
	
	with DUT_analysis.handle_task(TASK_NAME, drop_old_data=False) as _:
		with _.create_subrun(analysis_name, if_exists='skip').handle_task(TASK_NAME) as employee:
			with open(employee.path_to_run_directory/path_to_analysis_config_file.name, 'w') as ofile:
				json.dump(
					analysis_config,
					ofile,
					indent = '\t',
				)
			setup_config = utils_batch_level.load_setup_configuration_info(DUT_analysis.parent)
			
			tracks = utils_batch_level.load_tracks(
				DUT_analysis.parent, 
				only_multiplicity_one = True,
				trigger_on_DUTs = analysis_config.get('trigger_on_DUTs'),
			)
			DUT_z_position = list(set(setup_config.query(f'DUT_name == "{DUT_analysis.run_name}"')['z (m)']))
			if len(DUT_z_position) != 1: # This should never happen, but just in case it is better to be prepared.
				raise RuntimeError(f'Cannot determine DUT z position for {DUT_analysis.pseudopath}. ')
			else:
				DUT_z_position = DUT_z_position[0]
			tracks = tracks.join(tracks_utils.project_tracks(tracks, z=DUT_z_position))
			
			DUT_name_rowcol_left_and_right_pixels = {
				'left': f'{DUT_analysis.run_name} ({analysis_config["DUT_left_pixel"][0]},{analysis_config["DUT_left_pixel"][1]})',
				'right': f'{DUT_analysis.run_name} ({analysis_config["DUT_right_pixel"][0]},{analysis_config["DUT_right_pixel"][1]})',
			}
			DUT_hits = utils_batch_level.load_hits(
				TB_batch = DUT_analysis.parent,
				DUTs_and_hit_criterions = {DUT_name_rowcol_left_and_right_pixels[leftright]:analysis_config['DUT_hit_criterion'] for leftright in {'left','right'}},
			)
			
			transformation_parameters = get_transformation_parameters(DUT_analysis)
			
			if analysis_config.get('use_estimation_of_misreconstructed_tracks') == True:
				DUT_analysis.check_these_tasks_were_run_successfully('estimate_fraction_of_misreconstructed_tracks')
				misreconstruction_probability = pandas.read_pickle(DUT_analysis.path_to_directory_of_task('estimate_fraction_of_misreconstructed_tracks')/'probability_that_corry_fails.pickle')
				misreconstruction_probability = ufloat(misreconstruction_probability['probability_corry_fails'],misreconstruction_probability['probability_corry_fails_error'])
			
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
					x_translation = transformation_parameters['x_translation'],
					y_translation = transformation_parameters['y_translation'],
					angle_rotation = transformation_parameters['rotation_around_z_deg']/180*numpy.pi,
				)
				total_area = (total_area_corners.loc['bottom_right','x']-total_area_corners.loc['bottom_left','x'])*(total_area_corners.loc['top_left','y']-total_area_corners.loc['bottom_left','y'])
				
				tracks_with_no_hit_in_the_DUT = tracks_utils.tag_tracks_with_DUT_hits(tracks, DUT_hits).query('DUT_name_rowcol == "no hit"')
				number_of_tracks_with_no_hit_in_the_DUT = len(tracks_with_no_hit_in_the_DUT.index.drop_duplicates()) # The "drop duplicates" thing is just in case to avoid double counting, anyhow it is not expected.
				number_of_noHitTrack_that_are_fake_per_unit_area = number_of_tracks_with_no_hit_in_the_DUT*misreconstruction_probability/total_area
			else:
				number_of_noHitTrack_that_are_fake_per_unit_area = ufloat(0,0)
			
			logging.info('Applying transformation to tracks to center and align DUT...')
			tracks[['Px','Py']] = translate_and_then_rotate(
				points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
				x_translation = transformation_parameters['x_translation'],
				y_translation = transformation_parameters['y_translation'],
				angle_rotation = transformation_parameters['rotation_around_z_deg']/180*numpy.pi,
			)
			
			# Keep only necessary stuff:
			tracks = tracks[['Px','Py']] # Keep only stuff that will be used.
			tracks.reset_index('n_track', inplace=True, drop=True) # Not used anymore.
			
			# Guess which is the axis parallel to the pixels and which is the one perpendicular, and check that the pixels are consecutive:
			delta_rowcol_vector = tuple(analysis_config['DUT_right_pixel'][_] - analysis_config['DUT_left_pixel'][_] for _ in [0,1])
			match delta_rowcol_vector:
				case (-1,0):
					parallel_axis = 'y'
					perpendicular_axis = 'x'
				case (0,1):
					parallel_axis = 'x'
					perpendicular_axis = 'y'
				case _:
					raise RuntimeError(f'Cannot detemrine which of the xy axes is parallel and which is perpendicular to the left and right pixels for {employee.pseudopath}, one possible reason is that in the config file the "DUT_left_pixel" and "DUT_right_pixel" are wrong.')
			
			# Define ROI for the tracks:
			pixel_size = analysis_config['pixel_size']
			ROI_distance_offset_from_pixel_border = analysis_config['ROI_distance_offset_from_pixel_border']
			ROI_width = analysis_config['ROI_width']
			match parallel_axis:
				case 'x':
					xmin = -pixel_size - ROI_distance_offset_from_pixel_border
					xmax =  pixel_size + ROI_distance_offset_from_pixel_border
					ymin = -ROI_width/2 + pixel_size/2 - pixel_size*analysis_config['DUT_left_pixel'][0]
					ymax =  ROI_width/2 + pixel_size/2 - pixel_size*analysis_config['DUT_left_pixel'][0]
				case 'y':
					ymin = -pixel_size - ROI_distance_offset_from_pixel_border
					ymax =  pixel_size + ROI_distance_offset_from_pixel_border
					xmin = -ROI_width/2 - pixel_size/2 + pixel_size*analysis_config['DUT_left_pixel'][1]
					xmax =  ROI_width/2 - pixel_size/2 + pixel_size*analysis_config['DUT_left_pixel'][1]
				case _:
					raise RuntimeError('This should have never happen!')
			
			tracks_for_efficiency_calculation = tracks.query(f'{xmin}<Px and Px<{xmax} and {ymin}<Py and Py<{ymax}') # Select by physical ROI.
			
			with warnings.catch_warnings():
				warnings.simplefilter("ignore") # This is to hide the "A value is trying to be set on a copy of a slice from a DataFrame." warning from Pandas.
				tracks_for_efficiency_calculation['pixel_hit'] = 'none'
				for leftright in ['left','right']:
					this_case_hits = DUT_hits[DUT_name_rowcol_left_and_right_pixels[leftright]]
					tracks_for_efficiency_calculation = tracks_for_efficiency_calculation.join(this_case_hits)
					tracks_for_efficiency_calculation = tracks_for_efficiency_calculation.rename(columns={DUT_name_rowcol_left_and_right_pixels[leftright]: leftright})
					tracks_for_efficiency_calculation[leftright] = tracks_for_efficiency_calculation[leftright].fillna(False)
				
					tracks_for_efficiency_calculation.loc[tracks_for_efficiency_calculation[leftright]==True,'pixel_hit'] = leftright
				tracks_for_efficiency_calculation.loc[(tracks_for_efficiency_calculation['left']==True) & (tracks_for_efficiency_calculation['right']==True),'pixel_hit'] = 'both'
			
			if True:
				logging.info('Plotting tracks used for efficiency calculation...')
				fig = px.scatter(
					tracks_for_efficiency_calculation.reset_index().sort_values('pixel_hit'),
					title = f'Tracks used in efficiency calculation<br><sup>{employee.pseudopath}</sup>',
					x = 'Px',
					y = 'Py',
					color = 'pixel_hit',
					hover_data = ['n_run','n_event'],
					labels = utils.PLOTS_LABELS | THIS_FUNCTION_PLOTS_LABELS,
					color_discrete_map = THIS_FUNCTION_COLOR_DISCRETE_MAP_FOR_PIXEL_HIT,
				)
				fig.add_shape(
					type = "rect",
					x0 = xmin, 
					y0 = ymin, 
					x1 = xmax, 
					y1 = ymax,
					name = 'ROI',
					showlegend = True,
				)
				fig.add_shape(
					type = "rect",
					x0 = -250e-6, 
					y0 = -250e-6, 
					x1 = 250e-6, 
					y1 = 250e-6,
					name = 'DUT',
					showlegend = True,
				)
				fig.update_yaxes(
					scaleanchor = "x",
					scaleratio = 1,
				)
				fig.write_html(
					employee.path_to_directory_of_my_task/'tracks_used_in_efficiency_calculation_scatter.html',
					include_plotlyjs = 'cdn',
				)
				
				for xy in {'x','y'}:
					fig = px.ecdf(
						tracks_for_efficiency_calculation,
						title = f'Tracks used in efficiency calculation projected on {xy}<br><sup>{employee.pseudopath}</sup>',
						x = f'P{xy}',
						color = 'pixel_hit',
						labels = utils.PLOTS_LABELS | THIS_FUNCTION_PLOTS_LABELS,
						marginal = 'histogram',
						color_discrete_map = THIS_FUNCTION_COLOR_DISCRETE_MAP_FOR_PIXEL_HIT,
					)
					fig.write_html(
						employee.path_to_directory_of_my_task/f'tracks_used_in_efficiency_calculation_projected_on_{xy}.html',
						include_plotlyjs = 'cdn',
					)
			
			tracks_for_efficiency_calculation.sort_values(f'P{parallel_axis}', inplace=True)
			distance_axis = numpy.arange(
				start = tracks_for_efficiency_calculation[f'P{parallel_axis}'].min(),
				stop = tracks_for_efficiency_calculation[f'P{parallel_axis}'].max(),
				step = analysis_config['calculation_step'],
			)
			
			logging.info(f'Calculating efficiency vs distance left right for {employee.pseudopath}...')
			efficiency_data = []
			for leftright in ['left','right','both']:
				if leftright in {'left','right'}:
					DUT_hits_for_efficiency = DUT_hits.query(f'`{DUT_name_rowcol_left_and_right_pixels[leftright]}` == True')
				elif leftright == 'both':
					DUT_hits_for_efficiency = DUT_hits.query(f'`{DUT_name_rowcol_left_and_right_pixels["left"]}` == True or `{DUT_name_rowcol_left_and_right_pixels["right"]}` == True')
				else:
					raise RuntimeError('Check this, should never happen!')
				
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					efficiency_calculation_args = dict(
						tracks = tracks_for_efficiency_calculation.rename(columns={'Px':'x', 'Py':'y'})[['x','y']],
						DUT_hits = DUT_hits_for_efficiency.index,
						project_on = parallel_axis,
						distances = distance_axis,
						window_size = analysis_config['bin_size'],
						number_of_noHitTrack_that_are_fake_per_unit_area = number_of_noHitTrack_that_are_fake_per_unit_area.nominal_value,
					)
					error_minus, error_plus = efficiency_vs_1D_distance_rolling_error_estimation(
						**efficiency_calculation_args,
						number_of_noHitTrack_that_are_fake_per_unit_area_uncertainty = number_of_noHitTrack_that_are_fake_per_unit_area.std_dev,
						n_bootstraps = 33,
						confidence_level = .99,
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
				location = employee.path_to_directory_of_my_task,
			)
			
			logging.info('Plotting efficiency vs distance...')
			fig = plotly_utils.line(
				data_frame = efficiency_data.sort_index().reset_index(drop=False),
				title = f'Efficiency vs distance<br><sup>{employee.pseudopath}</sup>',
				x = 'Distance (m)',
				y = 'Efficiency',
				error_y = 'Efficiency error_+',
				error_y_minus = 'Efficiency error_-',
				color = 'Pixel',
				error_y_mode = 'bands',
				color_discrete_map = THIS_FUNCTION_COLOR_DISCRETE_MAP_FOR_PIXEL_HIT,
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/'efficiency_vs_distance.html',
				include_plotlyjs = 'cdn',
			)

def run_all_efficiency_vs_distance_left_right(DUT_analysis:RunBureaucrat, force:bool=False):
	"""Runs all `efficiency_vs_distance_left_right` analyses defined in
	the file `efficiency_vs_distance_left_right.config.json` within a
	DUT analysis directory."""
	
	# Read analysis config:
	path_to_analysis_config_file = DUT_analysis.path_to_run_directory/'efficiency_vs_distance_left_right.config.json'
	if not path_to_analysis_config_file.is_file():
		raise RuntimeError(f'Cannot find analysis config file for {DUT_analysis.pseudopath}. You have to create a json file named {path_to_analysis_config_file.name} in the run directory of {DUT_analysis.pseudopath}. ')
	with open(path_to_analysis_config_file, 'r') as ifile:
		analysis_config = json.load(ifile)
	
	for analysis_name in analysis_config.keys():
		logging.info(f'Launching `efficiency_vs_distance_left_right` for {DUT_analysis.pseudopath}/{analysis_name} (force={force})')
		efficiency_vs_distance_left_right(
			DUT_analysis = DUT_analysis,
			analysis_name = analysis_name,
			force = force,
		)

def efficiency_vs_distance_left_right_folding(DUT_analysis:RunBureaucrat, analysis_name:str, force:bool=False):
	"""Calculates the efficiency vs distance using a left and a right pixel,
	by sweeping a window along the transversal axis to the pixels, and 
	folding the pixels so there is more statistics.
	Expects to find a file named `efficiency_vs_distance_left_right_folding.config.json`
	in the run directory with a content like this:
	```
	{
		"Deleteme": {
			"trigger_on_DUTs": {
				"TI122 (0,0)": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9",
				"TI122 (0,1)": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9",
				"TI122 (1,0)": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9",
				"TI122 (1,1)": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9"
			},
			"DUT_hit_criterion": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9",
			"use_estimation_of_misreconstructed_tracks": false,
			"pixel_size": 250e-6,
			"ROI_distance_offset_from_pixel_border": 111e-6,
			"ROI_width": 133e-6,
			"calculation_step": 11e-6,
			"bin_size": 44e-6,
			"pairs_of_pixels_to_use": ["top_row","bottom_row","left_col","right_col"]
		}
	}
	```
	where the top level dictionary specifies different analyses with the
	key being the `analysis_name` and the items the analysis config.
	"""
	THIS_FUNCTION_PLOTS_LABELS = {
		'pixel_hit': 'Pixel hit',
	}
	_colors = px.colors.qualitative.Plotly
	THIS_FUNCTION_COLOR_DISCRETE_MAP_FOR_PIXEL_HIT = {
		'none': _colors[0],
		'left': _colors[1],
		'right': _colors[2],
		'both': _colors[3],
	}
	TASK_NAME = 'efficiency_vs_distance_left_right_folding'
	
	# The checking mechanism becomes a bit tricky because this function actually operates on a subrun...
	_this_analysis_bureaucrat = [_ for _ in DUT_analysis.list_subruns_of_task(TASK_NAME) if _.run_name==analysis_name]
	if len(_this_analysis_bureaucrat) > 1: # If it is 0, it was not run yet, if it is 1 it was.
		raise RuntimeError(f'This should have never happen, check!')
	if force == False and len(_this_analysis_bureaucrat)==1 and _this_analysis_bureaucrat[0].was_task_run_successfully(TASK_NAME):
		return
	
	# Read analysis config:
	path_to_analysis_config_file = DUT_analysis.path_to_run_directory/f'{TASK_NAME}.config.json'
	if not path_to_analysis_config_file.is_file():
		raise RuntimeError(f'Cannot find analysis config file for {DUT_analysis.pseudopath}. You have to create a json file named {path_to_analysis_config_file.name} in the run directory of {DUT_analysis.pseudopath}. ')
	with open(path_to_analysis_config_file, 'r') as ifile:
		analysis_config = json.load(ifile)
		_analyses_names_in_config_file = analysis_config.keys()
		analysis_config = analysis_config.get(analysis_name)
		if analysis_config is None:
			raise RuntimeError(f'No analysis named "{analysis_name}" found in analysis config file for {DUT_analysis.pseudopath}. Analyses names found {sorted(_analyses_names_in_config_file)}')
	
	with DUT_analysis.handle_task(TASK_NAME, drop_old_data=False) as _:
		with _.create_subrun(analysis_name, if_exists='skip').handle_task(TASK_NAME) as employee:
			with open(employee.path_to_run_directory/path_to_analysis_config_file.name, 'w') as ofile:
				json.dump(
					analysis_config,
					ofile,
					indent = '\t',
				)
			setup_config = utils_batch_level.load_setup_configuration_info(DUT_analysis.parent)
			
			tracks = utils_batch_level.load_tracks(
				DUT_analysis.parent, 
				only_multiplicity_one = True,
				trigger_on_DUTs = analysis_config.get('trigger_on_DUTs'),
			)
			DUT_z_position = list(set(setup_config.query(f'DUT_name == "{DUT_analysis.run_name}"')['z (m)']))
			if len(DUT_z_position) != 1: # This should never happen, but just in case it is better to be prepared.
				raise RuntimeError(f'Cannot determine DUT z position for {DUT_analysis.pseudopath}. ')
			else:
				DUT_z_position = DUT_z_position[0]
			tracks = tracks.join(tracks_utils.project_tracks(tracks, z=DUT_z_position))
			
			DUT_hits = utils_batch_level.load_hits(
				TB_batch = DUT_analysis.parent,
				DUTs_and_hit_criterions = {f'{DUT_analysis.run_name} ({pix[0]},{pix[1]})':analysis_config['DUT_hit_criterion'] for pix in [(0,0),(0,1),(1,0),(1,1)]},
			)
			
			transformation_parameters = get_transformation_parameters(DUT_analysis)
			
			if analysis_config.get('use_estimation_of_misreconstructed_tracks') == True:
				DUT_analysis.check_these_tasks_were_run_successfully('estimate_fraction_of_misreconstructed_tracks')
				misreconstruction_probability = pandas.read_pickle(DUT_analysis.path_to_directory_of_task('estimate_fraction_of_misreconstructed_tracks')/'probability_that_corry_fails.pickle')
				misreconstruction_probability = ufloat(misreconstruction_probability['probability_corry_fails'],misreconstruction_probability['probability_corry_fails_error'])
			
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
					x_translation = transformation_parameters['x_translation'],
					y_translation = transformation_parameters['y_translation'],
					angle_rotation = transformation_parameters['rotation_around_z_deg']/180*numpy.pi,
				)
				total_area = (total_area_corners.loc['bottom_right','x']-total_area_corners.loc['bottom_left','x'])*(total_area_corners.loc['top_left','y']-total_area_corners.loc['bottom_left','y'])
				
				tracks_with_no_hit_in_the_DUT = tracks_utils.tag_tracks_with_DUT_hits(tracks, DUT_hits).query('DUT_name_rowcol == "no hit"')
				number_of_tracks_with_no_hit_in_the_DUT = len(tracks_with_no_hit_in_the_DUT.index.drop_duplicates()) # The "drop duplicates" thing is just in case to avoid double counting, anyhow it is not expected.
				number_of_noHitTrack_that_are_fake_per_unit_area = number_of_tracks_with_no_hit_in_the_DUT*misreconstruction_probability/total_area
			else:
				number_of_noHitTrack_that_are_fake_per_unit_area = ufloat(0,0)
			
			logging.info('Applying transformation to tracks to center and align DUT...')
			tracks[['Px','Py']] = translate_and_then_rotate(
				points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
				x_translation = transformation_parameters['x_translation'],
				y_translation = transformation_parameters['y_translation'],
				angle_rotation = transformation_parameters['rotation_around_z_deg']/180*numpy.pi,
			)
			
			# Keep only necessary stuff:
			tracks = tracks[['Px','Py']] # Keep only stuff that will be used.
			tracks.reset_index('n_track', inplace=True, drop=True) # Not used anymore.
			
			tracks_for_efficiency_calculation = []
			for pairs_of_pixels_to_use in analysis_config['pairs_of_pixels_to_use']:
				if pairs_of_pixels_to_use not in ["top_row","bottom_row","left_col","right_col"]:
					raise ValueError(f'`pairs_of_pixels_to_use` has an invalid value, check the {DUT_analysis.path_to_run_directory}/{TASK_NAME}.config.json file.')
				
				# Define an ROI, and some other settings, according to each case:
				match pairs_of_pixels_to_use:
					case 'top_row':
						ROI_xmin = -analysis_config['pixel_size'] - analysis_config['ROI_distance_offset_from_pixel_border']
						ROI_xmax = analysis_config['pixel_size'] + analysis_config['ROI_distance_offset_from_pixel_border']
						ROI_ymin = analysis_config['pixel_size']/2 - analysis_config['ROI_width']/2
						ROI_ymax = analysis_config['pixel_size']/2 + analysis_config['ROI_width']/2
						x_translation = 0
						y_translation = -analysis_config['pixel_size']/2
						angle_rotation = 0
						left_pixel_name_rowcol = f'{DUT_analysis.run_name} (0,0)'
						right_pixel_name_rowcol = f'{DUT_analysis.run_name} (0,1)'
					case 'bottom_row':
						ROI_xmin = -analysis_config['pixel_size'] - analysis_config['ROI_distance_offset_from_pixel_border']
						ROI_xmax = analysis_config['pixel_size'] + analysis_config['ROI_distance_offset_from_pixel_border']
						ROI_ymin = -analysis_config['pixel_size']/2 - analysis_config['ROI_width']/2
						ROI_ymax = -analysis_config['pixel_size']/2 + analysis_config['ROI_width']/2
						x_translation = 0
						y_translation = analysis_config['pixel_size']/2
						angle_rotation = 0
						left_pixel_name_rowcol = f'{DUT_analysis.run_name} (1,0)'
						right_pixel_name_rowcol = f'{DUT_analysis.run_name} (1,1)'
					case 'left_col':
						ROI_ymin = -analysis_config['pixel_size'] - analysis_config['ROI_distance_offset_from_pixel_border']
						ROI_ymax = analysis_config['pixel_size'] + analysis_config['ROI_distance_offset_from_pixel_border']
						ROI_xmin = -analysis_config['pixel_size']/2 - analysis_config['ROI_width']/2
						ROI_xmax = -analysis_config['pixel_size']/2 + analysis_config['ROI_width']/2
						x_translation = analysis_config['pixel_size']/2
						y_translation = 0
						angle_rotation = -90
						left_pixel_name_rowcol = f'{DUT_analysis.run_name} (1,0)'
						right_pixel_name_rowcol = f'{DUT_analysis.run_name} (0,0)'
					case 'right_col':
						ROI_ymin = -analysis_config['pixel_size'] - analysis_config['ROI_distance_offset_from_pixel_border']
						ROI_ymax = analysis_config['pixel_size'] + analysis_config['ROI_distance_offset_from_pixel_border']
						ROI_xmin = analysis_config['pixel_size']/2 - analysis_config['ROI_width']/2
						ROI_xmax = analysis_config['pixel_size']/2 + analysis_config['ROI_width']/2
						x_translation = -analysis_config['pixel_size']/2
						y_translation = 0
						angle_rotation = -90
						left_pixel_name_rowcol = f'{DUT_analysis.run_name} (1,1)'
						right_pixel_name_rowcol = f'{DUT_analysis.run_name} (0,1)'
				
				this_case_tracks = tracks.query(f'Px>{ROI_xmin} and Px<{ROI_xmax} and Py>{ROI_ymin} and Py<{ROI_ymax}')
				this_case_tracks = this_case_tracks.copy() # To avoid a warning later on.
				
				this_case_tracks['pixels_group'] = pairs_of_pixels_to_use
				
				this_case_tracks[['Px_folded','Py_folded']] = translate_and_then_rotate(
					points = this_case_tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
					x_translation = x_translation,
					y_translation = y_translation,
					angle_rotation = angle_rotation/180*numpy.pi,
				)
				
				this_case_tracks['pixel_hit'] = 'none' # Initialize.
				for leftright, pixel_name_rowcol in {'left': left_pixel_name_rowcol, 'right': right_pixel_name_rowcol}.items():
					this_case_hits = DUT_hits[pixel_name_rowcol]
					this_case_tracks = this_case_tracks.join(this_case_hits)
					this_case_tracks = this_case_tracks.rename(columns={pixel_name_rowcol: leftright})
					this_case_tracks[leftright] = this_case_tracks[leftright].fillna(False)
					this_case_tracks.loc[this_case_tracks[leftright]==True,'pixel_hit'] = leftright
				this_case_tracks.loc[(this_case_tracks['left']==True) & (this_case_tracks['right']==True),'pixel_hit'] = 'both'
				
				tracks_for_efficiency_calculation.append(this_case_tracks)
			tracks_for_efficiency_calculation = pandas.concat(tracks_for_efficiency_calculation)
			tracks_for_efficiency_calculation = tracks_for_efficiency_calculation[~tracks_for_efficiency_calculation.index.duplicated(keep='first')]
			
			# Do some plots with the tracks used in the efficiency:
			if True:
				fig = px.scatter(
					data_frame = tracks_for_efficiency_calculation.reset_index(drop=False),
					title = f'Tracks used in efficiency calculation<br><sup>{employee.pseudopath}</sup>',
					x = 'Px_folded',
					y = 'Py_folded',
					color = 'pixel_hit',
					facet_row = 'pixels_group',
					hover_data = ['n_run','n_event'],
					labels = utils.PLOTS_LABELS | THIS_FUNCTION_PLOTS_LABELS,
					color_discrete_map = THIS_FUNCTION_COLOR_DISCRETE_MAP_FOR_PIXEL_HIT,
				)
				fig.update_yaxes(
					scaleanchor = "x",
					scaleratio = 1,
				)
				fig.write_html(
					employee.path_to_directory_of_my_task/'tracks_used_in_efficiency_calculation_after_folding.html',
					include_plotlyjs = 'cdn',
				)
				
				fig = px.scatter(
					data_frame = tracks_for_efficiency_calculation.reset_index(drop=False),
					title = f'Tracks used in efficiency calculation<br><sup>{employee.pseudopath}</sup>',
					x = 'Px',
					y = 'Py',
					color = 'pixel_hit',
					symbol = 'pixels_group',
					hover_data = ['n_run','n_event'],
					labels = utils.PLOTS_LABELS | THIS_FUNCTION_PLOTS_LABELS,
					color_discrete_map = THIS_FUNCTION_COLOR_DISCRETE_MAP_FOR_PIXEL_HIT,
				)
				draw_2x2_DUT(
					fig,
					pixel_size = analysis_config['pixel_size'],
				)
				fig.update_yaxes(
					scaleanchor = "x",
					scaleratio = 1,
				)
				fig.write_html(
					employee.path_to_directory_of_my_task/'tracks_used_in_efficiency_calculation_without_folding.html',
					include_plotlyjs = 'cdn',
				)
			
			# Now calculate the efficiency:
			distance_axis = numpy.arange(
				start = -analysis_config['pixel_size'] - analysis_config['ROI_distance_offset_from_pixel_border'],
				stop = analysis_config['pixel_size'] + analysis_config['ROI_distance_offset_from_pixel_border'],
				step = analysis_config['calculation_step'],
			)
			
			logging.info(f'Calculating efficiency vs distance left right for {employee.pseudopath}...')
			efficiency_data = []
			for leftright in ['left','right','both']:
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					if leftright in {'left','right'}:
						DUT_hits = tracks_for_efficiency_calculation.query(f'{leftright} == True').index
					elif leftright == 'both':
						DUT_hits = tracks_for_efficiency_calculation.query(f'left == True or right == True').index
					efficiency_calculation_args = dict(
						tracks = tracks_for_efficiency_calculation.rename(columns={'Px_folded':'x', 'Py_folded':'y'})[['x','y']],
						DUT_hits = DUT_hits,
						project_on = 'x',
						distances = distance_axis,
						window_size = analysis_config['bin_size'],
						number_of_noHitTrack_that_are_fake_per_unit_area = number_of_noHitTrack_that_are_fake_per_unit_area.nominal_value,
					)
					error_minus, error_plus = efficiency_vs_1D_distance_rolling_error_estimation(
						**efficiency_calculation_args,
						number_of_noHitTrack_that_are_fake_per_unit_area_uncertainty = number_of_noHitTrack_that_are_fake_per_unit_area.std_dev,
						n_bootstraps = 33,
						confidence_level = .99,
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
				location = employee.path_to_directory_of_my_task,
			)
			
			logging.info('Plotting efficiency vs distance...')
			fig = plotly_utils.line(
				data_frame = efficiency_data.sort_index().reset_index(drop=False),
				title = f'Efficiency vs distance<br><sup>{employee.pseudopath}</sup>',
				x = 'Distance (m)',
				y = 'Efficiency',
				error_y = 'Efficiency error_+',
				error_y_minus = 'Efficiency error_-',
				color = 'Pixel',
				error_y_mode = 'bands',
				color_discrete_map = THIS_FUNCTION_COLOR_DISCRETE_MAP_FOR_PIXEL_HIT,
			)
			SAFETY_ROI_MARGIN = 33e-6
			for x in [-analysis_config['pixel_size'] + analysis_config['bin_size'] + SAFETY_ROI_MARGIN, analysis_config['pixel_size'] - analysis_config['bin_size'] - SAFETY_ROI_MARGIN]:
				fig.add_vline(
					x = x,
					opacity = 0.5,
					line_width = 1,
					line_dash = 'dash',
				)
			fig.write_html(
				employee.path_to_directory_of_my_task/'efficiency_vs_distance.html',
				include_plotlyjs = 'cdn',
			)

def run_all_efficiency_vs_distance_left_right_folding(DUT_analysis:RunBureaucrat, force:bool=False):
	"""Runs all `efficiency_vs_distance_left_right_folding` analyses defined in
	the file `efficiency_vs_distance_left_right_folding.config.json` within a
	DUT analysis directory."""
	
	# Read analysis config:
	path_to_analysis_config_file = DUT_analysis.path_to_run_directory/'efficiency_vs_distance_left_right_folding.config.json'
	if not path_to_analysis_config_file.is_file():
		raise RuntimeError(f'Cannot find analysis config file for {DUT_analysis.pseudopath}. You have to create a json file named {path_to_analysis_config_file.name} in the run directory of {DUT_analysis.pseudopath}. ')
	with open(path_to_analysis_config_file, 'r') as ifile:
		analysis_config = json.load(ifile)
	
	for analysis_name in analysis_config.keys():
		logging.info(f'Launching `efficiency_vs_distance_left_right_folding` for {DUT_analysis.pseudopath}/{analysis_name} (force={force})')
		efficiency_vs_distance_left_right_folding(
			DUT_analysis = DUT_analysis,
			analysis_name = analysis_name,
			force = force,
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
	
	return IPD, IPD_error_up, IPD_error_down, interpolated['right']['val'](measure_at_efficiency), interpolated['left']['val'](measure_at_efficiency), measure_at_efficiency

def interpixel_distance_from_efficiency_vs_distance(efficiency_vs_distance_analysis:RunBureaucrat):
	efficiency_vs_distance_analysis.check_these_tasks_were_run_successfully('efficiency_vs_distance')
	
	with efficiency_vs_distance_analysis.handle_task('inter_pixel_distance') as employee:
		efficiency_data = pandas.read_pickle(efficiency_vs_distance_analysis.path_to_directory_of_task('efficiency_vs_distance')/'efficiency_vs_distance.pickle')
		
		IPD, IPD_error_up, IPD_error_down, IPD_measured_at_left, IPD_measured_at_right, measured_at_efficiency = calculate_interpixel_distance(
			efficiency_data = efficiency_data,
			measure_at_efficiency = .5,
		)
		utils.save_dataframe(
			pandas.Series(
				{
					'IPD (m)': IPD, 
					'IPD (m) error +': IPD_error_up,
					'IPD (m) error -': IPD_error_down,
				}
			),
			name = 'result',
			location = employee.path_to_directory_of_my_task,
		)
		
		fig = plotly_utils.line(
			data_frame = efficiency_data.sort_index().reset_index(drop=False),
			title = f'Inter-pixel distance calculation<br><sup>{efficiency_vs_distance_analysis.pseudopath}</sup>',
			x = 'Distance (m)',
			y = 'Efficiency',
			error_y = 'Efficiency error_+',
			error_y_minus = 'Efficiency error_-',
			color = 'Pixel',
			error_y_mode = 'bands',
		)
		arrow = go.layout.Annotation(
			dict(
				x = IPD_measured_at_left,
				y = measured_at_efficiency,
				ax = IPD_measured_at_right,
				ay = measured_at_efficiency,
				xref = "x", 
				yref = "y",
				text = "",
				showarrow = True,
				axref = "x", 
				ayref = 'y',
				arrowhead = 3,
				arrowwidth = 1.5,
			)
		)
		fig.update_layout(annotations=[arrow])
		fig.write_html(
			employee.path_to_directory_of_my_task/'IPD_measurement.html',
			include_plotlyjs = 'cdn',
		)

def interpixel_distance(TI_LGAD_analysis:RunBureaucrat, force:bool=False):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully(['this_is_a_TI-LGAD_analysis','efficiency_vs_distance_calculation'])
	
	if force==False and TI_LGAD_analysis.was_task_run_successfully('interpixel_distance'):
		return
	
	with TI_LGAD_analysis.handle_task('interpixel_distance') as employee:
		logging.info(f'Calculating IPD of {TI_LGAD_analysis.pseudopath}...')
		efficiency_data = []
		IPD = []
		for efficiency_analysis in TI_LGAD_analysis.list_subruns_of_task('efficiency_vs_distance_calculation'):
			interpixel_distance_from_efficiency_vs_distance(efficiency_analysis)
			_ = pandas.read_pickle(efficiency_analysis.path_to_directory_of_task('inter_pixel_distance')/'result.pickle')
			_['pixel_group'] = efficiency_analysis.run_name

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

def efficiency_increasing_centered_ROI(DUT_analysis:RunBureaucrat, analysis_name:str, force:bool=False):
	"""Calculates the efficiency sweeping a square ROI centered with the DUT 
	that increases in size. Expects to find a file named `efficiency_increasing_centered_ROI.config.json`
	in the run directory with a content like this:
	```
	{
		"centeredROI_CoincidenceWithTI145": {
			"DUT_hit_criterion": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9",
			"use_estimation_of_misreconstructed_tracks": false,
			"trigger_on_DUTs": {
				"TI145 (0,0)": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9",
				"TI145 (0,1)": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9",
				"TI145 (1,0)": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9",
				"TI145 (1,1)": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9"
			},
			"ROI_size_min": 3.3e-05,
			"ROI_size_max": 0.000666,
			"ROI_size_n_steps": 33
		}
	}
	```
	where the top level dictionary specifies different analyses with the
	key being the `analysis_name` and the items the analysis config.
	"""
	DUT_analysis.check_these_tasks_were_run_successfully(['this_is_a_TI-LGAD_analysis','transformation_for_centering_and_leveling'])
	LOCAL_PLOT_LABELS = {'hit_in_DUT': 'Detected by DUT', 'ROI_size (m)': 'ROI size (m)'}
	TASK_NAME = 'efficiency_increasing_centered_ROI'
	
	# The checking mechanism becomes a bit tricky because this function actually operates on a subrun...
	_this_analysis_bureaucrat = [_ for _ in DUT_analysis.list_subruns_of_task(TASK_NAME) if _.run_name==analysis_name]
	if len(_this_analysis_bureaucrat) > 1: # If it is 0, it was not run yet, if it is 1 it was.
		raise RuntimeError(f'This should have never happen, check!')
	if force == False and len(_this_analysis_bureaucrat)==1 and _this_analysis_bureaucrat[0].was_task_run_successfully(TASK_NAME):
		return
	
	# Read analysis config:
	path_to_analysis_config_file = DUT_analysis.path_to_run_directory/'efficiency_increasing_centered_ROI.config.json'
	if not path_to_analysis_config_file.is_file():
		raise RuntimeError(f'Cannot find analysis config file for {DUT_analysis.pseudopath}. You have to create a json file named {path_to_analysis_config_file.name} in the run directory of {DUT_analysis.pseudopath}. ')
	with open(path_to_analysis_config_file, 'r') as ifile:
		analysis_config = json.load(ifile)
		_analyses_names_in_config_file = analysis_config.keys()
		analysis_config = analysis_config.get(analysis_name)
		if analysis_config is None:
			raise RuntimeError(f'No analysis named "{analysis_name}" found in analysis config file for {DUT_analysis.pseudopath}. Analyses names found {sorted(_analyses_names_in_config_file)}')
	
	with DUT_analysis.handle_task(TASK_NAME, drop_old_data=False) as _:
		with _.create_subrun(analysis_name, if_exists='skip').handle_task(TASK_NAME) as employee:
			with open(employee.path_to_run_directory/path_to_analysis_config_file.name, 'w') as ofile:
				json.dump(
					analysis_config,
					ofile,
					indent = '\t',
				)
			
			setup_config = utils_batch_level.load_setup_configuration_info(DUT_analysis.parent)
			
			tracks = utils_batch_level.load_tracks(
				DUT_analysis.parent, 
				only_multiplicity_one = True,
				trigger_on_DUTs = analysis_config.get('trigger_on_DUTs'),
			)
			DUT_z_position = list(set(setup_config.query(f'DUT_name == "{DUT_analysis.run_name}"')['z (m)']))
			if len(DUT_z_position) != 1: # This should never happen, but just in case it is better to be prepared.
				raise RuntimeError(f'Cannot determine DUT z position for {DUT_analysis.pseudopath}. ')
			else:
				DUT_z_position = DUT_z_position[0]
			tracks = tracks.join(tracks_utils.project_tracks(tracks, z=DUT_z_position))
			
			DUT_hits = utils_batch_level.load_hits(
				TB_batch = DUT_analysis.parent,
				DUTs_and_hit_criterions = {DUT_name_rowcol:analysis_config['DUT_hit_criterion'] for DUT_name_rowcol in set(setup_config.query(f'DUT_name=="{DUT_analysis.run_name}"')['DUT_name_rowcol'])},
			)
			
			transformation_parameters = get_transformation_parameters(DUT_analysis)
			
			if analysis_config.get('use_estimation_of_misreconstructed_tracks') == True:
				DUT_analysis.check_these_tasks_were_run_successfully('estimate_fraction_of_misreconstructed_tracks')
				misreconstruction_probability = pandas.read_pickle(DUT_analysis.path_to_directory_of_task('estimate_fraction_of_misreconstructed_tracks')/'probability_that_corry_fails.pickle')
				misreconstruction_probability = ufloat(misreconstruction_probability['probability_corry_fails'],misreconstruction_probability['probability_corry_fails_error'])
			
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
					x_translation = transformation_parameters['x_translation'],
					y_translation = transformation_parameters['y_translation'],
					angle_rotation = transformation_parameters['rotation_around_z_deg']/180*numpy.pi,
				)
				total_area = (total_area_corners.loc['bottom_right','x']-total_area_corners.loc['bottom_left','x'])*(total_area_corners.loc['top_left','y']-total_area_corners.loc['bottom_left','y'])
				
				tracks_with_no_hit_in_the_DUT = tracks_utils.tag_tracks_with_DUT_hits(tracks, DUT_hits).query('DUT_name_rowcol == "no hit"')
				number_of_tracks_with_no_hit_in_the_DUT = len(tracks_with_no_hit_in_the_DUT.index.drop_duplicates()) # The "drop duplicates" thing is just in case to avoid double counting, anyhow it is not expected.
				number_of_noHitTrack_that_are_fake_per_unit_area = number_of_tracks_with_no_hit_in_the_DUT*misreconstruction_probability/total_area
			else:
				number_of_noHitTrack_that_are_fake_per_unit_area = ufloat(0,0)
			
			tracks[['Px','Py']] = translate_and_then_rotate(
				points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
				x_translation = transformation_parameters['x_translation'],
				y_translation = transformation_parameters['y_translation'],
				angle_rotation = transformation_parameters['rotation_around_z_deg']/180*numpy.pi,
			)
			
			tracks = tracks[['Px','Py']] # Keep only stuff that will be used.
			tracks.reset_index('n_track', inplace=True, drop=True) # Not used anymore.
			
			_ = DUT_hits.sum(axis=1)
			_.name = 'hit_in_DUT'
			_[_>0] = True
			tracks = tracks.join(_)
			tracks['hit_in_DUT'].fillna(False, inplace=True)
			
			ROI_sizes = numpy.linspace(analysis_config['ROI_size_min'], analysis_config['ROI_size_max'], analysis_config['ROI_size_n_steps'])
			
			efficiency = []
			save_plots_here = employee.path_to_directory_of_my_task/'plots'
			save_plots_here.mkdir()
			
			def calculate_efficiency(tracks, n_noHitTrack_that_are_fake):
				n_tracks_in_ROI = len(tracks.index.drop_duplicates()) # Drop duplicates is just in case, it should never happen.
				n_tracks_detected_in_ROI = len(tracks.query('hit_in_DUT==True').index.drop_duplicates()) # Drop duplicates is just in case, it should never happen.
				n_tracks_undetected_in_ROI = len(tracks.query('hit_in_DUT==False').index.drop_duplicates()) # Drop duplicates is just in case, it should never happen.
				
				try:
					_efficiency = ufloat(n_tracks_detected_in_ROI, n_tracks_detected_in_ROI**.5)/(ufloat(n_tracks_in_ROI, n_tracks_in_ROI**.5)-n_noHitTrack_that_are_fake)
				except ZeroDivisionError:
					_efficiency = ufloat(float('NaN'), float('NaN'))
				
				return _efficiency
			
			def estimate_efficiency_error(tracks, n_noHitTrack_that_are_fake, n_bootstraps:int, confidence_level:float=.68):
				replicas = []
				for n_bootstrap in range(n_bootstraps):
					_ = calculate_efficiency(
						tracks = tracks.sample(frac=1, replace=True),
						n_noHitTrack_that_are_fake = n_noHitTrack_that_are_fake,
					)
					replicas.append(_.nominal_value)
				replicas = numpy.array(replicas)
				value = numpy.quantile(replicas, q=.5, axis=0, method='interpolated_inverted_cdf')
				error_up = numpy.quantile(replicas, q=.5+confidence_level/2, axis=0, method='interpolated_inverted_cdf') - value
				error_down = value - numpy.quantile(replicas, q=.5-confidence_level/2, axis=0, method='interpolated_inverted_cdf')
				
				return error_down, error_up
			
			logging.info(f'Calculating efficiency vs ROI size for {employee.pseudopath}...')
			for ROI_size in ROI_sizes:
				tracks_within_ROI = tracks.query(f'Px>{-ROI_size/2} and Px<{ROI_size/2} and Py>{-ROI_size/2} and Py<{ROI_size/2}')
				_efficiency = calculate_efficiency(
					tracks = tracks_within_ROI,
					n_noHitTrack_that_are_fake = number_of_noHitTrack_that_are_fake_per_unit_area*ROI_size**2,
				)
				error_down, error_up = estimate_efficiency_error(
					tracks = tracks_within_ROI,
					n_noHitTrack_that_are_fake = number_of_noHitTrack_that_are_fake_per_unit_area*ROI_size**2,
					confidence_level = .99,
					n_bootstraps = 99,
				)
				
				efficiency.append(
					{
						'ROI_size (m)': ROI_size,
						'efficiency': _efficiency.nominal_value,
						'efficiency_error_up': error_up,
						'efficiency_error_down': error_down,
					}
				)
				
				# Do a plot ---
				fig = px.scatter(
					title = f'Efficiency calculation within centered ROI of size {ROI_size*1e6:.0f} µm<br><sup>{employee.pseudopath}</sup>',
					data_frame = tracks_within_ROI.reset_index(drop=False).sort_values('hit_in_DUT'),
					x = 'Px',
					y = 'Py',
					color = 'hit_in_DUT',
					hover_data = ['n_run','n_event'],
					labels = utils.PLOTS_LABELS | LOCAL_PLOT_LABELS,
				)
				fig.add_shape(
					type = "rect",
					x0 = -250e-6, 
					y0 = -250e-6, 
					x1 = 250e-6, 
					y1 = 250e-6,
					name = 'DUT',
					showlegend = True,
				)
				fig.add_shape(
					type = "rect",
					x0 = -ROI_size/2, 
					y0 = -ROI_size/2, 
					x1 = ROI_size/2, 
					y1 = ROI_size/2,
					line = dict(color="#15ab13"),
					name = 'ROI',
					showlegend = True,
				)
				fig.update_yaxes(
					scaleanchor = "x",
					scaleratio = 1,
				)
				fig.write_html(
					save_plots_here/f'tracks_with_ROI_{ROI_size*1e6:.0f}um.html',
					include_plotlyjs = 'cdn',
				)
			efficiency = pandas.DataFrame.from_records(efficiency).set_index('ROI_size (m)')
			
			utils.save_dataframe(
				efficiency,
				name = 'efficiency_vs_ROI_size',
				location = employee.path_to_directory_of_my_task,
			)
			
			fig = plotly_utils.line(
				data_frame = efficiency.sort_values('ROI_size (m)').reset_index(drop=False),
				title = f'Efficiency vs centered ROI size<br><sup>{employee.pseudopath}</sup>',
				x = 'ROI_size (m)',
				y = 'efficiency',
				error_y = 'efficiency_error_up',
				error_y_minus = 'efficiency_error_down',
				error_y_mode = 'band',
				labels = utils.PLOTS_LABELS | LOCAL_PLOT_LABELS,
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/'efficiency_vs_ROI_size.html',
				include_plotlyjs = 'cdn',
			)

def run_all_efficiency_increasing_centered_ROI(DUT_analysis:RunBureaucrat, force:bool=False):
	"""Runs all `efficiency_increasing_centered_ROI` analyses defined in
	the file `efficiency_increasing_centered_ROI.config.json` within a
	DUT analysis directory."""
	
	# Read analysis config:
	path_to_analysis_config_file = DUT_analysis.path_to_run_directory/'efficiency_increasing_centered_ROI.config.json'
	if not path_to_analysis_config_file.is_file():
		raise RuntimeError(f'Cannot find analysis config file for {DUT_analysis.pseudopath}. You have to create a json file named {path_to_analysis_config_file.name} in the run directory of {DUT_analysis.pseudopath}. ')
	with open(path_to_analysis_config_file, 'r') as ifile:
		analysis_config = json.load(ifile)
	
	for analysis_name in analysis_config.keys():
		logging.info(f'Launching `efficiency_increasing_centered_ROI` for {DUT_analysis.pseudopath}/{analysis_name} (force={force})')
		efficiency_increasing_centered_ROI(
			DUT_analysis = DUT_analysis,
			analysis_name = analysis_name,
			force = force,
		)

def plot_efficiency_2D(efficiency_analysis:RunBureaucrat, min_counts_cutoff:int):
	efficiency_analysis.check_these_tasks_were_run_successfully('efficiency_2D')
	if any([efficiency_analysis.was_task_run_successfully(_) for _ in {'this_is_a_TI-LGAD_analysis','this_is_an_RSD-LGAD_analysis'}]):
		raise RuntimeError(f'`efficiency_analysis` is pointing to {efficiency_analysis.pseudopath} which looks like an analysis of a DUT rather than an efficiency analysis...')
	
	efficiency = pandas.read_pickle(efficiency_analysis.path_to_directory_of_task('efficiency_2D')/'efficiency.pickle')
	
	for col in {'efficiency','efficiency_error'}:
		efficiency.loc[efficiency['total_count']<=min_counts_cutoff, col] = float('NaN') # Remove data in those places where there is almost no data.
	
	with efficiency_analysis.handle_task('plot_efficiency_vs_position') as employee:
		for col in {'efficiency','efficiency_error','detected_count','total_count'}:
			fig = px.imshow(
				efficiency.set_index(['x','y']).unstack('x')[col],
				title = f'{col} vs position<br><sup>{employee.pseudopath}</sup>',
				labels = {
					'x': 'x (m)',
					'y': 'y (m)',
				},
				aspect = 'equal',
				origin = 'lower',
			)
			fig.update_layout(
				coloraxis_colorbar_title_text = col,
			)
			fig.update_coloraxes(colorbar_title_side='right')
			for line_color, line_width in [('black',2.5),('white',1)]:
				fig.add_shape(
					type = "rect",
					x0 = -250e-6, 
					y0 = -250e-6, 
					x1 = 250e-6, 
					y1 = 250e-6,
					line = dict(
						color = line_color,
						width = line_width,
					),
				)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{col}_vs_position.html',
				include_plotlyjs = 'cdn',
			)
		
		# Efficiency in cuadratic scale, to better see details close to 1:
		power = 4
		fig = px.imshow(
			efficiency.set_index(['x','y']).unstack('x')['efficiency']**power,
			range_color = (0,1),
			title = f'efficiency vs position (x<sup>{power}</sup> scale)<br><sup>{employee.pseudopath}</sup>',
			labels = {
				'x': 'x (m)',
				'y': 'y (m)',
			},
			aspect = 'equal',
			origin = 'lower',
		)
		fig.update_layout(
			coloraxis_colorbar_title_text = 'efficiency',
		)
		COLOR_TICKS = [dict(val=_/10, text=f'{_/10:.1f}') for _ in [0,1,2,3,4,5,6,7,8,9,10]]
		fig.update_layout(
			coloraxis_colorbar = dict(
				tickvals = [_['val']**power for _ in COLOR_TICKS],
				ticktext = [_['text'] for _ in COLOR_TICKS],
			),
		)
		fig.update_coloraxes(colorbar_title_side='right')
		for line_color, line_width in [('black',2.5),('white',1)]:
			fig.add_shape(
				type = "rect",
				x0 = -250e-6, 
				y0 = -250e-6, 
				x1 = 250e-6, 
				y1 = 250e-6,
				line = dict(
					color = line_color,
					width = line_width,
				),
			)
		fig.write_html(
			employee.path_to_directory_of_my_task/f'efficiency_vs_position_nonlinear_scale.html',
			include_plotlyjs = 'cdn',
		)
	
def DUT_effective_efficiency(efficiency_analysis:RunBureaucrat):
	"""Computes the DUT effective efficiency from a 2D efficiency calculation
	assuming the following holds:
		- The bin size used for the 2D efficiency calculation is the same
		size as the pixels (the pitch).
	"""
	FUNCTION_SPECIFIC_LABELS_FOR_PLOTS = {
		'total_count': 'N events',
		'efficiency (%)': 'Efficiency (%)',
	}
	efficiency_analysis.check_these_tasks_were_run_successfully('efficiency_2D')
	if any([efficiency_analysis.was_task_run_successfully(_) for _ in {'this_is_a_TI-LGAD_analysis','this_is_an_RSD-LGAD_analysis'}]):
		raise RuntimeError(f'`efficiency_analysis` is pointing to {efficiency_analysis.pseudopath} which looks like an analysis of a DUT rather than an efficiency analysis...')
	
	with efficiency_analysis.handle_task('DUT_effective_efficiency') as employee:
		efficiency = pandas.read_pickle(efficiency_analysis.path_to_directory_of_task('efficiency_2D')/'efficiency.pickle')
		
		for col in {'efficiency','efficiency_error'}:
			efficiency[f'{col} (%)'] = efficiency[col]*100
		# ~ efficiency['efficiency_ufloat'] = efficiency[['efficiency','efficiency_error']].apply(lambda x: ufloat(x['efficiency'],x['efficiency_error']), axis=1)
		
		with open(efficiency_analysis.path_to_directory_of_task('efficiency_2D.config.json'), 'r') as ifile:
			analysis_config = json.load(ifile)
		
		# Define the ROI, only those bins that lie completely inside the DUT
		PIXEL_SIZE = 250e-6
		SAFETY_MARGIN_METERS = 22e-6
		x_max =  PIXEL_SIZE - analysis_config['bin_size_x']/2 - analysis_config['x_step'] - SAFETY_MARGIN_METERS
		x_min = -PIXEL_SIZE + analysis_config['bin_size_x']/2 + analysis_config['x_step'] + SAFETY_MARGIN_METERS
		y_max =  PIXEL_SIZE - analysis_config['bin_size_y']/2 - analysis_config['y_step'] - SAFETY_MARGIN_METERS
		y_min = -PIXEL_SIZE + analysis_config['bin_size_y']/2 + analysis_config['y_step'] + SAFETY_MARGIN_METERS
		
		efficiency['within_ROI'] = False
		efficiency.loc[(efficiency['x']>x_min) & (efficiency['x']<x_max) & (efficiency['y']>y_min) & (efficiency['y']<y_max), 'within_ROI'] = True
		efficiency = efficiency.sample(frac=1, replace=False).sort_values('within_ROI', ascending=False)
		efficiency['Bin number'] = numpy.arange(len(efficiency)) # Assign a measurement number to each bin.
		
		# Compute effective efficiency:
		effective_efficiency = ufloat(
			efficiency.query('within_ROI==True')['efficiency'].mean(),
			efficiency.query('within_ROI==True')['efficiency_error'].mean(),
		)
		utils.save_dataframe(
			pandas.Series(
				dict(
					effective_efficiency = effective_efficiency.nominal_value,
					effective_efficiency_error = effective_efficiency.std_dev,
				)
			),
			'effective_efficiency',
			employee.path_to_directory_of_my_task,
		)
		# Plot each of the different measurements of the efficiency we have:
		fig = px.scatter(
			data_frame = efficiency.reset_index(drop=False),
			x = 'Bin number',
			y = 'efficiency (%)',
			color = 'within_ROI',
			error_y = 'efficiency_error (%)',
			title = f'Effective efficiency<br><sup>{employee.pseudopath}</sup>',
			hover_data = ['total_count'],
			labels = utils.PLOTS_LABELS | FUNCTION_SPECIFIC_LABELS_FOR_PLOTS,
		)
		fig.add_hrect(
			y0 = (effective_efficiency.nominal_value - effective_efficiency.std_dev)*100,
			y1 = (effective_efficiency.nominal_value + effective_efficiency.std_dev)*100,
			line_width = 0,
			fillcolor = 'black', 
			opacity = 0.2,
		)
		fig.add_hline(
			y = effective_efficiency.nominal_value*100,
			line_color = 'black',
			annotation = dict(
				text = f'{effective_efficiency*100} %'.replace('+/-','±'),
				font_color = 'white',
				bgcolor = 'rgba(0,0,0,.8)',
			),
			annotation_position = 'top left',
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'effective_efficiency.html',
			include_plotlyjs = 'cdn',
		)
		
		# Heatmap with the efficiency:
		df = efficiency
		fig = px.imshow(
			df.set_index(['x','y']).unstack('x')['efficiency (%)'],
			# ~ range_color = (0,100),
			title = f'efficiency vs position<br><sup>{employee.pseudopath}</sup>',
			labels = {
				'x': 'x (m)',
				'y': 'y (m)',
			},
			aspect = 'equal',
			origin = 'lower',
		)
		hover_text_data = df.set_index(['within_ROI','x','y'], append=True).apply(lambda x: f'Bin number: {x["Bin number"]:.0f}<br>Efficiency: {ufloat(x["efficiency (%)"],x["efficiency_error (%)"])} %<br>N events: {x["total_count"]:.0f}'.replace('+/-',' ± '), axis=1)
		hover_text_data.name = 'text_for_plot'
		hover_text_data = hover_text_data.to_frame().reset_index()
		fig.update(
			data = [
				{
					'customdata': hover_text_data.set_index(['x','y']).unstack('x')['text_for_plot'],
					'hovertemplate': '%{customdata}<br>Bin x: %{x}<br>Bin y: %{y}',
				}
			]
		)
		fig.update_layout(
			coloraxis_colorbar_title_text = 'Efficiency (%)',
		)
		fig.update_coloraxes(colorbar_title_side='right')
		
		PIXEL_SIZE = 250e-6
		draw_2x2_DUT(fig, pixel_size=PIXEL_SIZE)
		
		fig.add_shape(
			type = "rect",
			x0 = x_min,
			y0 = y_min,
			x1 = x_max,
			y1 = y_max,
			line = dict(
				color = 'black',
				dash = 'dash',
			),
		)
		# ~ fig.add_annotation(
			# ~ x = PIXEL_SIZE*.1,
			# ~ y = y_max,
			# ~ text = f'Efficiency = {ufloat(efficiency_stats.loc["mean","efficiency"], efficiency_stats.loc["std","efficiency"])*100} %'.replace('+/-','±'),
			# ~ bgcolor = 'white',
			# ~ arrowcolor = 'black',
			# ~ bordercolor = 'black',
		# ~ )
		fig.write_html(
			employee.path_to_directory_of_my_task/f'efficiency_vs_position.html',
			include_plotlyjs = 'cdn',
		)

def efficiency_2D(DUT_analysis:RunBureaucrat, analysis_name:str, force:bool=False):
	"""Calculates the efficiency sweeping a square ROI along the surface
	in the x y plane. Expects to find a file named `efficiency_2D.config.json`
	in the run directory with a content like this:
	```
	{
		"my_2D_efficiency_calculation": {
			"DUT_hit_criterion": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9",
			"use_estimation_of_misreconstructed_tracks": false,
			"trigger_on_DUTs": {
				"TI145 (0,0)": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9",
				"TI145 (0,1)": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9",
				"TI145 (1,0)": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9",
				"TI145 (1,1)": "`Amplitude (V)`<-5e-3 AND 100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9"
			},
			"xmin": -222e-6,
			"xmax": 222e-6,
			"x_step": 11e-6,
			"bin_size_x": 22e-6,
			"ymin": -222e-6,
			"ymax": 222e-6,
			"y_step": 11e-6,
			"bin_size_y": 22e-6,
		}
	}
	```
	where the top level dictionary specifies different analyses with the
	key being the `analysis_name` and the items the analysis config.
	"""
	DUT_analysis.check_these_tasks_were_run_successfully(['this_is_a_TI-LGAD_analysis','transformation_for_centering_and_leveling'])
	LOCAL_PLOT_LABELS = {'hit_in_DUT': 'Detected by DUT', 'ROI_size (m)': 'ROI size (m)'}
	TASK_NAME = 'efficiency_2D'
	
	# The checking mechanism becomes a bit tricky because this function actually operates on a subrun...
	_this_analysis_bureaucrat = [_ for _ in DUT_analysis.list_subruns_of_task(TASK_NAME) if _.run_name==analysis_name]
	if len(_this_analysis_bureaucrat) > 1: # If it is 0, it was not run yet, if it is 1 it was.
		raise RuntimeError(f'This should have never happen, check!')
	if force == False and len(_this_analysis_bureaucrat)==1 and _this_analysis_bureaucrat[0].was_task_run_successfully(TASK_NAME):
		return
	
	# Read analysis config:
	path_to_analysis_config_file = DUT_analysis.path_to_run_directory/f'{TASK_NAME}.config.json'
	if not path_to_analysis_config_file.is_file():
		raise RuntimeError(f'Cannot find analysis config file for {DUT_analysis.pseudopath}. You have to create a json file named {path_to_analysis_config_file.name} in the run directory of {DUT_analysis.pseudopath}. ')
	with open(path_to_analysis_config_file, 'r') as ifile:
		analysis_config = json.load(ifile)
		_analyses_names_in_config_file = analysis_config.keys()
		analysis_config = analysis_config.get(analysis_name)
		if analysis_config is None:
			raise RuntimeError(f'No analysis named "{analysis_name}" found in analysis config file for {DUT_analysis.pseudopath}. Analyses names found {sorted(_analyses_names_in_config_file)}')
	
	with DUT_analysis.handle_task(TASK_NAME, drop_old_data=False) as _:
		with _.create_subrun(analysis_name, if_exists='skip').handle_task(TASK_NAME) as employee:
			with open(employee.path_to_run_directory/path_to_analysis_config_file.name, 'w') as ofile:
				json.dump(
					analysis_config,
					ofile,
					indent = '\t',
				)
			
			setup_config = utils_batch_level.load_setup_configuration_info(DUT_analysis.parent)
			
			tracks = utils_batch_level.load_tracks(
				DUT_analysis.parent, 
				only_multiplicity_one = True,
				trigger_on_DUTs = analysis_config.get('trigger_on_DUTs'),
			)
			DUT_z_position = list(set(setup_config.query(f'DUT_name == "{DUT_analysis.run_name}"')['z (m)']))
			if len(DUT_z_position) != 1: # This should never happen, but just in case it is better to be prepared.
				raise RuntimeError(f'Cannot determine DUT z position for {DUT_analysis.pseudopath}. ')
			else:
				DUT_z_position = DUT_z_position[0]
			tracks = tracks.join(tracks_utils.project_tracks(tracks, z=DUT_z_position))
			
			DUT_hits = utils_batch_level.load_hits(
				TB_batch = DUT_analysis.parent,
				DUTs_and_hit_criterions = {DUT_name_rowcol:analysis_config['DUT_hit_criterion'] for DUT_name_rowcol in set(setup_config.query(f'DUT_name=="{DUT_analysis.run_name}"')['DUT_name_rowcol'])},
			)
			
			transformation_parameters = get_transformation_parameters(DUT_analysis)
			
			if analysis_config.get('use_estimation_of_misreconstructed_tracks') == True:
				DUT_analysis.check_these_tasks_were_run_successfully('estimate_fraction_of_misreconstructed_tracks')
				misreconstruction_probability = pandas.read_pickle(DUT_analysis.path_to_directory_of_task('estimate_fraction_of_misreconstructed_tracks')/'probability_that_corry_fails.pickle')
				misreconstruction_probability = ufloat(misreconstruction_probability['probability_corry_fails'],misreconstruction_probability['probability_corry_fails_error'])
			
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
					x_translation = transformation_parameters['x_translation'],
					y_translation = transformation_parameters['y_translation'],
					angle_rotation = transformation_parameters['rotation_around_z_deg']/180*numpy.pi,
				)
				total_area = (total_area_corners.loc['bottom_right','x']-total_area_corners.loc['bottom_left','x'])*(total_area_corners.loc['top_left','y']-total_area_corners.loc['bottom_left','y'])
				
				tracks_with_no_hit_in_the_DUT = tracks_utils.tag_tracks_with_DUT_hits(tracks, DUT_hits).query('DUT_name_rowcol == "no hit"')
				number_of_tracks_with_no_hit_in_the_DUT = len(tracks_with_no_hit_in_the_DUT.index.drop_duplicates()) # The "drop duplicates" thing is just in case to avoid double counting, anyhow it is not expected.
				number_of_noHitTrack_that_are_fake_per_unit_area = number_of_tracks_with_no_hit_in_the_DUT*misreconstruction_probability/total_area
			else:
				number_of_noHitTrack_that_are_fake_per_unit_area = ufloat(0,0)
			
			tracks[['Px','Py']] = translate_and_then_rotate(
				points = tracks[['Px','Py']].rename(columns=dict(Px='x',Py='y')),
				x_translation = transformation_parameters['x_translation'],
				y_translation = transformation_parameters['y_translation'],
				angle_rotation = transformation_parameters['rotation_around_z_deg']/180*numpy.pi,
			)
			
			tracks = tracks[['Px','Py']] # Keep only stuff that will be used.
			tracks.reset_index('n_track', inplace=True, drop=True) # Not used anymore.
			
			def calculate_2D_efficiency(tracks:pandas.DataFrame, DUT_hits:pandas.Index, xmin:float, xmax:float, x_step:float, bin_size_x:float, ymin:float, ymax:float, y_step:float, bin_size_y:float, number_of_noHitTrack_that_are_fake_per_unit_area:float):
				if DUT_hits.names != tracks.index.names:
					raise ValueError(f'The level names of `tracks.index` must be the same as those in `DUT_hits`. ')
				
				detected_tracks = utils.select_by_multiindex(tracks, DUT_hits)
				efficiency = []
				for nx,x in enumerate(numpy.linspace(xmin,xmax,int((xmax-xmin)/x_step))):
					for ny,y in enumerate(numpy.linspace(ymin,ymax,int((ymax-ymin)/y_step))):
						detected_here = detected_tracks.query(f'Px>{x-bin_size_x/2} and Px<{x+bin_size_x/2} and Py>{y-bin_size_y/2} and Py<{y+bin_size_y/2}')
						total_here = tracks.query(f'Px>{x-bin_size_x/2} and Px<{x+bin_size_x/2} and Py>{y-bin_size_y/2} and Py<{y+bin_size_y/2}')
						
						detected_count = len(detected_here.index.drop_duplicates())
						total_count = len(total_here.index.drop_duplicates())
						
						detected_count = ufloat(detected_count, detected_count**.5)
						total_count = ufloat(total_count, total_count**.5)
						
						try:
							efficiency_here = detected_count/(total_count - number_of_noHitTrack_that_are_fake_per_unit_area*bin_size_x*bin_size_y)
						except ZeroDivisionError:
							efficiency_here = ufloat(float('NaN'), float('NaN'))
						
						efficiency.append(
							{
								'x': x,
								'n_x': nx,
								'y': y,
								'n_y': ny,
								'efficiency': efficiency_here.nominal_value,
								'efficiency_error': efficiency_here.std_dev,
								'detected_count': int(detected_count.nominal_value),
								'total_count': int(total_count.nominal_value),
							}
						)
				return pandas.DataFrame.from_records(efficiency).set_index(['n_x','n_y'])
			
			logging.info(f'Calculating efficiency 2D for {employee.pseudopath}...')
			efficiency = calculate_2D_efficiency(
				tracks = tracks,
				DUT_hits = DUT_hits.index,
				number_of_noHitTrack_that_are_fake_per_unit_area = number_of_noHitTrack_that_are_fake_per_unit_area,
				**{_: analysis_config[_] for _ in {'xmin','xmax','x_step','bin_size_x','ymin','ymax','y_step','bin_size_y'}},
			)
			
			utils.save_dataframe(
				efficiency,
				name = 'efficiency',
				location = employee.path_to_directory_of_my_task,
			)
			
	plot_efficiency_2D(
		efficiency_analysis = employee.boss,
		min_counts_cutoff = 5,
	)
	if 'EfctvEff' in employee.run_name: # This means it is interesting to run the DUT effective efficiency calculation, because the parameters should be accordingly set for this.
		DUT_effective_efficiency(
			efficiency_analysis = employee.boss,
		)

def run_all_efficiency_2D(DUT_analysis:RunBureaucrat, force:bool=False):
	"""Runs all `efficiency_2D` analyses defined in
	the file `efficiency_2D.config.json` within a
	DUT analysis directory."""
	
	# Read analysis config:
	path_to_analysis_config_file = DUT_analysis.path_to_run_directory/'efficiency_2D.config.json'
	if not path_to_analysis_config_file.is_file():
		raise RuntimeError(f'Cannot find analysis config file for {DUT_analysis.pseudopath}. You have to create a json file named {path_to_analysis_config_file.name} in the run directory of {DUT_analysis.pseudopath}. ')
	with open(path_to_analysis_config_file, 'r') as ifile:
		analysis_config = json.load(ifile)
	
	for analysis_name in analysis_config.keys():
		logging.info(f'Launching `efficiency_2D` for {DUT_analysis.pseudopath}/{analysis_name} (force={force})')
		efficiency_2D(
			DUT_analysis = DUT_analysis,
			analysis_name = analysis_name,
			force = force,
		)

def run_all_analyses_in_a_TILGAD(TI_LGAD_analysis:RunBureaucrat, force:bool=False):
	plot_DUT_distributions(TI_LGAD_analysis, force=force)
	plot_tracks_and_hits(TI_LGAD_analysis, do_3D_plot=False, force=force)
	transformation_for_centering_and_leveling(TI_LGAD_analysis, force=force)
	estimate_fraction_of_misreconstructed_tracks(TI_LGAD_analysis, force=force)
	efficiency_vs_distance_calculation(TI_LGAD_analysis, force=force)
	interpixel_distance(TI_LGAD_analysis, force=force)
	plot_cluster_size(TI_LGAD_analysis, force=force)

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
	
	if True: # This is so I can easily fold all this block...
		parser = argparse.ArgumentParser()
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
			'--efficiency_vs_distance_left_right',
			help = 'Pass this flag to run `efficiency_vs_distance_left_right`. This option works together with `--analysis_name`, if an analysis name is provided then `efficiency_vs_distance_left_right` is only run for such analysis, if no analysis name is provided then `efficiency_vs_distance_left_right` is run for all the analysis present in the config file.',
			required = False,
			dest = 'efficiency_vs_distance_left_right',
			action = 'store_true'
		)
		parser.add_argument(
			'--efficiency_vs_distance_left_right_folding',
			help = 'Pass this flag to run `efficiency_vs_distance_left_right_folding`. This option works together with `--analysis_name`, if an analysis name is provided then `efficiency_vs_distance_left_right_folding` is only run for such analysis, if no analysis name is provided then `efficiency_vs_distance_left_right_folding` is run for all the analysis present in the config file.',
			required = False,
			dest = 'efficiency_vs_distance_left_right_folding',
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
		parser.add_argument(
			'--efficiency_increasing_centered_ROI',
			help = 'If this flag is passed, it will execute `efficiency_increasing_centered_ROI`.',
			required = False,
			dest = 'efficiency_increasing_centered_ROI',
			action = 'store_true'
		)
		parser.add_argument(
			'--efficiency_2D',
			help = 'If this flag is passed, it will execute `efficiency_2D`.',
			required = False,
			dest = 'efficiency_2D',
			action = 'store_true'
		)
		parser.add_argument(
			'--analysis_name',
			help = 'Specifies the name of the analysis to run, if it has to be chosen from a config file with multiple analyses.',
			required = False,
			dest = 'analysis_name',
			type = str,
		)
		args = parser.parse_args()
	
	bureaucrat = RunBureaucrat(Path(args.directory))
	
	if bureaucrat.was_task_run_successfully('this_is_a_TI-LGAD_analysis'):
		if args.plot_DUT_distributions == True:
			plot_DUT_distributions(bureaucrat, force=args.force)
		if args.plot_tracks_and_hits == True:
			plot_tracks_and_hits(bureaucrat, force=args.force)
		if args.transformation_for_centering_and_leveling == True:
			transformation_for_centering_and_leveling(bureaucrat, draw_square=True, force=args.force)
		if args.efficiency_vs_distance_left_right == True:
			if args.analysis_name is not None:
				efficiency_vs_distance_left_right(bureaucrat, force=args.force, analysis_name=args.analysis_name)
			else:
				run_all_efficiency_vs_distance_left_right(bureaucrat, force=args.force)
		if args.efficiency_vs_distance_left_right_folding == True:
			if args.analysis_name is not None:
				efficiency_vs_distance_left_right_folding(bureaucrat, force=args.force, analysis_name=args.analysis_name)
			else:
				run_all_efficiency_vs_distance_left_right_folding(bureaucrat, force=args.force)
		if args.estimate_fraction_of_misreconstructed_tracks == True:
			estimate_fraction_of_misreconstructed_tracks(bureaucrat, force=args.force)
		if args.interpixel_distance == True:
			interpixel_distance(bureaucrat, force=args.force)
		if args.plot_cluster_size == True:
			plot_cluster_size(bureaucrat, force=args.force)
		if args.run_all_analyses_in_a_TILGAD:
			run_all_analyses_in_a_TILGAD(bureaucrat, force=args.force)
		if args.efficiency_2D:
			if args.analysis_name is not None:
				efficiency_2D(bureaucrat, force=args.force, analysis_name=args.analysis_name)
			else:
				run_all_efficiency_2D(bureaucrat, force=args.force)
		if args.efficiency_increasing_centered_ROI:
			if args.analysis_name is not None:
				efficiency_increasing_centered_ROI(bureaucrat, force=args.force, analysis_name=args.analysis_name)
			else:
				run_all_efficiency_increasing_centered_ROI(bureaucrat, force=args.force)
	elif args.setup_analysis_for_DUT != 'None':
		setup_TI_LGAD_analysis_within_batch(
			bureaucrat, 
			DUT_name = args.setup_analysis_for_DUT,
		)
	else:
		raise RuntimeError(f"Don't know what to do in {bureaucrat.path_to_run_directory}... Please read script help or source code.")
