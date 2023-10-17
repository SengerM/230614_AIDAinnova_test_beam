from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import utils
import logging
import pandas
import sqlite3
import plotly.express as px
import numpy
import plotly_utils
from corry_stuff import load_tracks_from_batch
import json
import warnings
from parse_waveforms import read_parsed_from_waveforms_from_batch
import multiprocessing

def load_analyses_config():
	logging.info(f'Reading analyses config from the cloud...')
	analyses = pandas.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vTaR20eM5ZQxtizmZiaAtHooE7hWYfSixSgc1HD5sVNZT_RNxZKmhI09wCEtXEVepjM8NB1n8BUBZnc/pub?gid=0&single=true&output=csv').set_index(['test_beam_campaign','batch_name','DUT_name']).query('DUT_type=="TI-LGAD"')
	return analyses

def load_this_TILGAD_analysis_config(TI_LGAD_analysis:RunBureaucrat):
	TB_batch = TI_LGAD_analysis.parent
	TB_campaign = TB_batch.parent
	analysis_config = load_analyses_config()
	return analysis_config.loc[(TB_campaign.run_name,TB_batch.run_name,TI_LGAD_analysis.run_name)]

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

def plot_DUT_distributions(TI_LGAD_analysis_bureaucrat:RunBureaucrat):
	"""Plot some raw distributions, useful to configure cuts and thresholds
	for further steps in the analysis."""
	MAXIMUM_NUMBER_OF_EVENTS = 9999
	TI_LGAD_analysis_bureaucrat.check_these_tasks_were_run_successfully('this_is_a_TI-LGAD_analysis') # To be sure we are inside what is supposed to be a TI-LGAD analysis.
	
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

def plot_tracks_and_hits(TI_LGAD_analysis:RunBureaucrat, do_3D_plot:bool=True):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully('this_is_a_TI-LGAD_analysis')
	
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

def translate_and_rotate_tracks(tracks:pandas.DataFrame, x_translation:float, y_translation:float, angle_rotation:float):
	for xy,translation in {'x':x_translation,'y':y_translation}.items():
			tracks[f'P{xy}'] += translation
	r = (tracks['Px']**2 + tracks['Py']**2)**.5
	phi = numpy.arctan2(tracks['Py'], tracks['Px'])
	tracks['Px'], tracks['Py'] = r*numpy.cos(phi+angle_rotation), r*numpy.sin(phi+angle_rotation)
	return tracks

def transformation_for_centering_and_leveling(TI_LGAD_analysis:RunBureaucrat, draw_square:bool=False):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully('this_is_a_TI-LGAD_analysis')
	
	with TI_LGAD_analysis.handle_task('transformation_for_centering_and_leveling') as employee:
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
		
		logging.info('Applying transformation to tracks to center and align DUT...')
		tracks = translate_and_rotate_tracks(
			tracks = tracks,
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

def efficiency_vs_1D_distance_rolling(tracks:pandas.DataFrame, DUT_hits, project_on:str, distances:numpy.array, window_size:float):
	if set(tracks.index.names) != set(DUT_hits.names):
		raise ValueError('The index levels of `tracks` and `DUT_hits` must be the same, they are not.')
	if set(tracks.columns) != {'x','y'}:
		raise ValueError('The columns of `tracks` must be "x" and "y"')
	if project_on not in {'x','y'}:
		raise ValueError('`project_on` must be "x" or "y"')
	
	total_tracks_count = distances*0
	DUT_hits_count = distances*0
	for i,d in enumerate(distances):
		total_tracks_count[i] = len(tracks.query(f'{d-window_size/2}<={project_on} and {project_on}<{d+window_size/2}'))
		DUT_hits_count[i] = len(utils.select_by_multiindex(tracks, DUT_hits).query(f'{d-window_size/2}<={project_on} and {project_on}<{d+window_size/2}'))
	return DUT_hits_count/total_tracks_count

def efficiency_vs_1D_distance_rolling_error_estimation(tracks:pandas.DataFrame, DUT_hits, project_on:str, distances:numpy.array, window_size:float, n_bootstraps:int=99, confidence_level:float=.68):
	original_efficiency = efficiency_vs_1D_distance_rolling(
		tracks = tracks,
		DUT_hits = DUT_hits,
		project_on = project_on,
		distances = distances,
		window_size = window_size,
	)
	replicas = []
	for n_bootstrap in range(n_bootstraps):
		efficiency = efficiency_vs_1D_distance_rolling(
			tracks = tracks.sample(frac=1, replace=True),
			DUT_hits = DUT_hits,
			project_on = project_on,
			distances = distances,
			window_size = window_size,
		)
		replicas.append(efficiency)
	replicas = numpy.array(replicas)
	value = numpy.quantile(replicas, q=.5, axis=0, method='interpolated_inverted_cdf')
	error_up = numpy.quantile(replicas, q=.5+confidence_level/2, axis=0, method='interpolated_inverted_cdf') - value
	error_down = value - numpy.quantile(replicas, q=.5-confidence_level/2, axis=0, method='interpolated_inverted_cdf')
	
	# ~ try:
		# ~ error_up[value==0] = sorted(set(error_up))[1]
	# ~ except Exception:
		# ~ pass
	# ~ try:
		# ~ error_down[value==1] = sorted(set(error_down))[1]
	# ~ except Exception:
		# ~ pass
	
	return error_down, error_up

def efficiency_vs_distance_calculation(TI_LGAD_analysis:RunBureaucrat, force:bool=False):
	TI_LGAD_analysis.check_these_tasks_were_run_successfully('this_is_a_TI-LGAD_analysis')
	
	TASK_NAME = 'efficiency_vs_distance_calculation'
	
	if force == False and TI_LGAD_analysis.was_task_run_successfully(TASK_NAME):
		return
	
	with TI_LGAD_analysis.handle_task(TASK_NAME) as employee:
		batch = TI_LGAD_analysis.parent
		
		analysis_config = load_this_TILGAD_analysis_config(TI_LGAD_analysis)
		
		tracks = load_tracks_from_batch(batch, only_multiplicity_one=True)
		tracks.reset_index('n_track', inplace=True) # We are loading with multiplicity one, so this is not needed anymore.
		
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
		
		DUT_hits = DUT_hits.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])[['DUT_name_rowcol','row','col']])
		
		tracks = tracks.join(DUT_hits['DUT_name_rowcol'])
		
		tracks['DUT_name_rowcol'] = tracks['DUT_name_rowcol'].fillna('no hit')
		tracks = tracks.sort_values('DUT_name_rowcol', ascending=False)
		
		logging.info('Applying transformation to tracks to center and align DUT...')
		tracks = translate_and_rotate_tracks(
			tracks = tracks,
			x_translation = analysis_config['x_translation'],
			y_translation = analysis_config['y_translation'],
			angle_rotation = analysis_config['rotation_around_z_deg']/180*numpy.pi,
		)
		
		################################################################
		# Some hardcoded stuff #########################################
		################################################################
		PIXEL_SIZE = 250e-6
		ROI_DISTANCE_OFFSET = 50e-6
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
		CALCULATION_STEP = 22e-6
		ROLLING_WINDOW_SIZE = 11e-6
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
				
				tracks_for_efficiency_calculation = tracks.query(f'{xmin}<Px and Px<{xmax} and {ymin}<Py and Py<{ymax}')
				tracks_for_efficiency_calculation = tracks_for_efficiency_calculation.reset_index(['n_CAEN','CAEN_n_channel'], drop=True)
				
				fig = px.scatter(
					tracks_for_efficiency_calculation.reset_index(),
					title = f'Tracks projected on the DUT after transformation<br><sup>{which_pixels_analysis.pseudopath}</sup>',
					x = 'Px',
					y = 'Py',
					color = 'DUT_name_rowcol',
					hover_data = ['n_run','n_event'],
					labels = {
						'Px': 'x (m)',
						'Py': 'y (m)',
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
				
				distance_axis = numpy.arange(
					start = tracks_for_efficiency_calculation[f'P{project_on}'].min(),
					stop = tracks_for_efficiency_calculation[f'P{project_on}'].max(),
					step = CALCULATION_STEP,
				)
				efficiency_data = []
				for leftright in ['left','right','both']:
					if leftright in {'left','right'}:
						pixel_rowcol = PIXEL_DEPENDENT_SETTINGS[which_pixels][f'{leftright}_pixel_rowcol']
						DUT_hits_for_efficiency = DUT_hits.query(f'row=={pixel_rowcol[0]} and col=={pixel_rowcol[1]}')
					elif leftright == 'both':
						pixel_rowcol = [PIXEL_DEPENDENT_SETTINGS[which_pixels][f'{_}_pixel_rowcol'] for _ in {'left','right'}]
						DUT_hits_for_efficiency = DUT_hits.query(f'(row=={pixel_rowcol[0][0]} and col=={pixel_rowcol[0][1]}) or (row=={pixel_rowcol[1][0]} and col=={pixel_rowcol[1][1]})')
					else:
						raise RuntimeError('Check this, should never happen!')
					
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						efficiency_calculation_args = dict(
							tracks = tracks_for_efficiency_calculation.rename(columns={'Px':'x', 'Py':'y'})[['x','y']],
							DUT_hits = DUT_hits_for_efficiency.reset_index(['n_CAEN','CAEN_n_channel'], drop=True).index,
							project_on = project_on,
							distances = distance_axis,
							window_size = ROLLING_WINDOW_SIZE,
						)
						
						error_minus, error_plus = efficiency_vs_1D_distance_rolling_error_estimation(
							**efficiency_calculation_args,
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

def run_all_analyses_in_a_TILGAD(TI_LGAD_analysis:RunBureaucrat):
	plot_DUT_distributions(TI_LGAD_analysis)
	plot_tracks_and_hits(TI_LGAD_analysis, do_3D_plot=False)
	transformation_for_centering_and_leveling(TI_LGAD_analysis)
	efficiency_vs_distance_calculation(TI_LGAD_analysis)

def execute_all_analyses():
	TB_bureaucrat = RunBureaucrat(Path('/media/msenger/230829_gray/AIDAinnova_test_beams/TB'))
	
	analyses_config = load_analyses_config()
	
	# First of all, create directories in which to perform the data analysis if they don't exist already...
	for (campaign_name,batch_name,DUT_name),_ in analyses_config.iterrows():
		TI_LGAD_analysis = RunBureaucrat(TB_bureaucrat.path_to_run_directory/'campaigns/subruns'/campaign_name/'batches/subruns'/batch_name/'TI-LGADs_analyses/subruns'/DUT_name) # This is ugly, but currently I have no other way of getting it...
		
		if TI_LGAD_analysis.exists() == False and any([analyses_config.loc[(campaign_name,batch_name,DUT_name),_]==True for _ in {'top_row','bottom_row','left_column','right_column'}]):
			setup_TI_LGAD_analysis_within_batch(TI_LGAD_analysis.parent, TI_LGAD_analysis.run_name)
	
	for campaign in TB_bureaucrat.list_subruns_of_task('campaigns'):
		if 'august' in  campaign.run_name.lower():
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
	args = parser.parse_args()
	
	bureaucrat = RunBureaucrat(Path(args.directory))
	
	if bureaucrat.was_task_run_successfully('this_is_a_TI-LGAD_analysis'):
		if args.plot_DUT_distributions == True:
			plot_DUT_distributions(bureaucrat)
		if args.plot_tracks_and_hits == True:
			plot_tracks_and_hits(bureaucrat, do_3D_plot=args.enable_3D_tracks_plot)
		if args.transformation_for_centering_and_leveling == True:
			transformation_for_centering_and_leveling(bureaucrat, draw_square=True)
		if args.efficiency_vs_distance_calculation == True:
			efficiency_vs_distance_calculation(bureaucrat, force=args.force)
	elif args.setup_analysis_for_DUT != 'None':
		setup_TI_LGAD_analysis_within_batch(
			bureaucrat, 
			DUT_name = args.setup_analysis_for_DUT,
		)
	else:
		raise RuntimeError(f"Don't know what to do in {bureaucrat.path_to_run_directory}... Please read script help or source code.")
