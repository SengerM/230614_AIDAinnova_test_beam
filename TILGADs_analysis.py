from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import utils
import logging
import pandas
import sqlite3
import plotly.express as px
import numpy
import plotly_utils
from corry_stuff import load_tracks_from_run
import json
import warnings

def load_analysis_config(bureaucrat:RunBureaucrat):
	bureaucrat.check_these_tasks_were_run_successfully('TI_LGAD_analysis_setup')
	with open(bureaucrat.path_to_run_directory/'analysis_configuration.json', 'r') as ifile:
		return json.load(ifile)

def setup_TI_LGAD_analysis_within_batch(bureaucrat:RunBureaucrat, DUT_name:str)->RunBureaucrat:
	"""Setup a directory structure to perform further analysis of a TI-LGAD
	that is inside a batch pointed by `bureaucrat`. This should be the 
	first step before starting a TI-LGAD analysis."""
	bureaucrat.check_these_tasks_were_run_successfully(['runs','batch_info'])
	
	with bureaucrat.handle_task('TI-LGADs_analyses', drop_old_data=False) as employee:
		setup_configuration_info = utils.load_setup_configuration_info(bureaucrat)
		if DUT_name not in set(setup_configuration_info['DUT_name']):
			raise RuntimeError(f'DUT_name {repr(DUT_name)} not present within the set of DUTs in {bureaucrat.pseudopath}, which is {set(setup_configuration_info["DUT_name"])}')
		
		try:
			TILGAD_bureaucrat = employee.create_subrun(DUT_name)
			with TILGAD_bureaucrat.handle_task('this_is_a_TI-LGAD_analysis'):
				pass
			logging.info(f'Directory structure for TI-LGAD analysis was created in {TILGAD_bureaucrat.pseudopath} ')
		except RuntimeError as e: # This will happen if the run already existed beforehand.
			if 'Cannot create run' in str(e):
				TILGAD_bureaucrat = [b for b in bureaucrat.list_subruns_of_task('TI-LGADs_analyses') if b.run_name==DUT_name][0]
		
	return TILGAD_bureaucrat

def read_parsed_from_waveforms_from_run(TB_run:RunBureaucrat, DUT_name:str, variables:list=['Amplitude (V)'], additional_SQL_selection:str=None, n_events:int=None):
	if not isinstance(variables, list):
		raise TypeError(f'`variables` must be a list of str, received object of type {type(variables)}. ')
	TB_run.check_these_tasks_were_run_successfully(['raw','parse_waveforms'])
	
	setup_config = utils.load_setup_configuration_info(TB_run.parent)
	
	if DUT_name not in set(setup_config["DUT_name"]):
		raise ValueError(f'DUT_name "{DUT_name}" not present in {TB_run.pseudopath}, available DUTs in this dataset are {set(setup_config["DUT_name"])}.')
	
	SQL_where_this_DUT = ' OR '.join([f'(n_CAEN=={_["n_CAEN"]} AND CAEN_n_channel=={_["CAEN_n_channel"]})' for idx,_ in setup_config.query(f'DUT_name=="{DUT_name}"').iterrows()])
	if len(variables) != 0:
		variables = ',' + ','.join([f'`{_}`' for _ in variables])
	else:
		variables = ''
	SQL_query = f'SELECT n_event,n_CAEN,CAEN_n_channel{variables} FROM dataframe_table WHERE ({SQL_where_this_DUT})'
	if additional_SQL_selection is not None:
		SQL_query += f' AND ({additional_SQL_selection})'
	if isinstance(n_events, int):
		SQL_query += f' AND n_event <= {n_events}'
	
	data = pandas.read_sql(
		SQL_query,
		con = sqlite3.connect(TB_run.path_to_directory_of_task('parse_waveforms')/f'{TB_run.run_name}.sqlite'),
	)
	data.set_index(['n_event','n_CAEN','CAEN_n_channel'], inplace=True)
	return data

def read_parsed_from_waveforms_from_batch(batch:RunBureaucrat, DUT_name:str, variables:list=['Amplitude (V)'], additional_SQL_selection:str=None, n_events:int=None):
	batch.check_these_tasks_were_run_successfully('runs')
	
	runs = batch.list_subruns_of_task('runs')
	data = []
	for run in runs:
		df = read_parsed_from_waveforms_from_run(
			TB_run = run,
			DUT_name = DUT_name,
			variables = variables,
			additional_SQL_selection = additional_SQL_selection,
			n_events = int(n_events/len(runs)) if isinstance(n_events, int) else None,
		)
		df = pandas.concat({int(run.run_name.split('_')[0].replace('run','')): df}, names=['n_run'])
		data.append(df)
	return pandas.concat(data)

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

def plot_tracks_and_hits(bureaucrat:RunBureaucrat, do_3D_plot:bool=True):
	bureaucrat.check_these_tasks_were_run_successfully(['TI_LGAD_analysis_setup','corry_reconstruct_tracks_with_telescope','parse_waveforms'])
	
	with bureaucrat.handle_task('plot_tracks_and_hits') as employee:
		analysis_config = load_analysis_config(bureaucrat)
		
		logging.info('Reading tracks data...')
		tracks = load_tracks_for_events_with_track_multiplicity_1(bureaucrat)
		
		logging.info('Reading DUT hits...')
		DUT_hits = read_parsed_from_waveforms(
			bureaucrat = bureaucrat,
			DUT_name = get_DUT_name(bureaucrat),
			variables = [],
			additional_SQL_selection = analysis_config['DUT_hit_selection_criterion_SQL_query'],
		)
		
		setup_config = utils.load_setup_configuration_info(bureaucrat)
		
		DUT_hits = DUT_hits.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])['DUT_name_rowcol'])
		
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
		
		tracks = tracks.join(DUT_hits['DUT_name_rowcol'])
		
		tracks['DUT_name_rowcol'] = tracks['DUT_name_rowcol'].fillna('no hit')
		tracks = tracks.sort_values('DUT_name_rowcol', ascending=False)
		
		logging.info('Plotting tracks and hits on DUT...')
		fig = px.scatter(
			tracks.reset_index(),
			title = f'Tracks projected on the DUT<br><sup>{utils.which_test_beam_campaign(bureaucrat)}/{bureaucrat.path_to_run_directory.parts[-4]}/{bureaucrat.run_name}</sup>',
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
				title = f'Tracks<br><sup>{utils.which_test_beam_campaign(bureaucrat)}/{bureaucrat.path_to_run_directory.parts[-4]}/{bureaucrat.run_name}</sup>',
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

def transformation_for_centering_and_leveling(bureaucrat:RunBureaucrat, draw_square:bool=False):
	bureaucrat.check_these_tasks_were_run_successfully(['TI_LGAD_analysis_setup','corry_reconstruct_tracks_with_telescope','parse_waveforms'])
	
	with bureaucrat.handle_task('transformation_for_centering_and_leveling') as employee:
		analysis_config = load_analysis_config(bureaucrat)
		
		logging.info('Reading tracks data...')
		tracks = load_tracks_for_events_with_track_multiplicity_1(bureaucrat)
		
		logging.info('Reading DUT hits...')
		DUT_hits = read_parsed_from_waveforms(
			bureaucrat = bureaucrat,
			DUT_name = get_DUT_name(bureaucrat),
			variables = [],
			additional_SQL_selection = analysis_config['DUT_hit_selection_criterion_SQL_query'],
		)
		
		setup_config = utils.load_setup_configuration_info(bureaucrat)
		
		DUT_hits = DUT_hits.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])['DUT_name_rowcol'])
		
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
		tracks = projected
		
		tracks = tracks.join(DUT_hits['DUT_name_rowcol'])
		
		tracks['DUT_name_rowcol'] = tracks['DUT_name_rowcol'].fillna('no hit')
		tracks = tracks.sort_values('DUT_name_rowcol', ascending=False)
		
		logging.info('Applying transformation to tracks to center and align DUT...')
		tracks = translate_and_rotate_tracks(
			tracks = tracks,
			x_translation = analysis_config['transformation_for_centering_and_leveling']['x_translation'],
			y_translation = analysis_config['transformation_for_centering_and_leveling']['y_translation'],
			angle_rotation = analysis_config['transformation_for_centering_and_leveling']['rotation_around_z_deg']/180*numpy.pi,
		)
		
		logging.info('Plotting tracks and hits on DUT...')
		fig = px.scatter(
			tracks.reset_index(),
			title = f'Tracks projected on the DUT after transformation<br><sup>{utils.which_test_beam_campaign(bureaucrat)}/{bureaucrat.path_to_run_directory.parts[-4]}/{bureaucrat.run_name}</sup>',
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
		
		logging.info('Saving transformation parameters into a file...')
		with open(employee.path_to_directory_of_my_task/'transformation_parameters.json', 'w') as ofile:
			json.dump(analysis_config['transformation_for_centering_and_leveling'], ofile)

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

def efficiency_vs_distance_calculation(bureaucrat:RunBureaucrat):
	bureaucrat.check_these_tasks_were_run_successfully('efficiency_vs_distance_analysis_config')
	TILGAD_bureaucrat = RunBureaucrat(bureaucrat.path_to_run_directory.parent.parent.parent) # This should point to a TI-LGAD analysis directory, let's check it in the next line
	TILGAD_bureaucrat.check_these_tasks_were_run_successfully(['corry_reconstruct_tracks_with_telescope','batch_info','parse_waveforms'])
	batch_bureaucrat = RunBureaucrat(TILGAD_bureaucrat.path_to_run_directory.parent.parent.parent)
	batch_bureaucrat.check_these_tasks_were_run_successfully(['corry_reconstruct_tracks_with_telescope','raw_to_root'])
	
	with bureaucrat.handle_task('efficiency_vs_distance_calculation') as employee:
		with open(bureaucrat.path_to_directory_of_task('efficiency_vs_distance_analysis_config')/'analysis_config.json', 'r') as ifile:
			analysis_config = json.load(ifile)
		
		logging.info('Reading tracks data...')
		tracks = load_tracks_for_events_with_track_multiplicity_1(TILGAD_bureaucrat)
		
		logging.info('Reading DUT hits...')
		DUT_hits = read_parsed_from_waveforms(
			bureaucrat = TILGAD_bureaucrat,
			DUT_name = get_DUT_name(TILGAD_bureaucrat),
			variables = [],
			additional_SQL_selection = analysis_config['DUT_hit_selection_criterion_SQL_query'],
		)
		
		setup_config = utils.load_setup_configuration_info(batch_bureaucrat)
		
		DUT_hits = DUT_hits.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])[['DUT_name_rowcol','row','col']])
		
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
		tracks = projected
		
		tracks = tracks.join(DUT_hits['DUT_name_rowcol'])
		
		tracks['DUT_name_rowcol'] = tracks['DUT_name_rowcol'].fillna('no hit')
		tracks = tracks.sort_values('DUT_name_rowcol', ascending=False)
		
		logging.info('Applying transformation to tracks to center and align DUT...')
		tracks = translate_and_rotate_tracks(
			tracks = tracks,
			x_translation = analysis_config['transformation_for_centering_and_leveling']['x_translation'],
			y_translation = analysis_config['transformation_for_centering_and_leveling']['y_translation'],
			angle_rotation = analysis_config['transformation_for_centering_and_leveling']['rotation_around_z_deg']/180*numpy.pi,
		)
		
		xmin = analysis_config['ROI']['x_min']
		xmax = analysis_config['ROI']['x_max']
		ymin = analysis_config['ROI']['y_min']
		ymax = analysis_config['ROI']['y_max']
		tracks_for_efficiency_calculation = tracks.query(f'{xmin}<Px and Px<{xmax} and {ymin}<Py and Py<{ymax}')
		tracks_for_efficiency_calculation = tracks_for_efficiency_calculation.reset_index(['n_CAEN','CAEN_n_channel'], drop=True)
		
		logging.info('Plotting tracks and hits on DUT...')
		plots_subtitle = f'{utils.which_test_beam_campaign(bureaucrat)}/{batch_bureaucrat.run_name}/{TILGAD_bureaucrat.run_name}/{bureaucrat.run_name}'
		fig = px.scatter(
			tracks_for_efficiency_calculation.reset_index(),
			title = f'Tracks projected on the DUT after transformation<br><sup>{plots_subtitle}</sup>',
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
			employee.path_to_directory_of_my_task/'tracks_projected_on_DUT.html',
			include_plotlyjs = 'cdn',
		)
		
		logging.info('Calculating efficiency vs distance...')
		
		distance_axis = numpy.arange(
			start = tracks_for_efficiency_calculation[f'P{analysis_config["project_on"]}'].min(),
			stop = tracks_for_efficiency_calculation[f'P{analysis_config["project_on"]}'].max(),
			step = analysis_config['calculation_step'],
		)
		efficiency_data = []
		for leftright in ['left','right','both']:
			if leftright in {'left','right'}:
				pixel_rowcol = analysis_config[f'{leftright}_pixel_rowcol']
				DUT_hits_for_efficiency = DUT_hits.query(f'row=={pixel_rowcol[0]} and col=={pixel_rowcol[1]}')
			elif leftright == 'both':
				pixel_rowcol = [analysis_config[f'{_}_pixel_rowcol'] for _ in {'left','right'}]
				DUT_hits_for_efficiency = DUT_hits.query(f'(row=={pixel_rowcol[0][0]} and col=={pixel_rowcol[0][1]}) or (row=={pixel_rowcol[1][0]} and col=={pixel_rowcol[1][1]})')
			else:
				raise RuntimeError('Check this, should never happen!')
			
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				efficiency_calculation_args = dict(
					tracks = tracks_for_efficiency_calculation.rename(columns={'Px':'x', 'Py':'y'})[['x','y']],
					DUT_hits = DUT_hits_for_efficiency.reset_index(['n_CAEN','CAEN_n_channel'], drop=True).index,
					project_on = analysis_config['project_on'],
					distances = distance_axis,
					window_size = analysis_config['rolling_window_size'],
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
			location = employee.path_to_directory_of_my_task,
		)
		
		logging.info('Plotting efficiency vs distance...')
		fig = plotly_utils.line(
			data_frame = efficiency_data.sort_index().reset_index(drop=False),
			title = f'Efficiency vs distance<br><sup>{plots_subtitle}</sup>',
			x = 'Distance (m)',
			y = 'Efficiency',
			error_y = 'Efficiency error_+',
			error_y_minus = 'Efficiency error_-',
			color = 'Pixel',
			error_y_mode = 'bands',
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'efficiency_vs_distance.html',
			include_plotlyjs = 'cdn',
		)

def setup_efficiency_vs_distance_analysis(bureaucrat, amplitude_threshold:float, DUT_z_position:float, x_translation:float, y_translation:float, rotation_around_z_deg:float, analyze_these_pixels:str, window_size_meters:float, window_step_meters:float, if_exists='raise error')->RunBureaucrat:
	PIXEL_SIZE = 250e-6
	ROI_DISTANCE_OFFSET = 50e-6
	ROI_WIDTH = PIXEL_SIZE/3
	PIXEL_DEPENDENT_SETTINGS = {
		'top row': dict(
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
		'bottom row': dict(
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
		'left column': dict(
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
		'right column': dict(
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
	
	if analyze_these_pixels not in PIXEL_DEPENDENT_SETTINGS.keys():
		raise ValueError(f'`analyze_these_pixels` must be in {set(PIXEL_DEPENDENT_SETTINGS.keys())}, but received {repr(analyze_these_pixels)}')
	
	bureaucrat.check_these_tasks_were_run_successfully('TI_LGAD_analysis_setup')
	
	analysis_config = {
		"DUT_hit_selection_criterion_SQL_query": f"100e-9<`t_50 (s)` AND `t_50 (s)`<150e-9 AND `Time over 50% (s)`>1e-9 AND `Amplitude (V)`<{amplitude_threshold}",
		"DUT_z_position": DUT_z_position,
		"transformation_for_centering_and_leveling": {
			"x_translation": x_translation,
			"y_translation": y_translation,
			"rotation_around_z_deg": rotation_around_z_deg,
		},
		"analysis_name": f"{analyze_these_pixels.replace(' ','_')}_{int(window_size_meters*1e6)}um_{int(window_step_meters*1e6)}um",
		"project_on": PIXEL_DEPENDENT_SETTINGS[analyze_these_pixels]['project_on'],
		"ROI": PIXEL_DEPENDENT_SETTINGS[analyze_these_pixels]['ROI'],
		"left_pixel_rowcol": PIXEL_DEPENDENT_SETTINGS[analyze_these_pixels]['left_pixel_rowcol'],
		"right_pixel_rowcol": PIXEL_DEPENDENT_SETTINGS[analyze_these_pixels]['right_pixel_rowcol'],
		"rolling_window_size": window_size_meters,
		"calculation_step": window_step_meters
	}
	
	with bureaucrat.handle_task('efficiency_vs_distance', drop_old_data=False) as employee:
		subrun = employee.create_subrun(analysis_config['analysis_name'], if_exists=if_exists)
		with subrun.handle_task('efficiency_vs_distance_analysis_config') as employee2:
			with open(employee2.path_to_directory_of_my_task/'analysis_config.json', 'w') as ofile:
				json.dump(analysis_config, ofile)
	return subrun

def setup_analyses_from_GoogleSpreadsheet(path_to_AIDAinnova_test_beams:Path):
	analyses = pandas.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vTaR20eM5ZQxtizmZiaAtHooE7hWYfSixSgc1HD5sVNZT_RNxZKmhI09wCEtXEVepjM8NB1n8BUBZnc/pub?gid=0&single=true&output=csv').set_index(['test_beam_campaign','batch_name','DUT_name']).query('DUT_type=="TI-LGAD"')
	
	PIXEL_GROUPS = {'top row','bottom row','left column','right column'}
	
	analyses = analyses.query(' or '.join([f'`{_}` == True' for _ in PIXEL_GROUPS]))
	
	tmp = []
	for pg in PIXEL_GROUPS:
		df = analyses.query(f'`{pg}` == True').drop(columns=sorted(PIXEL_GROUPS))
		df['pixel_group'] = pg
		tmp.append(df)
	analyses = pandas.concat(tmp).set_index('pixel_group', append=True).sort_index()
	
	for test_beam_campaign, test_beam_campaign_analyses in analyses.groupby('test_beam_campaign'):
		for batch_name, this_batch_analyses in test_beam_campaign_analyses.groupby('batch_name'):
			batch_bureaucrat = RunBureaucrat(path_to_AIDAinnova_test_beams/test_beam_campaign/'analysis'/batch_name)
			for DUT_name, this_DUT_analyses in this_batch_analyses.groupby('DUT_name'):
				try:
					setup_TI_LGAD_analysis(
						bureaucrat = batch_bureaucrat,
						DUT_name = DUT_name,
					)
					logging.info(f'Analysis for {repr(DUT_name)} was created')
				except RuntimeError as e:
					if all([_ in repr(e) for _ in {'Cannot create run','because it already exists'}]):
						logging.info(f'Analysis for {repr(DUT_name)} already exists')
					else:
						raise e
				
				DUT_bureaucrat = [_ for _ in batch_bureaucrat.list_subruns_of_task('TI-LGADs_analyses') if _.run_name==DUT_name][0]
				for pixel_group, analysis_settings in this_DUT_analyses.groupby('pixel_group'):
					analysis_bureaucrat = setup_efficiency_vs_distance_analysis(
						bureaucrat = DUT_bureaucrat,
						amplitude_threshold = analysis_settings['Amplitude threshold (V)'].values[0],
						DUT_z_position = analysis_settings['DUT_z_position'].values[0],
						x_translation = analysis_settings['x_translation'].values[0],
						y_translation = analysis_settings['y_translation'].values[0],
						rotation_around_z_deg = analysis_settings['rotation_around_z_deg'].values[0],
						analyze_these_pixels = pixel_group,
						window_size_meters = 22e-6, 
						window_step_meters = 5e-6,
						if_exists = 'override',
					)
					logging.info(f'Analysis for "{test_beam_campaign}/{batch_name}/{DUT_name}/{pixel_group}" has been created in {analysis_bureaucrat.path_to_run_directory}')

def execute_all_analyses(path_to_AIDAinnova_test_beams:Path):
	for test_beam_campaign in {'230614_June','230830_August'}:
		for p in (path_to_AIDAinnova_test_beams/test_beam_campaign/'analysis').iterdir():
			batch_bureaucrat = RunBureaucrat(p)
			for TILGAD_bureaucrat in batch_bureaucrat.list_subruns_of_task('TI-LGADs_analyses'):
				for analysis_bureaucrat in TILGAD_bureaucrat.list_subruns_of_task('efficiency_vs_distance'):
					current_analysis_pseudopath = f'{test_beam_campaign}/{batch_bureaucrat.run_name}/{TILGAD_bureaucrat.run_name}/{analysis_bureaucrat.run_name}'
					logging.info(f'Starting with {current_analysis_pseudopath}...')
					efficiency_vs_distance_calculation(analysis_bureaucrat)
					logging.info(f'Finished {current_analysis_pseudopath} ✅')

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
	args = parser.parse_args()
	
	bureaucrat = RunBureaucrat(Path(args.directory))
	
	# ~ setup_analyses_from_GoogleSpreadsheet(
		# ~ path_to_AIDAinnova_test_beams = Path('/media/msenger/230829_gray/AIDAinnova_test_beams'),
	# ~ )
	# ~ execute_all_analyses(
		# ~ path_to_AIDAinnova_test_beams = Path('/media/msenger/230829_gray/AIDAinnova_test_beams'),
	# ~ )
	
	# ~ a
	
	if bureaucrat.was_task_run_successfully('this_is_a_TI-LGAD_analysis'):
		if args.plot_DUT_distributions == True:
			plot_DUT_distributions(bureaucrat)
		if args.plot_tracks_and_hits == True:
			plot_tracks_and_hits(bureaucrat, do_3D_plot=args.enable_3D_tracks_plot)
		if args.transformation_for_centering_and_leveling == True:
			transformation_for_centering_and_leveling(bureaucrat, draw_square=True)
		if args.efficiency_vs_distance_calculation == True:
			efficiency_vs_distance_calculation(bureaucrat)
	elif args.setup_analysis_for_DUT != 'None':
		setup_TI_LGAD_analysis_within_batch(
			bureaucrat, 
			DUT_name = args.setup_analysis_for_DUT,
		)
	else:
		raise RuntimeError(f"Don't know what to do in {bureaucrat.path_to_run_directory}... Please read script help or source code.")
