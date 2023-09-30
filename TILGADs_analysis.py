from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import utils
import logging
import pandas
import sqlite3
import plotly.express as px
import numpy
import plotly_utils
from corry_stuff import load_tracks_for_events_with_track_multiplicity_1
from parse_waveforms import read_parsed_from_waveforms
import json

def load_analysis_config(bureaucrat:RunBureaucrat):
	bureaucrat.check_these_tasks_were_run_successfully('TI_LGAD_analysis_setup')
	with open(bureaucrat.path_to_run_directory/'analysis_configuration.json', 'r') as ifile:
		return json.load(ifile)

def setup_TI_LGAD_analysis(bureaucrat:RunBureaucrat, DUT_name:str):
	"""Setup a directory structure to perform further analysis of a TI-LGAD
	that is inside a batch pointed by `bureaucrat`. This should be the 
	first step before starting a TI-LGAD analysis."""
	bureaucrat.check_these_tasks_were_run_successfully(['batch_info','corry_reconstruct_tracks_with_telescope','parse_waveforms'])
	
	with bureaucrat.handle_task('TI-LGADs_analyses', drop_old_data=False) as employee:
		setup_configuration_info = utils.load_setup_configuration_info(bureaucrat)
		if DUT_name not in set(setup_configuration_info['DUT_name']):
			raise RuntimeError(f'DUT_name {repr(DUT_name)} not present within the set of DUTs in {bureaucrat.run_name}, which is {set(setup_configuration_info["DUT_name"])}')
		
		logging.info(f'Creating directory for analysis of {DUT_name}...')
		TILGAD_bureaucrat = employee.create_subrun(DUT_name)
		for task_name in ['corry_reconstruct_tracks_with_telescope','parse_waveforms','batch_info']:
			(TILGAD_bureaucrat.path_to_run_directory/task_name).symlink_to(Path('../../../'+task_name))
		
		with TILGAD_bureaucrat.handle_task('TI_LGAD_analysis_setup'):
			pass
		
		logging.info(f'Directory for analysis of {DUT_name} created in "{TILGAD_bureaucrat.path_to_run_directory}"')

def get_DUT_name(bureaucrat:RunBureaucrat):
	"""Return the DUT name for the analysis pointed by `bureaucrat`."""
	bureaucrat.check_these_tasks_were_run_successfully('TI_LGAD_analysis_setup')
	return bureaucrat.run_name

def plot_DUT_distributions(bureaucrat:RunBureaucrat):
	"""Plot some raw distributions, useful to configure cuts and thresholds
	for further steps in the analysis."""
	MAXIMUM_NUMBER_OF_POINTS_TO_PLOT = 9999
	bureaucrat.check_these_tasks_were_run_successfully('TI_LGAD_analysis_setup')
	
	with bureaucrat.handle_task('plot_DUT_distributions') as employee:
		setup_config = utils.load_setup_configuration_info(bureaucrat)
		
		save_distributions_plots_here = employee.path_to_directory_of_my_task/'distributions'
		save_distributions_plots_here.mkdir()
		for variable in ['Amplitude (V)','t_50 (s)','Noise (V)','Time over 50% (s)',]:
			logging.info(f'Plotting {variable} distribution...')
			data = read_parsed_from_waveforms(
				bureaucrat = bureaucrat,
				DUT_name = get_DUT_name(bureaucrat),
				variables = [variable],
				limit = MAXIMUM_NUMBER_OF_POINTS_TO_PLOT,
			)
			data = data.sample(n=MAXIMUM_NUMBER_OF_POINTS_TO_PLOT) # To limit the number of entries plotted.
			data = data.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])['DUT_name_rowcol'])
			fig = px.ecdf(
				data.sort_values('DUT_name_rowcol'),
				title = f'{variable} distribution<br><sup>{utils.which_test_beam_campaign(bureaucrat)}/{bureaucrat.path_to_run_directory.parts[-4]}/{bureaucrat.run_name}</sup>',
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
			data = read_parsed_from_waveforms(
				bureaucrat = bureaucrat,
				DUT_name = get_DUT_name(bureaucrat),
				variables = [x,y],
				limit = MAXIMUM_NUMBER_OF_POINTS_TO_PLOT,
			)
			data = data.sample(n=MAXIMUM_NUMBER_OF_POINTS_TO_PLOT) # To limit the number of entries plotted.
			data = data.join(setup_config.set_index(['n_CAEN','CAEN_n_channel'])['DUT_name_rowcol'])
			fig = px.scatter(
				data.sort_values('DUT_name_rowcol'),
				title = f'{y} vs {x} scatter plot<br><sup>{utils.which_test_beam_campaign(bureaucrat)}/{bureaucrat.path_to_run_directory.parts[-4]}/{bureaucrat.run_name}</sup>',
				x = x,
				y = y,
				color = 'DUT_name_rowcol',
			)
			fig.write_html(
				save_scatter_plots_here/f'{y}_vs_{x}_scatter.html',
				include_plotlyjs = 'cdn',
			)

def plot_tracks_and_hits(bureaucrat:RunBureaucrat):
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

def preview_transformation_for_centering_and_leveling(bureaucrat:RunBureaucrat):
	bureaucrat.check_these_tasks_were_run_successfully(['TI_LGAD_analysis_setup','corry_reconstruct_tracks_with_telescope','parse_waveforms'])
	
	with bureaucrat.handle_task('preview_transformation_for_centering_and_leveling') as employee:
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
		fig.update_yaxes(
			scaleanchor = "x",
			scaleratio = 1,
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'tracks_projected_on_DUT.html',
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
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)
	args = parser.parse_args()
	
	bureaucrat = RunBureaucrat(Path(args.directory))
	preview_transformation_for_centering_and_leveling(bureaucrat)
