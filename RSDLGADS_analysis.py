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

def load_tracks(RSD_analysis:RunBureaucrat, DUT_z_position:float):
	RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	batch = RSD_analysis.parent
	
	tracks = load_tracks_from_batch(batch, only_multiplicity_one=True)
	tracks = tracks.join(project_tracks(tracks=tracks, z_position=DUT_z_position))
	
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

def plot_cluster_size(RSD_analysis:RunBureaucrat, force:bool=False):
	RSD_analysis.check_these_tasks_were_run_successfully('this_is_an_RSD-LGAD_analysis')
	TASK_NAME = 'plot_cluster_size'
	
	if force==False and RSD_analysis.was_task_run_successfully(TASK_NAME):
		return
	
	with RSD_analysis.handle_task(TASK_NAME) as employee:
		analysis_config = load_this_RSD_analysis_config(RSD_analysis)
		
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
		fig.write_html(
			employee.path_to_directory_of_my_task/'cluster_size.html',
			include_plotlyjs = 'cdn',
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
		'--plot_cluster_size',
		help = 'Pass this flag to run `plot_cluster_size`.',
		required = False,
		dest = 'plot_cluster_size',
		action = 'store_true'
	)
	args = parser.parse_args()
	
	_bureaucrat = RunBureaucrat(args.directory)
	if _bureaucrat.was_task_run_successfully('this_is_an_RSD-LGAD_analysis'):
		if args.plot_distributions == True:
			plot_distributions(_bureaucrat, force=args.force)
		if args.plot_tracks_and_hits == True:
			plot_tracks_and_hits(_bureaucrat, force=args.force)
		if args.plot_cluster_size == True:
			plot_cluster_size(_bureaucrat, force=args.force)
	elif _bureaucrat.was_task_run_successfully('batch_info') and args.setup_analysis_for_DUT is not None:
		setup_RSD_LGAD_analysis_within_batch(batch=_bureaucrat, DUT_name=args.setup_analysis_for_DUT)
	else:
		raise RuntimeError(f"Don't know what to do in {_bureaucrat.path_to_run_directory}... Please read script help or source code.")
