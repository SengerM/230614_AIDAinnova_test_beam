from pathlib import Path
import numpy
import pandas
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import plotly.graph_objects as go
import plotly.express as px
import utils
import logging
from parse_waveforms import read_parsed_from_waveforms_from_batch
import dominate # https://github.com/Knio/dominate

def DUT_distributions_overview(batch:RunBureaucrat, max_events_to_plot:int=int(50e3)):
	batch.check_these_tasks_were_run_successfully(['runs','batch_info'])
	for run in batch.list_subruns_of_task('runs'):
		run.check_these_tasks_were_run_successfully(['raw','parse_waveforms'])
	
	with batch.handle_task('DUT_distributions_overview') as employee:
		setup_config = utils.load_setup_configuration_info(batch)
		
		save_1D_distributions_here = employee.path_to_directory_of_my_task/'distributions'
		save_1D_distributions_here.mkdir()
		
		VARIABLES_TO_PLOT_DISTRIBUTION = {'Amplitude (V)','t_50 (s)','Time over 50% (s)','Noise (V)','SNR'}
		for DUT_name,DUT_config in setup_config.groupby('DUT_name'):
			DUT_config = DUT_config.set_index(['n_CAEN','CAEN_n_channel'])
			save_plots_here = save_1D_distributions_here/DUT_name
			save_plots_here.mkdir(parents=True)
			for variable in VARIABLES_TO_PLOT_DISTRIBUTION:
				logging.info(f'Producing distribution plot for {variable} for DUT {DUT_name} in {batch.pseudopath}...')
				data = read_parsed_from_waveforms_from_batch(
					batch = batch,
					DUT_name = DUT_name,
					variables = [variable],
					n_events = max_events_to_plot,
				)
				data = data.merge(DUT_config[['DUT_name_rowcol']], left_index=True, right_index=True)
				
				fig = px.ecdf(
					data_frame = data.reset_index(drop=False).sort_values('DUT_name_rowcol'),
					x = variable,
					color = 'DUT_name_rowcol',
					title = f'{variable} for {DUT_name}<br><sup>{batch.pseudopath}</sup>',
					marginal = 'histogram',
				)
				fig.write_html(
					save_plots_here/f'{variable}.html',
					include_plotlyjs = 'cdn',
				)
		for variable in VARIABLES_TO_PLOT_DISTRIBUTION:
			document_title = f'{variable} distribution for all DUTs in {batch.pseudopath}'
			html_doc = dominate.document(title=document_title)
			with html_doc:
				dominate.tags.h1(document_title)
				for DUT_name in sorted(setup_config['DUT_name'].drop_duplicates()):
					dominate.tags.iframe(
						src = f'distributions/{DUT_name}/{variable}.html',
						style = f'height: 88vh; width: 100%; border-style: solid;',
					)
			with open(employee.path_to_directory_of_my_task/f'{variable}.html', 'w') as ofile:
				print(html_doc, file=ofile)
		
		
		PAIRS_OF_VARIABLES_FOR_SCATTER_PLOTS = {('t_50 (s)','Amplitude (V)'),('Time over 50% (s)','Amplitude (V)')}
		save_2D_scatter_plots_here = employee.path_to_directory_of_my_task/'scatter_plots'
		save_2D_scatter_plots_here.mkdir()
		for DUT_name,DUT_config in setup_config.groupby('DUT_name'):
			DUT_config = DUT_config.set_index(['n_CAEN','CAEN_n_channel'])
			save_plots_here = save_2D_scatter_plots_here/DUT_name
			save_plots_here.mkdir(parents=True)
			for x,y in PAIRS_OF_VARIABLES_FOR_SCATTER_PLOTS:
				logging.info(f'Producing 2D scatter plot for {y} vs {x} for DUT {DUT_name} in {batch.pseudopath}...')
				data = read_parsed_from_waveforms_from_batch(
					batch = batch,
					DUT_name = DUT_name,
					variables = [x,y],
					n_events = max_events_to_plot,
				)
				data = data.merge(DUT_config[['DUT_name_rowcol']], left_index=True, right_index=True)
				
				fig = px.scatter(
					data_frame = data.reset_index(drop=False).sort_values('DUT_name_rowcol'),
					x = x,
					y = y,
					color = 'DUT_name_rowcol',
					title = f'{y} vs {x} for {DUT_name}<br><sup>{batch.pseudopath}</sup>',
					hover_data = data.index.names,
				)
				fig.write_html(
					save_plots_here/f'{y} vs {x}.html',
					include_plotlyjs = 'cdn',
				)
		for x,y in PAIRS_OF_VARIABLES_FOR_SCATTER_PLOTS:
			document_title = f'{y} vs {x} scatter preview for all DUTs in {batch.pseudopath}'
			html_doc = dominate.document(title=document_title)
			with html_doc:
				dominate.tags.h1(document_title)
				for DUT_name in sorted(setup_config['DUT_name'].drop_duplicates()):
					dominate.tags.iframe(
						src = f'scatter_plots/{DUT_name}/{y} vs {x}.html',
						style = f'height: 88vh; width: 100%; border-style: solid;',
					)
			with open(employee.path_to_directory_of_my_task/f'{y} vs {x}.html', 'w') as ofile:
				print(html_doc, file=ofile)
		
if __name__=='__main__':
	import argparse
	from plotly_utils import set_my_template_as_default
	import sys
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.DEBUG,
		format = '%(asctime)s|%(levelname)s|%(message)s',
		datefmt = '%Y-%m-%d %H:%M:%S',
	)
	
	set_my_template_as_default()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to a batch.',
		required = True,
		dest = 'directory',
		type = str,
	)
	
	args = parser.parse_args()
	bureaucrat = RunBureaucrat(Path(args.directory))
	
	DUT_distributions_overview(bureaucrat, max_events_to_plot=1111)
