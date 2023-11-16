import uproot # https://github.com/scikit-hep/uproot5
from pathlib import Path
from signals.PeakSignal import PeakSignal, draw_in_plotly # https://github.com/SengerM/signals
import numpy
import pandas
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import plotly.graph_objects as go
import logging
import utils_batch_level
from parse_waveforms import plot_waveform
import dominate # https://github.com/Knio/dominate

def create_event_within_run(bureaucrat:RunBureaucrat, n_event:int, if_exists:str='override')->RunBureaucrat:
	bureaucrat.check_these_tasks_were_run_successfully('raw') # So we are sure this points to a "run node bureaucrat".
	
	if not isinstance(n_event, int):
		raise TypeError(f'`n_event` must be an integer number, received an object of type {type(n_event)}')
	
	with bureaucrat.handle_task('events', drop_old_data=False) as events_employee:
		event = events_employee.create_subrun(f'event_{n_event}', if_exists=if_exists)
		with event.handle_task('this_is_an_event'):
			pass
	return event

def plot_event_waveforms(bureaucrat:RunBureaucrat):
	bureaucrat.check_these_tasks_were_run_successfully('this_is_an_event')
	
	n_event = int(bureaucrat.run_name.replace('event_',''))
	
	bureaucrat_of_the_TB_run = bureaucrat.parent
	bureaucrat_of_the_TB_run.check_these_tasks_were_run_successfully('raw_to_root')
	
	root_file = uproot.open(bureaucrat_of_the_TB_run.path_to_directory_of_task('raw_to_root')/f'{bureaucrat_of_the_TB_run.run_name}.root')
	metadata = root_file['Metadata']
	waveforms = root_file['Waveforms']

	CAENs_channels_numbers = {}
	for n_CAEN,list_of_lists_of_channels in enumerate(metadata['channel'].array(library='np')):
		channels_numbers = [int(_) for listita in list_of_lists_of_channels for _ in listita]
		CAENs_channels_numbers[n_CAEN] = channels_numbers
	
	sampling_frequency = metadata['sampling_frequency_MHz'].array()[0]*1e6
	samples_per_waveform = metadata['samples_per_waveform'].array()[0]
	time_array = numpy.linspace(0,(samples_per_waveform-1)/sampling_frequency,samples_per_waveform)

	n_events = waveforms['event'].array(library='np')
	indices_where_to_look_for = numpy.where(n_events==n_event)[0]
	if len(indices_where_to_look_for) == 0:
		raise ValueError(f'n_event {n_event} not present in run number {n_run}. ')

	signals_connections = utils_batch_level.load_setup_configuration_info(bureaucrat.parent.parent)
	signals_connections.set_index(['CAEN_name','CAEN_n_channel'], inplace=True)

	with bureaucrat.handle_task('plot_waveforms') as employee:
		save_plots_here = employee.path_to_directory_of_my_task/'waveforms_plots'
		save_plots_here.mkdir()
		for n_CAEN,idx in enumerate(indices_where_to_look_for):
			CAEN_name = waveforms['producer'].array(entry_start=idx,entry_stop=idx+1, library='np')[0]
			CAEN_name = CAEN_name.replace('CAEN_','') # In some point in EUDAQ or in the raw to root conversion.
			
			for i,wf in enumerate(waveforms['voltages'].array(entry_start=idx,entry_stop=idx+1, library='np')[0]):
				CAEN_n_channel = CAENs_channels_numbers[n_CAEN][i]
				DUT_name = signals_connections.loc[(CAEN_name,CAEN_n_channel),'DUT_name']
				row = signals_connections.loc[(CAEN_name,CAEN_n_channel),'row']
				col = signals_connections.loc[(CAEN_name,CAEN_n_channel),'col']
				CAEN_channel_name = signals_connections.loc[(CAEN_name,CAEN_n_channel),'CAEN_channel_name']
				
				samples = numpy.array(wf)
				signal = PeakSignal(
					samples = samples,
					time = time_array,
					peak_polarity = 'guess',
				)
				fig = plot_waveform(signal, peak_start_time=False)
				fig.update_layout(
					title = f'DUT: {DUT_name} ({row},{col}), CAEN: {CAEN_name} {CAEN_channel_name}<br><sup>{bureaucrat.pseudopath}</sup>',
				)
				fig.update_traces(
					hoverinfo = 'skip',
				)
				fig.write_html(
					save_plots_here/f'{DUT_name}_{row}{col}_CAEN_{CAEN_name}.html',
					include_plotlyjs = 'cdn',
				)
			
			document_title = f'Waveforms for {bureaucrat.pseudopath}'
			html_doc = dominate.document(title=document_title)
			with html_doc:
				dominate.tags.h1(document_title)
				with dominate.tags.div(style='display: flex; flex-direction: column; width: 100%;'):
					for path_to_plot in sorted(save_plots_here.iterdir()):
						path_to_plot = path_to_plot.relative_to(path_to_plot.parent.parent)
						dominate.tags.iframe(src=str(path_to_plot), style=f'height: 88vh; min-height: 333px; width: 100%; min-width: 888px; border-style: solid;')
			with open(save_plots_here.parent/f'waveforms.html', 'w') as ofile:
				print(html_doc, file=ofile)
			logging.info(f'Waveforms were plotted for {bureaucrat.pseudopath} ✅')

def analyze_event_of_run(bureaucrat:RunBureaucrat, n_event:int):
	bureaucrat.check_these_tasks_were_run_successfully('raw') # So we are sure this points to a "run node bureaucrat".
	
	event_bureaucrat = create_event_within_run(bureaucrat, n_event, if_exists='override')
	plot_event_waveforms(event_bureaucrat)

if __name__=='__main__':
	import argparse
	from plotly_utils import set_my_template_as_default
	import sys
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.DEBUG,
		format = '%(asctime)s|%(levelname)s|%(funcName)s|%(message)s',
		datefmt = '%Y-%m-%d %H:%M:%S',
	)
	
	set_my_template_as_default()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to a run.',
		required = True,
		dest = 'directory',
		type = str,
	)
	parser.add_argument('--n_event',
		metavar = 'N',
		help = 'Number of event within the run.',
		required = True,
		dest = 'n_event',
		type = int,
	)
	
	args = parser.parse_args()
	bureaucrat = RunBureaucrat(Path(args.directory))
	analyze_event_of_run(
		bureaucrat,
		n_event = args.n_event,
	)
