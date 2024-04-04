import uproot # https://github.com/scikit-hep/uproot5
from pathlib import Path
from signals.PeakSignal import PeakSignal, draw_in_plotly # https://github.com/SengerM/signals
import numpy
import pandas
from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import plotly.graph_objects as go
import logging
import utils_batch_level
from parse_waveforms import plot_waveform
import dominate # https://github.com/Knio/dominate

def create_event_dn(EUDAQ_run_dn:DatanodeHandler, n_event:int, if_exists:str='override')->DatanodeHandler:
	if not isinstance(n_event, int):
		raise TypeError(f'`n_event` must be an integer number, received an object of type {type(n_event)}')
	
	with EUDAQ_run_dn.handle_task('events', check_datanode_class='EUDAQ_run', keep_old_data=True) as events_task:
		event = events_task.create_subdatanode(
			f'event_{n_event}', 
			subdatanode_class = 'TB_event',
			if_exists = if_exists,
		)
	return event

def plot_waveforms(event_dn:DatanodeHandler):
	with event_dn.handle_task('plot_waveforms', 'TB_event') as plot_waveforms_task:
		n_event = int(event_dn.datanode_name.replace('event_',''))
		
		EUDAQ_run_dn = event_dn.parent
		EUDAQ_run_dn.check_datanode_class('EUDAQ_run')
		
		root_file = uproot.open(EUDAQ_run_dn.path_to_directory_of_task('raw_to_root')/f'{EUDAQ_run_dn.datanode_name}.root')
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

		signals_connections = utils_batch_level.load_setup_configuration_info(event_dn.parent.parent)
		signals_connections.set_index(['CAEN_name','CAEN_n_channel'], inplace=True)

	
		save_plots_here = plot_waveforms_task.path_to_directory_of_my_task/'waveforms_plots'
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
					title = f'DUT: {DUT_name} ({row},{col}), CAEN: {CAEN_name} {CAEN_channel_name}<br><sup>{event_dn.pseudopath}</sup>',
				)
				fig.update_traces(
					hoverinfo = 'skip',
				)
				fig.write_html(
					save_plots_here/f'{DUT_name}_{row}{col}_CAEN_{CAEN_name}.html',
					include_plotlyjs = 'cdn',
				)
			
			document_title = f'Waveforms for {event_dn.pseudopath}'
			html_doc = dominate.document(title=document_title)
			with html_doc:
				dominate.tags.h1(document_title)
				with dominate.tags.div(style='display: flex; flex-direction: column; width: 100%;'):
					for path_to_plot in sorted(save_plots_here.iterdir()):
						path_to_plot = path_to_plot.relative_to(path_to_plot.parent.parent)
						dominate.tags.iframe(src=str(path_to_plot), style=f'height: 88vh; min-height: 333px; width: 100%; min-width: 888px; border-style: solid;')
			with open(save_plots_here.parent/f'waveforms.html', 'w') as ofile:
				print(html_doc, file=ofile)
			logging.info(f'Waveforms were plotted for {event_dn.pseudopath} ✅')

def analyze_event(EUDAQ_run_dn:DatanodeHandler, n_event:int):
	EUDAQ_run_dn.check_datanode_class('EUDAQ_run')
	
	event_dn = create_event_dn(EUDAQ_run_dn, n_event, if_exists='override')
	plot_waveforms(event_dn)

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
	parser.add_argument('--datanode',
		metavar = 'path',
		help = 'Path to a `EUDAQ_run` event.',
		required = True,
		dest = 'datanode',
		type = Path,
	)
	parser.add_argument('--n_event',
		metavar = 'N',
		help = 'Number of event within the run.',
		required = True,
		dest = 'n_event',
		type = int,
	)
	args = parser.parse_args()
	
	analyze_event(
		EUDAQ_run_dn = DatanodeHandler(args.datanode),
		n_event = args.n_event,
	)
