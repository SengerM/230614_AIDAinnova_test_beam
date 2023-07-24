import uproot # https://github.com/scikit-hep/uproot5
from pathlib import Path
from signals.PeakSignal import PeakSignal, draw_in_plotly # https://github.com/SengerM/signals
import numpy
import pandas
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import plotly.graph_objects as go
import logging
import utils
from parse_waveforms import plot_waveform
import dominate # https://github.com/Knio/dominate

def analyze_event(bureaucrat:RunBureaucrat, n_run:int, n_event:int):
	bureaucrat.check_these_tasks_were_run_successfully(['raw_to_root','batch_info'])
	
	path_to_root_file = [p for p in (bureaucrat.path_to_directory_of_task('raw_to_root')/'root_files').iterdir() if str(n_run) in p.name.split('_')[0]]
	if len(path_to_root_file) != 1:
		raise RuntimeError(f"Cannot locate root file for n_run {n_run} in {bureaucrat.path_to_directory_of_task('raw_to_root')/'root_files'}. ")
	path_to_root_file = path_to_root_file[0]
	
	ifile = uproot.open(path_to_root_file)
	
	metadata = ifile['Metadata']
	waveforms = ifile['Waveforms']
	
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
	
	planes = pandas.read_excel(bureaucrat.path_to_directory_of_task('batch_info')/'setup_connections.ods', sheet_name='planes', index_col='plane_number')
	signals_connections = pandas.read_excel(bureaucrat.path_to_directory_of_task('batch_info')/'setup_connections.ods', sheet_name='signals', index_col='plane_number')
	signals_connections = signals_connections.reset_index(drop=False).merge(
		utils.CAENs_CHANNELS_MAPPING_TO_INTEGERS.set_index('CAEN_channel_name')['CAEN_n_channel'],
		on = 'CAEN_channel_name',
	)
	
	signals_connections = signals_connections.merge(
		planes['DUT_name'],
		on = 'plane_number',
	)
	signals_connections.set_index(['CAEN_name','CAEN_n_channel'], inplace=True)
	signals_connections.sort_index(inplace=True)
	
	with bureaucrat.handle_task('analyze_specific_events', drop_old_data=False) as employee:
		save_plots_here = employee.path_to_directory_of_my_task/f'n_run_{n_run}_n_event_{n_event}/waveforms'
		save_plots_here.mkdir(parents=True, exist_ok=False)
		for n_CAEN,idx in enumerate(indices_where_to_look_for):
			CAEN_name = waveforms['producer'].array(entry_start=idx,entry_stop=idx+1, library='np')[0]
			CAEN_name = CAEN_name.replace('CAEN_','') # In some point in EUDAQ or in the raw to root conversion, the CAENs names are prepended "CAEN_", which is annoying...
			
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
					title = f'Run {n_run}, event {n_event}, DUT: {DUT_name} ({row},{col}), CAEN: {CAEN_name} {CAEN_channel_name}<br><sup>{bureaucrat.run_name}</sup>',
				)
				fig.update_traces(
					hoverinfo = 'skip',
				)
				fig.write_html(
					save_plots_here/f'{DUT_name}_{row}{col}_CAEN_{CAEN_name}.html',
					include_plotlyjs = 'cdn',
				)
			
			document_title = f'Waveforms for run {n_run}, event {n_event}'
			html_doc = dominate.document(title=document_title)
			with html_doc:
				dominate.tags.h1(document_title)
				with dominate.tags.div(style='display: flex; flex-direction: column; width: 100%;'):
					for path_to_plot in sorted(save_plots_here.iterdir()):
						path_to_plot = path_to_plot.relative_to(path_to_plot.parent.parent)
						dominate.tags.iframe(src=str(path_to_plot), style=f'height: 88vh; min-height: 333px; width: 100%; min-width: 888px; border-style: solid;')
			with open(save_plots_here.parent/f'waveforms.html', 'w') as ofile:
				print(html_doc, file=ofile)

if __name__=='__main__':
	import argparse
	from grafica.plotly_utils.utils import set_my_template_as_default
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
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)
	parser.add_argument('--n_run',
		metavar = 'N',
		help = 'Number of run in which to look for the event (i.e. the EUDAQ run number).',
		required = True,
		dest = 'n_run',
		type = int,
	)
	parser.add_argument('--n_event',
		metavar = 'N',
		help = 'Number of event within the EUDAQ run.',
		required = True,
		dest = 'n_event',
		type = int,
	)
	
	
	args = parser.parse_args()
	bureaucrat = RunBureaucrat(Path(args.directory))
	analyze_event(
		bureaucrat, 
		n_run = args.n_run,
		n_event = args.n_event,
	)
