import uproot # https://github.com/scikit-hep/uproot5
from pathlib import Path
from signals.PeakSignal import PeakSignal, draw_in_plotly # https://github.com/SengerM/signals
import numpy
import pandas
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper # https://github.com/SengerM/huge_dataframe
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import plotly.graph_objects as go
from raw_to_root import raw_to_root
import logging
import utils
import subprocess

def parse_waveform(signal:PeakSignal, vertical_unit:str, horizontal_unit:str):
	"""Parse a waveform and extract features like the amplitude, noise,
	rise time, etc."""
	parsed = {
		f'Amplitude ({vertical_unit})': signal.amplitude,
		f'Noise ({vertical_unit})': signal.noise,
		f'Rise time ({horizontal_unit})': signal.rise_time,
		f'Collected charge ({vertical_unit} {horizontal_unit})': signal.peak_integral,
		f'Time over noise ({horizontal_unit})': signal.time_over_noise,
		f'Peak start time ({horizontal_unit})': signal.peak_start_time,
		f'Whole signal integral ({vertical_unit} {horizontal_unit})': signal.integral_from_baseline,
		'SNR': signal.SNR
	}
	for threshold_percentage in [10,20,30,40,50,60,70,80,90]:
		try:
			time_over_threshold = signal.find_time_over_threshold(threshold_percentage)
		except Exception:
			time_over_threshold = float('NaN')
		parsed[f'Time over {threshold_percentage}% ({horizontal_unit})'] = time_over_threshold
	for pp in [10,20,30,40,50,60,70,80,90]:
		try:
			time_at_this_pp = float(signal.find_time_at_rising_edge(pp))
		except Exception:
			time_at_this_pp = float('NaN')
		parsed[f't_{pp} ({horizontal_unit})'] = time_at_this_pp
	return parsed

def plot_waveform(waveform:PeakSignal, peak_start_time:bool=True):
	fig = draw_in_plotly(waveform, peak_start_time=peak_start_time)
	fig.update_layout(
		xaxis_title = "Time (s)",
		yaxis_title = "Amplitude (V)",
	)
	MARKERS = { # https://plotly.com/python/marker-style/#custom-marker-symbols
		10: 'circle',
		20: 'square',
		30: 'diamond',
		40: 'cross',
		50: 'x',
		60: 'star',
		70: 'hexagram',
		80: 'star-triangle-up',
		90: 'star-triangle-down',
	}
	for pp in [10,20,30,40,50,60,70,80,90]:
		try:
			fig.add_trace(
				go.Scatter(
					x = [waveform.find_time_at_rising_edge(pp)],
					y = [waveform(waveform.find_time_at_rising_edge(pp))],
					mode = 'markers',
					name = f'Time at {pp} %',
					marker=dict(
						color = 'rgba(0,0,0,.5)',
						size = 11,
						symbol = MARKERS[pp]+'-open-dot',
						line = dict(
							color = 'rgba(0,0,0,.5)',
							width = 2,
						)
					),
				)
			)
		except Exception as e:
			pass
	return fig

def parse_waveforms_from_root_file_and_create_sqlite_database(root_file_path:Path, sqlite_database_path:Path, number_of_events_for_which_to_produce_control_plots:int=0):
	"""Parse the waveforms contained in a root file that was created
	with `caenCliRootWF`."""
	ifile = uproot.open(root_file_path)
	
	metadata = ifile['Metadata']
	waveforms = ifile['Waveforms']
	
	CAENs_channels_numbers = {}
	for n_CAEN,list_of_lists_of_channels in enumerate(metadata['channel'].array(library='np')):
		channels_numbers = [int(_) for listita in list_of_lists_of_channels for _ in listita]
		CAENs_channels_numbers[n_CAEN] = channels_numbers
	
	sampling_frequency = metadata['sampling_frequency_MHz'].array()[0]*1e6
	samples_per_waveform = metadata['samples_per_waveform'].array()[0]
	time_array = numpy.linspace(0,(samples_per_waveform-1)/sampling_frequency,samples_per_waveform)
	
	number_of_events_to_be_processed = int(len(waveforms['event'].array())/2)
	
	if number_of_events_for_which_to_produce_control_plots > 0:
		path_to_directory_in_which_to_save_the_control_plots = sqlite_database_path.with_suffix('.control_plots')
		path_to_directory_in_which_to_save_the_control_plots.mkdir()
	
	iterators = [waveforms[_].iterate(step_size=1, library='np') for _ in ['event','producer','voltages']]
	CAENs_names = []
	with SQLiteDataFrameDumper(sqlite_database_path, dump_after_n_appends = 11111, dump_after_seconds = 60, delete_database_if_already_exists=True) as parsed_data_dumper: 
		previous_n_event = None
		for n_event,producer,voltages in zip(*iterators):
			n_event = n_event['event'][0]
			CAEN_name = producer['producer'][0]
			voltages = voltages['voltages'][0]
			
			if CAEN_name not in CAENs_names:
				CAENs_names.append(CAEN_name)
			n_CAEN = CAENs_names.index(CAEN_name)
			
			if n_event != previous_n_event:
				previous_n_event = n_event
				produce_control_plots_for_this_event = numpy.random.randint(0,number_of_events_to_be_processed) < number_of_events_for_which_to_produce_control_plots
			
			this_event_parsed_data = []
			for i, waveform_samples in enumerate(voltages):
				CAEN_n_channel = CAENs_channels_numbers[n_CAEN][i]
				samples = numpy.array(waveform_samples)
				if CAEN_n_channel in {16,17}: # These are the trigger signals, which are square pulses. They require a bit of a special treatment.
					samples[-10:] = samples[:10].mean() # This is so the signal looks like a peak, i.e. a signal that rises and goes down, and my analysis framework for peak signals can handle also this step function.
				waveform = PeakSignal(
					samples = samples,
					time = time_array,
					peak_polarity = 'guess',
				)
				parsed = parse_waveform(
					waveform,
					vertical_unit = 'V', # Volts.
					horizontal_unit = 's', # Seconds.
				)
				parsed['CAEN_n_channel'] = CAEN_n_channel
				parsed['n_CAEN'] = n_CAEN
				parsed['n_event'] = n_event
				this_event_parsed_data.append(parsed)
				
				if produce_control_plots_for_this_event:
					fig = plot_waveform(waveform)
					fig.update_layout(
						title = f'n_event {n_event}, n_CAEN {n_CAEN} ({CAENs_names[n_CAEN]}), CAEN_n_channel {CAEN_n_channel}<br><sup>{bureaucrat.run_name}</sup>',
					)
					fig.write_html(
						path_to_directory_in_which_to_save_the_control_plots/f'n_event_{n_event}_n_CAEN_{n_CAEN}_CAEN_n_channel_{CAEN_n_channel}.html',
						include_plotlyjs = 'cdn',
					)
					
				
			this_event_parsed_data = pandas.DataFrame.from_records(this_event_parsed_data)
			this_event_parsed_data.set_index(['n_event','n_CAEN','CAEN_n_channel'], inplace=True)
			parsed_data_dumper.append(this_event_parsed_data)
			
			if n_event%99==0 and n_CAEN == len(CAENs_names)-1:
				logging.info(f'{n_event}/{number_of_events_to_be_processed} events processed. ({n_event/number_of_events_to_be_processed*100:.0f} %, {root_file_path.name})')
		
		CAENs_names = pandas.Series(CAENs_names)
		CAENs_names.index.set_names('n_CAEN',inplace=True)
		CAENs_names.name = 'CAEN_name'
		utils.save_dataframe(CAENs_names, name=sqlite_database_path.name.replace('.sqlite','_CAENs_names'), location=sqlite_database_path.parent)

def parse_waveforms(bureaucrat:RunBureaucrat, force:bool=False):
	"""Parse the waveforms from a run and store the data in SQLite
	databases.
	
	Arguments
	---------
	bureaucrat: RunBureaucrat
		A bureaucrat pointing to the run for which to parse the waveforms.
	force: bool, default False
		If `False` and the task `raw_to_root` was already executed successfully
		for the run being handled by `bureaucrat`, nothing is done.
	"""
	bureaucrat.check_these_tasks_were_run_successfully('raw_to_root')
	
	if force==False and bureaucrat.was_task_run_successfully('parse_waveforms'):
		return
	
	NUMBER_OF_EVENTS_FOR_WHICH_TO_PRODUCE_CONTROL_PLOTS = 5
	
	with bureaucrat.handle_task('parse_waveforms') as employee:
		path_to_directory_in_which_to_save_sqlite_databases = employee.path_to_directory_of_my_task/'parsed_data'
		path_to_directory_in_which_to_save_sqlite_databases.mkdir()
		path_to_CAENs_names_dir = employee.path_to_directory_of_my_task/'CAENs_names'
		path_to_CAENs_names_dir.mkdir()
		for path_to_root_file in (employee.path_to_directory_of_task('raw_to_root')/'root_files').iterdir():
			sqlite_database_path = path_to_directory_in_which_to_save_sqlite_databases/path_to_root_file.name.replace('.root','.sqlite')
			parse_waveforms_from_root_file_and_create_sqlite_database(
				root_file_path = path_to_root_file,
				sqlite_database_path = sqlite_database_path,
				number_of_events_for_which_to_produce_control_plots = NUMBER_OF_EVENTS_FOR_WHICH_TO_PRODUCE_CONTROL_PLOTS,
			)
			for fmt in ['csv','pickle']:
				subprocess.run(['mv', str(sqlite_database_path.parent)+f'/{sqlite_database_path.name.replace(".sqlite","")}_CAENs_names.{fmt}', str(path_to_CAENs_names_dir)])
			CAENs_names_path = sqlite_database_path.parent/(sqlite_database_path.name.replace('.sqlite','_CAENs_names'))
			if NUMBER_OF_EVENTS_FOR_WHICH_TO_PRODUCE_CONTROL_PLOTS > 0:
				sqlite_database_path.with_suffix('.control_plots').rename(sqlite_database_path.parent.parent/sqlite_database_path.with_suffix('.control_plots').name)

def parse_from_raw(bureaucrat:RunBureaucrat, force_raw_to_root:bool=False, force_root_to_sqlite:bool=False):
	utils.setup_batch_info(bureaucrat)
	raw_to_root(bureaucrat=bureaucrat, force=force_raw_to_root)
	parse_waveforms(bureaucrat=bureaucrat, force=force_root_to_sqlite)

if __name__=='__main__':
	import argparse
	import sys
	from grafica.plotly_utils.utils import set_my_template_as_default
	
	set_my_template_as_default()
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.DEBUG,
		format = '%(asctime)s|%(levelname)s|%(funcName)s|%(message)s',
		datefmt = '%Y-%m-%d %H:%M:%S',
	)
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)
	parser.add_argument(
		'--force_raw_to_root',
		help = 'If this flag is passed, it will force the processing even if it was already done beforehand. Old data will be deleted.',
		required = False,
		dest = 'force_raw_to_root',
		action = 'store_true'
	)
	parser.add_argument(
		'--force_root_to_sqlite',
		help = 'If this flag is passed, it will force the processing even if it was already done beforehand. Old data will be deleted.',
		required = False,
		dest = 'force_root_to_sqlite',
		action = 'store_true'
	)
	
	args = parser.parse_args()
	bureaucrat = RunBureaucrat(Path(args.directory))
	
	parse_from_raw(bureaucrat, force_raw_to_root=args.force_raw_to_root, force_root_to_sqlite=args.force_root_to_sqlite)
