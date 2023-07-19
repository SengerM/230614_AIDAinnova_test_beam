import uproot # https://github.com/scikit-hep/uproot5
from pathlib import Path
from signals.PeakSignal import PeakSignal, draw_in_plotly # https://github.com/SengerM/signals
import numpy
import pandas
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper # https://github.com/SengerM/huge_dataframe
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat

def parse_waveform(signal:PeakSignal):
	parsed = {
		'Amplitude (V)': signal.amplitude,
		'Noise (V)': signal.noise,
		'Rise time (s)': signal.rise_time,
		'Collected charge (V s)': signal.peak_integral,
		'Time over noise (s)': signal.time_over_noise,
		'Peak start time (s)': signal.peak_start_time,
		'Whole signal integral (V s)': signal.integral_from_baseline,
		'SNR': signal.SNR
	}
	for threshold_percentage in [10,20,30,40,50,60,70,80,90]:
		try:
			time_over_threshold = signal.find_time_over_threshold(threshold_percentage)
		except Exception:
			time_over_threshold = float('NaN')
		parsed[f'Time over {threshold_percentage}% (s)'] = time_over_threshold
	for pp in [10,20,30,40,50,60,70,80,90]:
		try:
			time_at_this_pp = float(signal.find_time_at_rising_edge(pp))
		except Exception:
			time_at_this_pp = float('NaN')
		parsed[f't_{pp} (s)'] = time_at_this_pp
	return parsed

def parse_waveforms_from_root_file_and_create_sqlite_database(root_file_path:Path, sqlite_database_path:Path):
	ifile = uproot.open(root_file_path)
	
	metadata = ifile['Metadata;1']
	sampling_frequency = metadata['sampling_frequency_MHz'].array()[0]*1e6
	samples_per_waveform = metadata['samples_per_waveform'].array()[0]
	time_array = numpy.linspace(0,(samples_per_waveform-1)/sampling_frequency,samples_per_waveform)
	
	waveforms = ifile['Waveforms;1']
	iterators = [waveforms[_].iterate(step_size=1, library='np') for _ in ['event','producer','voltages']]
	CAENs_names = []
	with SQLiteDataFrameDumper(sqlite_database_path, dump_after_n_appends = 11111, dump_after_seconds = 60, delete_database_if_already_exists=True) as parsed_data_dumper: 
		for n_event,producer,voltages in zip(*iterators):
			n_event = n_event['event'][0]
			CAEN_name = producer['producer'][0]
			voltages = voltages['voltages'][0]
			
			if CAEN_name not in CAENs_names:
				CAENs_names.append(CAEN_name)
			n_CAEN = CAENs_names.index(CAEN_name)
			
			this_event_parsed_data = []
			for n_channel, waveform_samples in enumerate(voltages):
				parsed = parse_waveform(
					PeakSignal(
						samples = -1*numpy.array(waveform_samples), # Multiply by -1 to make them positive.
						time = time_array,
					)
				)
				parsed['n_channel'] = n_channel
				parsed['n_CAEN'] = n_CAEN
				parsed['n_event'] = n_event
				this_event_parsed_data.append(parsed)
			this_event_parsed_data = pandas.DataFrame.from_records(this_event_parsed_data)
			this_event_parsed_data.set_index(['n_event','n_CAEN','n_channel'], inplace=True)
			parsed_data_dumper.append(this_event_parsed_data)

def parse_waveforms(bureaucrat:RunBureaucrat, force:bool=False):
	bureaucrat.check_these_tasks_were_run_successfully('raw_to_root')
	
	if force==False and bureaucrat.was_task_run_successfully('parse_waveforms'):
		return
	
	with bureaucrat.handle_task('parse_waveforms') as employee:
		path_to_directory_in_which_to_save_sqlite_databases = employee.path_to_directory_of_my_task/'parsed_data'
		path_to_directory_in_which_to_save_sqlite_databases.mkdir()
		for path_to_root_file in (employee.path_to_directory_of_task('raw_to_root')/'root_files').iterdir():
			parse_waveforms_from_root_file_and_create_sqlite_database(
				root_file_path = path_to_root_file,
				sqlite_database_path = path_to_directory_in_which_to_save_sqlite_databases/path_to_root_file.name.replace('.root','.sqlite')
			)

if __name__=='__main__':
	bureaucrat = RunBureaucrat(Path('/media/msenger/230302_red/June_test_beam/analysis/testing_deleteme'))
	parse_waveforms(bureaucrat)
