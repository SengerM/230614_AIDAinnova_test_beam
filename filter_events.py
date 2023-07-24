from pathlib import Path
import numpy
import pandas
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import logging
import utils
import sqlite3
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper

def filter_two_pixels_for_beta_scan_like_time_resolution(bureaucrat:RunBureaucrat, pixels:dict, n_event_batch_size:int=11111):
	"""Creates a new sub run that looks like a beta scan.
	
	Arguments
	---------
	bureaucrat: RunBureaucrat
		A bureaucrat pointing to the run in which to process this.
	pixels: dict
		A dictionary of the form
		```
		{'TI143':(0,0), 'time_reference_1':(1,0)}
		```
		i.e. with two DUTs names and one pixel per DUT.
	"""
	bureaucrat.check_these_tasks_were_run_successfully('parse_waveforms')
	
	if len(pixels) != 2 or any([len(_) != 2 for DUT_name,_ in pixels.items()]):
		raise ValueError(f'`pixels` is wrong, please check.')
	
	signals_connections = utils.load_setup_configuration_info(bureaucrat)
	we_are_interested_in = []
	for DUT_name,_ in pixels.items():
		row,col = _
		we_are_interested_in.append(signals_connections.query(f'DUT_name=={repr(DUT_name)} and row=={row} and col=={col}'))
	we_are_interested_in = pandas.concat(we_are_interested_in)
	we_are_interested_in.set_index(['DUT_name','row','col'], inplace=True)
	
	with bureaucrat.handle_task('beta_scan_like_subsets', drop_old_data=True) as employee:
		logging.info(f'About to process {bureaucrat.run_name}...')
		subrun = employee.create_subrun('_'.join([f'{DUT_name}_{pixels[DUT_name][0]}{pixels[DUT_name][1]}' for DUT_name in sorted(pixels)]))
		with subrun.handle_task('beta_scan') as subemployee:
			with SQLiteDataFrameDumper(subemployee.path_to_directory_of_my_task/'parsed_from_waveforms.sqlite', dump_after_n_appends=999) as data_dumper:
				current_lowest_n_waveform = 0
				absolute_n_trigger = 0
				for sqlite_file_path in (bureaucrat.path_to_directory_of_task('parse_waveforms')/'parsed_data').iterdir():
					n_run = int(sqlite_file_path.name.split('_')[0].replace('run',''))
					connection = sqlite3.connect(sqlite_file_path)
					n_event_low = 0
					while True:
						n_event_high = n_event_low+n_event_batch_size
						data = pandas.read_sql(f'SELECT * FROM dataframe_table WHERE n_event>={n_event_low} AND n_event<{n_event_high}', connection)
						data['n_run'] = n_run
						data.set_index(['n_run','n_event','n_CAEN','CAEN_n_channel'], inplace=True)
						if len(data)==0:
							break
						
						keep_this = []
						for idx,info in we_are_interested_in.iterrows():
							DUT_name,row,col = idx
							keep_this.append(
								data.query(f'n_CAEN=={info["n_CAEN"]} and CAEN_n_channel=={info["CAEN_n_channel"]}')
							)
						keep_this = pandas.concat(keep_this)
						keep_this = keep_this.reset_index(drop=False).merge(we_are_interested_in.reset_index(drop=False).set_index(['n_CAEN','CAEN_n_channel'])[['DUT_name','row','col']], on=['n_CAEN','CAEN_n_channel'])
						keep_this['signal_name'] = keep_this[['DUT_name','row','col']].apply(lambda x: f'{x["DUT_name"]}_{x["row"]}{x["col"]}', axis=1)
						
						this_batch_n_events = keep_this['n_event'].drop_duplicates().to_frame()
						this_batch_n_events['n_trigger'] = numpy.arange(start=absolute_n_trigger, stop=absolute_n_trigger+len(this_batch_n_events))
						this_batch_n_events.set_index('n_event', inplace=True)
						this_batch_n_events = this_batch_n_events['n_trigger']
						keep_this = keep_this.join(this_batch_n_events, on='n_event')
						
						keep_this.set_index(['n_trigger','signal_name'], inplace=True)
						keep_this['n_waveform'] = numpy.array([n_waveform for n_waveform in range(current_lowest_n_waveform,current_lowest_n_waveform+len(keep_this))])
						for col in ['Amplitude (V)','Collected charge (V s)','Whole signal integral (V s)']:
							keep_this[col] *= -1
						data_dumper.append(keep_this)
						logging.info(f'{n_event_low} events from file {sqlite_file_path.name}, total {absolute_n_trigger}')
						n_event_low = n_event_high
						current_lowest_n_waveform += len(keep_this)
						absolute_n_trigger += len(keep_this)
						

if __name__=='__main__':
	from grafica.plotly_utils.utils import set_my_template_as_default
	import sys
	
	set_my_template_as_default()
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.DEBUG,
		format = '%(asctime)s|%(levelname)s|%(funcName)s|%(message)s',
		datefmt = '%Y-%m-%d %H:%M:%S',
	)

	filter_two_pixels_for_beta_scan_like_time_resolution(
		bureaucrat = RunBureaucrat(Path('/home/msenger/June_test_beam_data/analysis/batch_2_230V')),
		pixels = {'TI115':(1,0), 'time_reference_1':(1,0)},
	)
