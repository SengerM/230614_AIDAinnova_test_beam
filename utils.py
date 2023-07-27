import pandas
from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat

def save_dataframe(df, name:str, location:Path):
	for extension,method in {'pickle':df.to_pickle,'csv':df.to_csv}.items():
		method(location/f'{name}.{extension}')

def setup_batch_info(bureaucrat:RunBureaucrat):
	"""Add some batch-wise information needed for the analysis, like
	for example a link to the setup connection spreadsheet."""
	PATH_TO_SETUP_CONNECTIONS_FILES = Path.home()/'June_test_beam_data/AIDAInnova_June/setup_connections'
	
	with bureaucrat.handle_task('batch_info') as employee:
		with open(bureaucrat.path_to_run_directory/'n_batch', 'r') as ifile:
			n_batch = int(ifile.readline())
		path_to_setup_connection_ods = PATH_TO_SETUP_CONNECTIONS_FILES/f'Batch{n_batch}.ods'
		
		have_to_navigate_backwards = [_ for i,_ in enumerate(employee.path_to_directory_of_my_task.parts) if _!=path_to_setup_connection_ods.parts[i]]
		have_to_navigate_upwards_afterward = [_ for i,_ in enumerate(path_to_setup_connection_ods.parts) if _!=employee.path_to_directory_of_my_task.parts[i]]
		
		relative_path_to_setup_connection_ods = Path('/'.join(['..' for i in have_to_navigate_backwards]))/Path('/'.join(have_to_navigate_upwards_afterward))
		
		(employee.path_to_directory_of_my_task/'setup_connections.ods').symlink_to(relative_path_to_setup_connection_ods)

CAENs_CHANNELS_MAPPING_TO_INTEGERS = pandas.DataFrame(
	# This codification into integers comes from the producer, see in line 208 of `CAENDT5742Producer.py`. The reason is that EUDAQ can only handle integers tags, or something like this.
	{
		'CAEN_n_channel': list(range(18)),
		'CAEN_channel_name': [f'CH{i}' if i<16 else f'trigger_group_{i-16}' for i in range(18)]
	}
)

def load_setup_configuration_info(bureaucrat:RunBureaucrat)->pandas.DataFrame:
	bureaucrat.check_these_tasks_were_run_successfully(['batch_info','parse_waveforms'])
	
	planes = pandas.read_excel(bureaucrat.path_to_directory_of_task('batch_info')/'setup_connections.ods', sheet_name='planes', index_col='plane_number')
	signals_connections = pandas.read_excel(bureaucrat.path_to_directory_of_task('batch_info')/'setup_connections.ods', sheet_name='signals', index_col='plane_number')
	
	CAENs_names = []
	for sqlite_file_path in (bureaucrat.path_to_directory_of_task('parse_waveforms')/'parsed_data').iterdir():
		n_run = int(sqlite_file_path.name.split('_')[0].replace('run',''))
		df = pandas.read_pickle(bureaucrat.path_to_directory_of_task('parse_waveforms')/'CAENs_names'/sqlite_file_path.name.replace('.sqlite','_CAENs_names.pickle'))
		df = df.to_frame()
		df['n_run'] = n_run
		df.set_index('n_run',append=True,inplace=True)
		CAENs_names.append(df)
	CAENs_names = pandas.concat(CAENs_names)
	CAENs_names['CAEN_name'] = CAENs_names['CAEN_name'].apply(lambda x: x.replace('CAEN_','')) # In some point in EUDAQ or in the raw to root conversion, the CAENs names are prepended "CAEN_", which is annoying...
	
	# Here we assume that the CAENs were not changed within a batch, which is reasonable.
	_ = CAENs_names.reset_index('n_CAEN',drop=False).set_index('CAEN_name',append=True).reset_index('n_run',drop=True)
	_ = _[~_.index.duplicated(keep='first')]
	signals_connections = signals_connections.reset_index(drop=False).merge(
		_,
		on = 'CAEN_name',
	)
	
	signals_connections = signals_connections.merge(
		CAENs_CHANNELS_MAPPING_TO_INTEGERS.set_index('CAEN_channel_name')['CAEN_n_channel'],
		on = 'CAEN_channel_name',
	)
	
	signals_connections = signals_connections.merge(
		planes['DUT_name'],
		on = 'plane_number',
	)
	
	signals_connections['CAEN_trigger_group_n'] = signals_connections['CAEN_n_channel'].apply(lambda x: 0 if x in {0,1,2,3,4,5,6,7,16} else 1 if x in {8,9,10,11,12,13,14,15,17} else -1)
	
	signals_connections['rowcol'] = signals_connections[['row','col']].apply(lambda x: f'{x["row"]}{x["col"]}', axis=1)
	signals_connections['DUT_name_rowcol'] = signals_connections[['DUT_name','row','col']].apply(lambda x: f'{x["DUT_name"]} ({x["row"]},{x["col"]})', axis=1)
	
	return signals_connections

if __name__=='__main__':
	load_setup_configuration_info(RunBureaucrat(Path('/home/msenger/June_test_beam_data/analysis/batch_3')))
