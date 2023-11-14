import sys
import argparse
from plotly_utils import set_my_template_as_default
from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import pandas
import logging
import plotly.express as px
import utils

def read_TI_LGADs_devices_sheet():
	logging.info('Reading TI-LGADs devices sheet from the cloud...')
	data = pandas.read_csv(
		'https://docs.google.com/spreadsheets/d/e/2PACX-1vQSI45HIdP6620mizZv3m9F5jsg67ny6cXJhD5_0Zh2yrVEzSoWAd6Eq9aqQtRAHCtCPr8jUvWbW86H/pub?gid=371872088&single=true&output=csv',
		dtype = {
			'wafer': int,
			'trenches': int,
			'neutrons (neq/cm^2)': float,
		}
	)
	data.rename(columns={'device_name':'DUT_name'}, inplace=True)
	data.set_index('DUT_name', inplace=True)
	return data
	
def collect_TI_LGAD_IPD(TB:RunBureaucrat, if_cant:str='skip'):
	IPD_data = []
	
	with TB.handle_task('collect_TI_LGAD_IPD') as employee:
		TB.check_these_tasks_were_run_successfully('campaigns')
		for campaign in TB.list_subruns_of_task('campaigns'):
				for batch in campaign.list_subruns_of_task('batches'):
					for TILGAD in batch.list_subruns_of_task('TI-LGADs_analyses'):
						if not TILGAD.was_task_run_successfully('interpixel_distance'):
							if if_cant == 'skip':
								continue
							else:
								raise RuntimeError(f'Cannot find `interpixel_distance` for {TILGAD.pseudopath}')
						_ = pandas.read_pickle(TILGAD.path_to_directory_of_task('interpixel_distance')/'IPD_final_value.pickle')
						_['DUT_name'] = TILGAD.run_name
						_['batch'] = batch.run_name
						_['campaign'] = campaign.run_name
						IPD_data.append(_)
		IPD_data = pandas.DataFrame.from_records(IPD_data)
		IPD_data.set_index(['campaign','batch','DUT_name'], inplace=True)
		IPD_data.sort_index(inplace=True)
		
		TI_LGADs_info = read_TI_LGADs_devices_sheet()
		
		IPD_data = IPD_data.join(TI_LGADs_info[['wafer','trench process','trench depth','trenches','pixel border','contact type']])
		
		print(IPD_data)
		
		fig = px.line(
			title = f'Inter-pixel distance in test beam',
			data_frame = IPD_data.reset_index(drop=False).sort_values(['campaign','batch','pixel border','contact type','trenches','trench depth']),
			x = 'batch',
			y = 'IPD (m)',
			error_y = 'IPD (m) error',
			color = 'contact type',
			facet_col = 'pixel border',
			facet_row = 'trenches',
			symbol = 'trench depth',
			line_group = 'DUT_name',
			hover_data = ['campaign','batch','DUT_name','wafer'],
			labels = {
			 'IPD (m)': 'Inter-pixel distance (m)',
			},
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'interpixel_distance.html',
			include_plotlyjs = 'cdn',
		)

def collect_TI_LGAD_effective_efficiency(TB:RunBureaucrat, if_cant:str='skip'):
	data = []
	
	PLOTS_LABELS = {
		'Fluence (neq/cm^-2×10^15)': 'Fluence (×10<sup>15</sup> n<sub>eq</sub> cm<sup>-2</sup>)',
		'effective_efficiency (%)': 'Effective efficiency (%)',
	}
	
	with TB.handle_task('collect_TI_LGAD_effective_efficiency') as employee:
		TB.check_these_tasks_were_run_successfully('campaigns')
		for campaign in TB.list_subruns_of_task('campaigns'):
			for batch in campaign.list_subruns_of_task('batches'):
				for TILGAD in batch.list_subruns_of_task('TI-LGADs_analyses'):
					for efficiency_2D_analysis in TILGAD.list_subruns_of_task('efficiency_2D'):
						if efficiency_2D_analysis.was_task_run_successfully('DUT_effective_efficiency'):
							_ = pandas.read_pickle(efficiency_2D_analysis.path_to_directory_of_task('DUT_effective_efficiency')/'effective_efficiency.pickle')
							_['DUT_name'] = TILGAD.run_name
							_['batch'] = batch.run_name
							_['campaign'] = campaign.run_name
							_['coincidence_with_another_DUT'] = 'NoCoinc' not in efficiency_2D_analysis.run_name
							_['analysis_name'] = efficiency_2D_analysis.run_name
							_['measurement'] = str(efficiency_2D_analysis.pseudopath)
							data.append(_)
		data = pandas.DataFrame.from_records(data)
		data.set_index(['campaign','batch','DUT_name','analysis_name'], inplace=True)
		data.sort_index(inplace=True)
		
		# For those devices that have two measurements (one with and one without a coincidence with another DUT), I manually select the one with a coincidence:
		KEEP_THESE = {
			'TB/230614_June/batch_2_230V/TI143/EfctvEffNoCoinc',
			'TB/230614_June/batch_2_230V/TI122/EfctvEffCoincTI116',
			'TB/230614_June/batch_2_230V/TI123/EfctvEffNoCoinc',
			'TB/230830_August/batch_3/TI145/EfctvEffCoincTI229',
			'TB/230830_August/batch_3/TI146/EfctvEffCoincTI229',
			'TB/230830_August/batch_3/TI229/EfctvEffCoincTI145',
			'TB/230830_August/batch_3/TI230/EfctvEffNoCoinc',
			'TB/230614_June/batch_2_230V/TI116/EfctvEffCoincTI122',
		}
		data = data.query(f'measurement in {list(KEEP_THESE)}')
		
		utils.save_dataframe(
			data,
			'effective_efficiency',
			employee.path_to_directory_of_my_task,
		)
		
		TI_LGADs_info = read_TI_LGADs_devices_sheet()
		
		data = data.join(TI_LGADs_info[['wafer','trench process','trench depth','trenches','pixel border','contact type','neutrons (neq/cm^2)']])
		
		for col in {'effective_efficiency','effective_efficiency_error'}:
			data[f'{col} (%)'] = data[col]*100
		
		data['Fluence (neq/cm^-2×10^15)'] = data['neutrons (neq/cm^2)']*1e-15
		
		print(data)
		print(sorted(list(data['measurement'])))
		print(sorted(data))
		
		fig = px.line(
			title = f'Effective efficiency in test beam',
			data_frame = data.reset_index(drop=False).sort_values(['campaign','batch','pixel border','contact type','trenches','trench depth']),
			facet_col = 'pixel border',
			y = 'effective_efficiency (%)',
			error_y = 'effective_efficiency_error (%)',
			color = 'contact type',
			facet_row = 'trenches',
			x = 'Fluence (neq/cm^-2×10^15)',
			symbol = 'contact type',
			line_group = 'DUT_name',
			hover_data = ['campaign','batch','DUT_name','wafer'],
			labels = PLOTS_LABELS | utils.PLOTS_LABELS,
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'effective_efficiency.html',
			include_plotlyjs = 'cdn',
		)

if __name__ == '__main__':
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.INFO,
		format = '%(asctime)s|%(levelname)s|%(message)s',
		datefmt = '%H:%M:%S',
	)
	
	set_my_template_as_default()
	
	collect_TI_LGAD_effective_efficiency(
		RunBureaucrat(Path('../TB')),
		# ~ if_cant = 'error',
	)
