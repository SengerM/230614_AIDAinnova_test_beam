from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import json
import pandas

def relevate_efficiency_analyses(TB:RunBureaucrat, analysis_type:str, parameters_to_copy:set):
	DUTs_TYPES = {
		'TI-LGAD': 'TI-LGADs_analyses',
		'RSD-LGAD': 'RSD-LGADs_analyses',
	}
	analyses_relevados = []
	for DUT_type in DUTs_TYPES:
		for campaign in TB.list_subruns_of_task('campaigns'):
			for batch in campaign.list_subruns_of_task('batches'):
				for DUT in batch.list_subruns_of_task(DUTs_TYPES[DUT_type]):
					for efficiency_analysis in DUT.list_subruns_of_task(analysis_type):
						data = {
							'campaign': campaign.run_name,
							'batch': batch.run_name,
							'DUT_name': DUT.run_name,
							'DUT_type': DUT_type,
							'analysis_type': analysis_type,
							'efficiency_analysis_name': efficiency_analysis.run_name,
						}
						with open(efficiency_analysis.path_to_run_directory/f'{analysis_type}.config.json', 'r') as ifile:
							analysis_config = json.load(ifile)
						
						for param in parameters_to_copy:
							data[param] = analysis_config.get(param)
						
						# Special handling of param 'trigger_on_DUTs':
						# ~ if 'trigger_on_DUTs' in analysis_config:
							# ~ data['trigger_on_DUTs'] = sorted([_ for _ in analysis_config['trigger_on_DUTs'].keys()])
						
						analyses_relevados.append(data)
	
	analyses_relevados = pandas.DataFrame.from_records(analyses_relevados)
	analyses_relevados.set_index(['campaign','batch','DUT_type','DUT_name','efficiency_analysis_name','analysis_type'], inplace=True)
	return analyses_relevados.sort_index()

def relevate_efficiency_vs_distance_left_right(TB:RunBureaucrat):
	return relevate_efficiency_analyses(
		TB = TB,
		analysis_type = 'efficiency_vs_distance_left_right',
		parameters_to_copy = ['pixel_size','ROI_distance_offset_from_pixel_border','ROI_width','calculation_step','bin_size','use_estimation_of_misreconstructed_tracks','DUT_hit_criterion','trigger_on_DUTs'],
	)

def relevate_efficiency_increasing_centered_ROI(TB:RunBureaucrat):
	return relevate_efficiency_analyses(
		TB = TB,
		analysis_type = 'efficiency_increasing_centered_ROI',
		parameters_to_copy = ['use_estimation_of_misreconstructed_tracks','ROI_size_min','ROI_size_max','ROI_size_n_steps','DUT_hit_criterion','trigger_on_DUTs'],
	)
	
if __name__ == '__main__':
	TB_bureaucrat = RunBureaucrat(Path('/media/msenger/230829_gray/AIDAinnova_test_beams/TB').resolve())
	
	with TB_bureaucrat.handle_task('relevar_analyses') as employee:
		for analysis_type, f in {'efficiency_vs_distance_left_right': relevate_efficiency_vs_distance_left_right, 'efficiency_increasing_centered_ROI': relevate_efficiency_increasing_centered_ROI}.items():
			analyses = f(TB_bureaucrat)
			analyses.to_csv(employee.path_to_directory_of_my_task/f'{analysis_type}.csv')
