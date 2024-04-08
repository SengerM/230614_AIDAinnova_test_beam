from datanodes import DatanodeHandler # https://github.com/SengerM/datanodes
import logging
import VoltagePoint
from pathlib import Path
import json
import numpy
import plotly.express as px
import pandas
from scipy.stats import binomtest
from utils import save_dataframe

def create_two_pixels_efficiency_analysis(voltage_point_dn:DatanodeHandler, analysis_name:str, left_pixel_chubut_channel_number:int, right_pixel_chubut_channel_number:int, x_center_of_the_pair_of_pixels:float, y_center_of_the_pair_of_pixels:float, rotation_angle_deg:float, y_acceptance_width:float):
	"""Create a new "two pixels efficiency analysis" inside a voltage point.
	
	Arguments
	---------
	voltage_point_dn: DatanodeHandler
		A `DatanodeHandler` pointing to a voltage point.
	"""
	with voltage_point_dn.handle_task('two_pixels_efficiency_analyses', check_datanode_class='voltage_point', keep_old_data=True) as task:
		analysis_dn = task.create_subdatanode(analysis_name, subdatanode_class='two_pixels_efficiency_analysis')
		
		with analysis_dn.handle_task('analysis_config') as analysis_config_task:
			with open(analysis_config_task.path_to_directory_of_my_task/'config.json', 'w') as ofile:
				json.dump(
					dict(
						left_pixel_chubut_channel_number = left_pixel_chubut_channel_number,
						right_pixel_chubut_channel_number = right_pixel_chubut_channel_number,
						x_center_of_the_pair_of_pixels = x_center_of_the_pair_of_pixels,
						y_center_of_the_pair_of_pixels = y_center_of_the_pair_of_pixels,
						rotation_angle_deg = rotation_angle_deg,
						y_acceptance_width = y_acceptance_width,
					),
					indent = '\t',
					fp = ofile,
				)
		
	logging.info(f'Analysis efficiency {repr(str(analysis_dn.pseudopath))} was created. ✅')

class DatanodeHandlerTwoPixelsEfficiencyAnalysis(DatanodeHandler):
	def __init__(self, path_to_datanode:Path):
		super().__init__(path_to_datanode=path_to_datanode, check_datanode_class='two_pixels_efficiency_analysis')
	
	@property
	def parent(self):
		return super().parent.as_type(VoltagePoint.DatanodeHandlerVoltagePoint)
	
	@property
	def analysis_config(self):
		if not hasattr(self, '_analysis_config'):
			with open(self.path_to_directory_of_task('analysis_config')/'config.json', 'r') as ifile:
				self._analysis_config = json.load(ifile)
		return self._analysis_config
	
	def load_hits_on_DUT(self, hit_amplitude_threshold:float):
		"""Load hits on DUT and indicates whether it is within the ROI
		for this analysis as well as whether it has a hit in left or right
		pixels according to the configuration of this analysis.
		
		Arguments
		---------
		hit_amplitude_threshold: float
			Value of amplitude to be considered as a "pixel active".
			
		Returns
		-------
		hits: pandas.DataFrame
			Example:
			```
										  x (m)     y (m)  within_ROI   left  right which_pixel
			EUDAQ_run n_event n_track                                                          
			226       43      0       -0.000342  0.001269       False  False  False        none
					  77      0       -0.000330 -0.002630       False  False  False        none
					  90      0       -0.000647  0.001220       False  False  False        none
					  126     0       -0.000037  0.000443       False  False  False        none
					  130     0       -0.000262  0.000049        True   True  False        left
			...                             ...       ...         ...    ...    ...         ...
			227       113473  2       -0.000616  0.000120       False  False  False        none
					  113476  0        0.000403  0.000915       False  False  False        none
					  113477  0        0.000447 -0.000425       False  False  False        none
					  113481  0       -0.000151  0.000594       False  False  False        none
					  113485  0        0.003883 -0.002768       False  False  False        none

			[23409 rows x 6 columns]
			```
		"""
		voltage_point_dn = self.parent
		hits = voltage_point_dn.load_hits_on_DUT()
		
		# Apply transformation to center the hits in xy:
		for xy in ['x', 'y']:
			hits[f'{xy} (m)'] -= self.analysis_config[f'{xy}_center_of_the_pair_of_pixels']
			r = (hits['x (m)']**2 + hits['y (m)']**2)**.5
			phi = numpy.arctan2(hits['y (m)'], hits['x (m)'])
			hits['x (m)'], hits['y (m)'] = r*numpy.cos(phi+self.analysis_config['rotation_angle_deg']*numpy.pi/180), r*numpy.sin(phi+self.analysis_config['rotation_angle_deg']*numpy.pi/180)
		
		# Create a column that tells whether the hit is inside the ROI for this analysis:
		hits['within_ROI'] = (hits['y (m)'] > -self.analysis_config['y_acceptance_width']) & (hits['y (m)'] < self.analysis_config['y_acceptance_width'])
		
		# Now indicate if left or right pixels had a hit:
		pixels_with_signal = self.parent.load_parsed_from_waveforms(
			where = f'`Amplitude (V)` < {-abs(hit_amplitude_threshold)} AND `Time over 50% (s)` > 1-9',
			variables = 'Amplitude (V)',
		)
		pixels_with_signal = pixels_with_signal.unstack(['n_CAEN','CAEN_n_channel']) # Unstack these so now we have one event per row.
		pixels_with_signal = pixels_with_signal['Amplitude (V)'] # Drop useless column level.
		pixels_with_signal = pixels_with_signal.notnull().astype(bool) # Convert amplitudes to boolean, indicating if there was a hit or not
		if len(set(pixels_with_signal.columns.get_level_values('n_CAEN'))) != 1:
			raise NotImplementedError('Had no time to implement this for more than 1 CAEN, sorry.')
		# Now it is just a stupid mapping from CAEN_n_channel to left or right, but it is super intricate:
		chubut_channels_numbers_to_leftright_mapping = {self.analysis_config[key]: key.split('_')[0] for key in {'left_pixel_chubut_channel_number', 'right_pixel_chubut_channel_number'}}
		batch_dn = self.parent.parent.parent
		batch_dn.check_datanode_class('TB_batch') # Not needed, but just in case...
		channels_mapping = batch_dn.setup_config.query(f'chubut_channel in {list(chubut_channels_numbers_to_leftright_mapping.keys())}')[['chubut_channel','CAEN_n_channel']]
		channels_mapping['leftright'] = [chubut_channels_numbers_to_leftright_mapping[chubut_channel] for chubut_channel in channels_mapping['chubut_channel']]
		pixels_with_signal.columns = [channels_mapping.loc[channels_mapping['CAEN_n_channel']==ch,'leftright'].values[0] for ch in pixels_with_signal.columns.get_level_values('CAEN_n_channel')]
		
		hits = hits.join(pixels_with_signal)
		hits[['left','right']] = hits[['left','right']].fillna(False) # Fill all the NaN values resulting from the join operation with `False`, i.e. not a hit.
		
		# Add an extra column tagging which pixel was hit, useful for plots:
		which_pixel = hits[['left','right']].apply(
			lambda x: 
				'none' if x['left']==x['right']==False 
				else 'both' if x['left']==x['right']==True
				else 'left' if x['left']==True and x['right'] == False
				else 'right',
			axis = 1,
		)
		which_pixel.name = 'which_pixel'
		hits = hits.join(which_pixel)
		
		return hits
	
	def plot_hits(self, hit_amplitude_threshold:float):
		with self.handle_task('plot_hits') as task:
			hits = self.load_hits_on_DUT(hit_amplitude_threshold=hit_amplitude_threshold)
			
			if len(hits) > 9999:
				hits = hits.sample(n=9999) # We don't want the plot to have soooo many points.
			fig = px.scatter(
				title = f'Hits used in efficiency vs distance calculation<br><sup>{self.pseudopath}, amplitude < {-abs(hit_amplitude_threshold)*1e3} mV</sup>',
				data_frame = hits.reset_index(),
				x = 'x (m)',
				y = 'y (m)',
				color = 'which_pixel',
				facet_col = 'within_ROI',
				hover_data = ['EUDAQ_run','n_event'],
				category_orders = {
					'which_pixel': ['none','left','right','both'],
				},
			)
			fig.update_yaxes(
				scaleanchor="x",
				scaleratio=1,
			)
			fig.write_html(
				task.path_to_directory_of_my_task/'hits.html',
				include_plotlyjs = 'cdn',
			)
		logging.info(f'Plotted the hits in {self.pseudopath} ✅')
	
	def binned_efficiency_vs_distance(self, hit_amplitude_threshold:float, bin_size_meters:float):
		CATEGORY_ORDERS = {
			'which_pixel': ['none','left','right','both'],
		}
		with self.handle_task('efficiency_vs_distance') as task:
			hits = self.load_hits_on_DUT(hit_amplitude_threshold=hit_amplitude_threshold)
			hits = hits.query('within_ROI == True') # For the efficiency analysis we only use the hits inside the ROI.
			
			fig = px.histogram(
				title = f'Hits histogram<br><sup>{self.pseudopath}, amplitude < {-abs(hit_amplitude_threshold)}</sup>',
				data_frame = hits,
				x = 'x (m)',
				color = 'which_pixel',
				category_orders = CATEGORY_ORDERS,
			)
			fig.write_html(
				task.path_to_directory_of_my_task/'hits_histogram.html',
				include_plotlyjs = 'cdn',
			)
			
			bins_edges = numpy.arange(
				start = hits['x (m)'].min(),
				stop = hits['x (m)'].max(),
				step = bin_size_meters,
			)
			
			counts = []
			efficiencies = []
			for n_bin, bin_left_edge in enumerate(bins_edges[:-1]):
				bin_right_edge = bins_edges[n_bin + 1]
				thisbinhits = hits.query(f'`x (m)` > {bin_left_edge} and `x (m)` < {bin_right_edge}')
				this_bin_counts = thisbinhits.groupby('which_pixel').count()
				this_bin_counts = this_bin_counts.iloc[:,0] # Keep only the first column, whichever it is, as they are all identical.
				for which_pixel in set(hits['which_pixel']):
					if which_pixel not in this_bin_counts:
						this_bin_counts[which_pixel] = 0 # Here I add missing values, which happens when the count is 0.
				# For computing the efficiency, "both" means that it was detected by left, right or by the two of them, so I have to sum them now:
				both_counted = this_bin_counts[['left','right','both']].sum()
				left_counted = this_bin_counts[['left','both']].sum()
				right_counted = this_bin_counts[['right','both']].sum()
				this_bin_counts['left'] = left_counted
				this_bin_counts['right'] = right_counted
				this_bin_counts['both'] = both_counted
				this_bin_counts['n_bin'] = n_bin
				counts.append(this_bin_counts)
				
				# Now compute the efficiency:
				for which_pixel in {'left','right','both'}:
					k_detected_hits = this_bin_counts[which_pixel]
					n_total_hits = this_bin_counts[['both','none']].sum() # Hits that either fired one pixel ('both') or did not fire any pixel ('none').
					if n_total_hits > 0: # If there is at least 1 track..
						efficiency_confidence_interval = binomtest(
							k = k_detected_hits, # Number of successes, i.e. number of detected hits.
							n = n_total_hits, # Number of trials, i.e. total hits that went through.
						).proportion_ci(confidence_level = .68)
						low = efficiency_confidence_interval.low
						high = efficiency_confidence_interval.high
						val = k_detected_hits/n_total_hits
					else: # If there is not even a single track in this bin...
						low = float('NaN')
						high = float('NaN')
						val = float('NaN')
					efficiencies.append(
						pandas.Series(
							{
								'low': low,
								'high': high,
								'val': val,
								'err_low': val - low,
								'err_high': high - val,
								'which_pixel': which_pixel,
								'n_bin': n_bin,
							}
						),
					)
			counts = pandas.DataFrame.from_records(counts).set_index('n_bin')
			counts = counts.stack('which_pixel')
			counts.name = 'count'
			efficiencies = pandas.DataFrame.from_records(efficiencies).set_index(['n_bin','which_pixel'])
			
			position = pandas.Series(
				(bins_edges[:-1] + bins_edges[1:])/2,
				name = 'Position (m)',
			)
			position.index.name = 'n_bin'
			
			fig = px.line(
				title = f'Counts used for efficiency vs distance calculation<br><sup>{self.pseudopath}, amplitude < {-abs(hit_amplitude_threshold)}</sup>',
				data_frame = counts.to_frame().join(position, on='n_bin').reset_index().sort_values('Position (m)'),
				color = 'which_pixel',
				y = 'count',
				x = 'Position (m)',
				line_shape = 'hvh',
				category_orders = CATEGORY_ORDERS,
			)
			fig.write_html(
				task.path_to_directory_of_my_task/'counts_vs_distance.html',
				include_plotlyjs = 'cdn',
			)
			
			efficiencies = efficiencies.join(position, on='n_bin')
			for col in {'val','err_low','err_high','low','high'}:
				efficiencies[f'{col} (%)'] = efficiencies[col]*100
			fig = px.line(
				title = f'Efficiency vs distance<br><sup>{self.pseudopath}, amplitude < {-abs(hit_amplitude_threshold)}</sup>',
				data_frame = efficiencies.reset_index().sort_values('Position (m)'),
				color = 'which_pixel',
				y = 'val (%)',
				x = 'Position (m)',
				error_y = 'err_high (%)',
				error_y_minus = 'err_low (%)',
				line_shape = 'hvh',
				labels = {
					'val (%)': 'Efficiency (%)',
				},
				category_orders = CATEGORY_ORDERS,
			)
			fig.write_html(
				task.path_to_directory_of_my_task/'efficiency_vs_distance.html',
				include_plotlyjs = 'cdn',
			)
			

			for name,df in {'bins':position,'counts':counts,'efficiency':efficiencies}.items():
				save_dataframe(df, name, task.path_to_directory_of_my_task)
			
			logging.info(f'Computed efficiency vs distance (and produced plots) in {self.pseudopath} ✅')
