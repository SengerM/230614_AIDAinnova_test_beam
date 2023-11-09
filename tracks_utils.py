from pathlib import Path
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import numpy
import pandas
import utils

def _project_track_in_z(A:numpy.array, B:numpy.array, z:float):
	"""Given two points in a (straight) track, A and B, finds the projection
	at some given z.
	
	Arguments:
	A, B: numpy.array
		Two points along a track. Can be a collection of points from multiple
		tracks, in this case the shape has to be (3,whatever...).
	z: float
		Value of z on which to project the tracks.
	"""
	def dot(a,b):
		return numpy.sum(a*b, axis=0)
	if A.shape != B.shape or A.shape[0] != 3:
		raise ValueError('Either `A` or `B`, or both, is invalid.')
	track_direction = (A-B)/(numpy.linalg.norm(A-B, axis=0))
	return A + track_direction*(z-A[2])/dot(track_direction, numpy.tile(numpy.array([0,0,1]), (int(A.size/3),1)).T)

def project_tracks(tracks:pandas.DataFrame, z:float):
	projected = _project_track_in_z(
		A = tracks[[f'A{_}' for _ in ['x','y','z']]].to_numpy().T,
		B = tracks[[f'B{_}' for _ in ['x','y','z']]].to_numpy().T,
		z = z,
	).T
	return pandas.DataFrame(
		projected,
		columns = ['Px','Py','Pz'],
		index = tracks.index,
	)

def tag_tracks_with_DUT_hits(tracks, DUT_hits):
	"""Combines tracks with DUT hits.
	
	Arguments
	---------
	tracks: pandas.DataFrame
		Data frame with the tracks, of the form 
		```
		                       is_fitted       chi2  ndof        Ax        Ay     Az        Bx        By    Bz  chi2/ndof        Px        Py     Pz
		n_run n_event n_track                                                                                                                       
		934   -1      0                1  10.008198     8 -0.002445  0.000963  0.099 -0.002429  0.000940 -99.0   1.251025 -0.002445  0.000963  0.353
			   0      0                1   5.321146     8 -0.001958 -0.000287  0.099 -0.001967 -0.000296 -99.0   0.665143 -0.001958 -0.000287  0.353
			   1      0                1  10.751408     8 -0.002930  0.000534  0.099 -0.002938  0.000530 -99.0   1.343926 -0.002930  0.000534  0.353
			   2      0                1   2.431047     8 -0.002877 -0.000961  0.099 -0.002857 -0.000962 -99.0   0.303881 -0.002877 -0.000961  0.353
			   3      0                1   3.703792     8 -0.002950  0.000373  0.099 -0.002948  0.000368 -99.0   0.462974 -0.002950  0.000373  0.353
		...                          ...        ...   ...       ...       ...    ...       ...       ...   ...        ...       ...       ...    ...
		937    1879   0                1   2.178779     8 -0.002431 -0.000973  0.099 -0.002432 -0.000997 -99.0   0.272347 -0.002431 -0.000973  0.353
			   1885   0                1   6.661766     8 -0.002225 -0.000288  0.099 -0.002234 -0.000315 -99.0   0.832721 -0.002225 -0.000288  0.353
			   1887   0                1   3.259248     8 -0.002294  0.000044  0.099 -0.002328  0.000048 -99.0   0.407406 -0.002294  0.000044  0.353
			   1888   0                1   2.518802     8 -0.002113 -0.000402  0.099 -0.002110 -0.000401 -99.0   0.314850 -0.002113 -0.000402  0.353
			   1889   0                1  15.561113     8 -0.002251 -0.000711  0.099 -0.002252 -0.000716 -99.0   1.945139 -0.002251 -0.000711  0.353

		[247513 rows x 13 columns]
		```
	DUT_hits: pandas.DataFrame
		A data frame with the DUT hits, of the form
		```
		DUT_name_rowcol  TI143 (0,0)  TI143 (0,1)  TI143 (1,0)  TI143 (1,1)
		n_run n_event                                                      
		932   18               False        False        False         True
			  83               False        False         True        False
			  174               True        False        False        False
			  187              False         True        False        False
			  222              False        False         True        False
		...                      ...          ...          ...          ...
		937   1414             False        False         True        False
			  1431             False        False         True        False
			  1653             False        False         True        False
			  1713             False        False         True        False
			  1785             False        False         True        False

		[13792 rows x 4 columns]
		```
	
	Returns
	-------
	tracks: pandas.DataFrame
		The tracks data frame with an extra column denoting which DUT was
		hit, and possibly extra rows if there were events in which more than
		one DUT was hit. Example:
		```
		                       is_fitted       chi2  ndof        Ax        Ay     Az        Bx        By    Bz  chi2/ndof        Px        Py     Pz DUT_name_rowcol
		n_run n_event n_track                                                                                                                                       
		932   -1      0              1.0   2.727382   8.0 -0.002202 -0.000413  0.099 -0.002195 -0.000417 -99.0   0.340923 -0.002202 -0.000413  0.353          no hit
			   0      0              1.0   3.769029   8.0 -0.001474 -0.001033  0.099 -0.001482 -0.001040 -99.0   0.471129 -0.001474 -0.001033  0.353          no hit
			   3      0              1.0   1.941348   8.0 -0.001764 -0.000605  0.099 -0.001771 -0.000613 -99.0   0.242668 -0.001764 -0.000605  0.353          no hit
			   4      0              1.0   8.075487   8.0 -0.001023  0.000958  0.099 -0.001022  0.000941 -99.0   1.009436 -0.001023  0.000958  0.353          no hit
			   6      0              1.0   4.384299   8.0 -0.001282 -0.000735  0.099 -0.001272 -0.000749 -99.0   0.548037 -0.001282 -0.000735  0.353          no hit
		...                          ...        ...   ...       ...       ...    ...       ...       ...   ...        ...       ...       ...    ...             ...
		937    1879   0              1.0   2.178779   8.0 -0.002431 -0.000973  0.099 -0.002432 -0.000997 -99.0   0.272347 -0.002431 -0.000973  0.353          no hit
			   1885   0              1.0   6.661766   8.0 -0.002225 -0.000288  0.099 -0.002234 -0.000315 -99.0   0.832721 -0.002225 -0.000288  0.353          no hit
			   1887   0              1.0   3.259248   8.0 -0.002294  0.000044  0.099 -0.002328  0.000048 -99.0   0.407406 -0.002294  0.000044  0.353          no hit
			   1888   0              1.0   2.518802   8.0 -0.002113 -0.000402  0.099 -0.002110 -0.000401 -99.0   0.314850 -0.002113 -0.000402  0.353          no hit
			   1889   0              1.0  15.561113   8.0 -0.002251 -0.000711  0.099 -0.002252 -0.000716 -99.0   1.945139 -0.002251 -0.000711  0.353          no hit

		[247575 rows x 14 columns]
		```
	"""
	
	DUT_hits = DUT_hits.copy()
	DUT_hits.columns = pandas.MultiIndex.from_product([['has_hit'], DUT_hits.columns])
	DUT_hits = DUT_hits.stack('DUT_name_rowcol')
	DUT_hits = DUT_hits.reset_index('DUT_name_rowcol', drop=False)
	DUT_hits = DUT_hits.query('has_hit==True') # Keep only rows where there is a hit.
	tracks = tracks.join(DUT_hits['DUT_name_rowcol'], how='outer')
	tracks['DUT_name_rowcol'] = tracks['DUT_name_rowcol'].fillna('no hit')
	tracks = tracks.dropna(how='any') # This will discard any event for which there was a hit in the DUT but no track reconstructed. Normally we don't care about them.
	return tracks
