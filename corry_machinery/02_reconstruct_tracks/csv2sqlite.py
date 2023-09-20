from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper # https://github.com/SengerM/huge_dataframe
import pandas
from pathlib import Path

dtype = dict(
	n_event = int,
	n_track = int,
	is_fitted = bool,
	chi2 = float,
	ndof = float,
	Ax = float,
	Ay = float,
	Az = float,
	Bx = float,
	By = float,
	Bz = float,
)

with SQLiteDataFrameDumper(Path('tracks.sqlite'), dump_after_n_appends=1e3) as tracks_dumper:
	for df in pandas.read_csv('tracks.csv', chunksize=1111, index_col=['n_event','n_track']):
		tracks_dumper.append(df)
