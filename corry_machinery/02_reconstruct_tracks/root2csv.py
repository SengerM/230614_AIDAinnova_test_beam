import ROOT

CSV_COLUMNS = 'n_event,n_track,is_fitted,chi2,ndof,Ax,Ay,Az,Bx,By,Bz'

ROOT.gInterpreter.AddIncludePath('/analysis/corryvreckan/src/objects')
ROOT.gSystem.AddDynamicPath('/analysis/corryvreckan/lib')

ROOT.gSystem.Load('libCorryvreckanObjects.so')

TrackClass = ROOT.corryvreckan.Track # StraightLineTrack

track = ROOT.std.vector(TrackClass)()

f = ROOT.TFile.Open('corry_output_deleteme/tracks.root')
tree = f.Get('Track')

with open('tracks.csv', 'w') as csv_file:
	print(CSV_COLUMNS, file=csv_file)
	for n_event,event in enumerate(tree):
		current_track_vector = getattr(event,'global')
		for n_track, track in enumerate(current_track_vector):
			A = track.getIntercept(99)
			B = track.getIntercept(-99)
			data = dict(
				n_event = n_event,
				n_track = n_track,
				is_fitted = track.isFitted(),
				chi2 = track.getChi2(),
				ndof = track.getNdof(),
				Ax = A.X(),
				Ay = A.Y(),
				Az = A.Z(),
				Bx = B.X(),
				By = B.Y(),
				Bz = B.Z(),
			)
			print(','.join([str(data[_]) for _ in CSV_COLUMNS.split(',')]), file=csv_file)
		if n_event % 999 == 0:
			print(f'{n_event} events have been processed')
