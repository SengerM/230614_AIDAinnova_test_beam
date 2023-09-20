corry -c 01_mask_noisy_pixels.conf
corry -c 02_prealign-telescope.conf

corry -c 03_align-telescope.conf -o AlignmentTrackChi2.align_orientation=false
corry -c 03_align-telescope.conf -o detectors_file="corry_output_deleteme/batch_2_align-telescope.geo" -o Tracking4D.spatial_cut_abs=200um,200um -o AlignmentTrackChi2.max_track_chi2ndof=20
corry -c 03_align-telescope.conf -o detectors_file="corry_output_deleteme/batch_2_align-telescope.geo" -o Tracking4D.spatial_cut_abs=100um,100um -o AlignmentTrackChi2.max_track_chi2ndof=10 -o Tracking4D.min_hits_on_track=5

corry -c 04_align-telescope-mille.conf
corry -c 05_check-alignment-telescope.conf
