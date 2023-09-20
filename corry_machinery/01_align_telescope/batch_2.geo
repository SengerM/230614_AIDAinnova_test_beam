# GEO-ID: Telescope and DUTs Setup 
# It includes the type and position/rotation of the different
# detectors in global coordinates, as well as their number of
# pixels and size. Also the role of each detector: reference,
# DUT or auxiliary.
#
# The headers identify the detector, they indicate the ID. It
# is in the range 1-9 for Mimosa26 sensors, 20-30 for FEI4 
# reference sensors, and 30-60 or numbers from 100 for the DUTs
# depending on the DAQ system.
# 
# The CAEN 
#
[TLU_0]
number_of_pixels = 0, 0
orientation_mode = "xyz"
pixel_pitch = 0mm, 0mm
position = 0, 0, 0mm
spatial_resolution = 0um, 0um
time_resolution = 1s
material_budget = 0.0
type="tlu"
role = "auxiliary"

[MIMOSA26_0]
material_budget = 0.00075
number_of_pixels = 1152, 576
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 18.4um, 18.4um
position = 0, 0, 0mm
spatial_resolution = 5.2um, 5.2um
time_resolution = 230us
type = "mimosa26"
roi = [380, 190], [380, 335], [530, 335], [530, 190]

[MIMOSA26_1]
material_budget = 0.00075
number_of_pixels = 1152, 576
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 18.4um, 18.4um
position = 0, 0, 96mm
spatial_resolution = 5.2um, 5.2um
time_resolution = 230us
type = "mimosa26"
roi = [380, 205], [380, 350], [520, 350], [520, 205]  

[MIMOSA26_2]
material_budget = 0.00075
number_of_pixels = 1152, 576
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 18.4um, 18.4um
position = 0, 0, 190mm
spatial_resolution = 5.2um, 5.2um
time_resolution = 230us
role = "reference"
type = "mimosa26"
roi = [393, 220], [393, 361], [534, 361], [534, 220]  


# The DUTs
# The extraction of the channel and pixel description is given by teh euCliReader 
# INFO printout
# For each motorized support, two DUTs are placed (see Batch setup in the spreadsheet:
# https://docs.google.com/spreadsheets/d/1Un2SCmU_lmghlmvQ-rXDjS7DsXeofMdb-jQFq5K1ZPk
#
# Each DUT was measured with two digitizers, one per c//

# DUT  1st motor : position = 0, 0, 351mm 
# DUT  2nd motor : position = 0, 0, 436mm
# DUT  3rd motor : position = 0, 0, 516mm --> 516 and 518
# DUT  4th motor : position = 0, 0, 615mm

# ======================================
# The DUT at the 1st motor -- Upstream 
# --> Left reference (for trigger, not included)???
# --> HPK V2 Split4
[CAEN_IJS_5]
material_budget = 0.00001 # ??
number_of_pixels = 2, 2
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 999um, 999um
position = 0, 0, 351mm
spatial_resolution = 288.7um, 288.7um
time_resolution = 25ns
role = "dut"
type = "caen5748"

[CAEN_UZH_5]
material_budget = 0.00001 # ??
number_of_pixels = 2, 2
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 999um, 999um
position = 0, 0, 351mm
spatial_resolution = 288.7um, 288.7um
time_resolution = 25ns
role = "dut"
type = "caen5748"

# ======================================
# The DUT at the 1st motor -- Downstream 
# --> TI143
[CAEN_IJS_4]
material_budget = 0.00001 # ??
number_of_pixels = 2, 2
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 250um, 250um
position = 0, 0, 353mm
spatial_resolution = 72.2um, 72.2um
time_resolution = 25ns
role = "dut"
type = "caen5748"

[CAEN_UZH_4]
material_budget = 0.00001 # ??
number_of_pixels = 2, 2
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 250um, 250um
position = 0, 0, 353mm
spatial_resolution = 72.2um, 72.2um
time_resolution = 25ns
role = "dut"
type = "caen5748"
# ======================================


# ======================================
# The DUT at the 2nd motor -- Upstream 
# --> TI123
[CAEN_IJS_3]
material_budget = 0.00001 # ??
number_of_pixels = 2, 2
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 250um, 250um
position = 0, 0, 436mm
spatial_resolution = 72.2um, 72.2um
time_resolution = 25ns
role = "dut"
type = "caen5748"

[CAEN_UZH_3]
material_budget = 0.00001 # ??
number_of_pixels = 2, 2
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 250um, 250um
position = 0, 0, 436mm
spatial_resolution = 72.2um, 72.2um
time_resolution = 25ns
role = "dut"
type = "caen5748"

# ======================================
# The DUT at the 2nd motor -- Downstream 
# --> TI122
[CAEN_IJS_2]
material_budget = 0.00001 # ??
number_of_pixels = 2, 2
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 250um, 250um
position = 0, 0, 438mm
spatial_resolution = 72.2um, 72.2um
time_resolution = 25ns
role = "dut"
type = "caen5748"

[CAEN_UZH_2]
material_budget = 0.00001 # ??
number_of_pixels = 2, 2
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 250um, 250um
position = 0, 0, 438mm
spatial_resolution = 72.2um, 72.2um
time_resolution = 25ns
role = "dut"
type = "caen5748"
# ======================================


# ======================================
# The DUT at the 3rd motor -- Upstream
# --> TI116
[CAEN_IJS_1]
material_budget = 0.00001 # ??
number_of_pixels = 2, 2
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 250um, 250um
position = 0, 0, 516mm
spatial_resolution = 72.2um, 72.2um
time_resolution = 25ns
role = "dut"
type = "caen5748"

[CAEN_UZH_1]
material_budget = 0.00001 # ??
number_of_pixels = 2, 2
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 250um, 250um
position = 0, 0, 516mm
spatial_resolution = 72.2um, 72.2um
time_resolution = 25ns
role = "dut"
type = "caen5748"

# ======================================
# The DUT at the 3rd motor -- Downstream
# --> TI155
[CAEN_IJS_0]
material_budget = 0.00001 # ??
number_of_pixels = 2, 2
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 250um, 250um
position = 0, 0, 518mm
spatial_resolution = 72.2um, 72.2um
time_resolution = 25ns
role = "dut"
type = "caen5748"

[CAEN_UZH_0]
material_budget = 0.00001 # ??
number_of_pixels = 2, 2
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 250um, 250um
position = 0, 0, 518mm
spatial_resolution = 72.2um, 72.2um
time_resolution = 25ns
role = "dut"
type = "caen5748"
# ======================================


# ======================================
# The DUT at the 4th motor -- Downstream
# --> Right reference 
# --> HPK V2 Split4
[CAEN_IJS_6]
material_budget = 0.00001 # ??
number_of_pixels = 2, 2
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 999um, 999um
position = 0, 0, 617mm
spatial_resolution = 288.7um, 288.7um
time_resolution = 25ns
role = "dut"
type = "caen5748"

[CAEN_UZH_6]
material_budget = 0.00001 # ??
number_of_pixels = 2, 2
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 999um, 999um
position = 0, 0, 617mm
spatial_resolution = 288.7um, 288.7um
time_resolution = 25ns
role = "dut"
type = "caen5748"

# XXX -- TO BE REMOVED
# ======================================
# DUT_1 RD53A LF Foundry Fresh
# 
# The sensor layout is given in the form Y_X_connectivity. Y and X are the pixel pitch in the
# Y and X direction respectively. The connectivity to the readout chip can be odd or even.
# XXX -- TO BE REMOVED == UP

[MIMOSA26_3]
material_budget = 0.00075
number_of_pixels = 1152, 576
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 18.4um, 18.4um
position = 0, 0,732mm
spatial_resolution = 5.2um, 5.2um
time_resolution = 230us
type = "mimosa26"
roi = [367, 252], [367, 395], [511, 395], [511, 252]  

[MIMOSA26_4]
material_budget = 0.00075
number_of_pixels = 1152, 576
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 18.4um, 18.4um
position = 0, 0, 827mm
spatial_resolution = 5.2um, 5.2um
time_resolution = 230us
type = "mimosa26"
roi = [370, 274], [370, 415], [512, 415], [512, 274]  

[MIMOSA26_5]
material_budget = 0.00075
number_of_pixels = 1152, 576
orientation = 0deg, 0deg, 0deg
orientation_mode = "xyz"
pixel_pitch = 18.4um, 18.4um
position = 0, 0, 947mm
spatial_resolution = 5.2um, 5.2um
time_resolution = 230us
type = "mimosa26"
roi = [370, 232], [370, 377], [518, 377], [518, 232]  
