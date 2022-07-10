#This is the list of parameters that EDER-VIANN uses for reconstruction, model training etc. There have been collated here in one place for the user convenience
# Part of EDER-TSU package
#Made by Filips Fedotovs
#Current version 1.0

######List of naming conventions
x='x' #Column name x-coordinate of the track hit
y='y' #Column name for y-coordinate of the track hit
z='z' #Column name for z-coordinate of the track hit
FEDRA_Track_ID='FEDRA Track ID' #Column nameActual track id for FEDRA (or other reconstruction software)
FEDRA_Track_QUADRANT='Quadrant' #Quarter of the ECC where the track is reconstructed If not present in the data please put the Track ID (the same as above)
MC_Track_ID='MC Track'  #Column name for Track ID for MC Truth reconstruction data
MC_Event_ID='MC Event' #Column name for Event id for MC truth reconstruction data (If absent please enter the MCTrack as for above)
MC_Mother_PDG='Mother PDG'

Config='SND'
if Config=='SND':
    ########List of the package run parameters
    MaxSegmentsPerJob=10000 #This parameter imposes the limit on the number of the tracks form the Start plate when forming the Seeds.
    MaxFitTracksPerJob=10000
    MaxTracksPerTrPool=20000

    ######List of geometrical constain parameters
    MaxSLG=7000
    MaxSTG=160#This parameter restricts the maximum length of of the longitudinal and transverse distance between track segments.
    MinHitsTrack=2
    MaxTrainSampleSize=50000
    MaxValSampleSize=100000
    MaxDoca=50
    MinAngle=0 #Seed Opening Angle (Magnitude) in radians
    MaxAngle=1 #Seed Opening Angle (Magnitude) in radians

    ##Model parameters
    pre_acceptance=0.5
    post_acceptance=0.5
    #pre_vx_acceptance=0.662
    resolution=50
    MaxX=1000.0
    MaxY=1000.0
    MaxZ=3000.0
    Pre_CNN_Model_Name='1T_50_SHIP_PREFIT_1_model'
    Post_CNN_Model_Name='1T_50_SHIP_POSTFIT_1_model'
    #ModelArchitecture=[[6, 4, 1, 2, 2, 2, 2], [], [],[], [], [1, 4, 2], [], [], [], [], [7, 1, 1, 4]]
    ModelArchitecture=[[1, 4, 1, 2, 2, 2, 2], [1, 4, 1, 2, 2, 2, 2], [1, 4, 1, 2, 2, 2, 2],[], [], [1, 4, 2], [], [], [], [], [7, 1, 1, 4]]
    ModelArchitecturePlus=[[1, 4, [2, 2, 2], [32, 32, 32], 2, 2, 2], [], [],[], [], [1, 4, 2], [], [], [], [], [7, 1, 1, 4]]
else:
    ########List of the package run parameters
    MaxSegmentsPerJob=15000 #This parameter imposes the limit on the number of the tracks form the Start plate when forming the Seeds.
    MaxFitTracksPerJob=10000
    MaxTracksPerTrPool=20000

    ######List of geometrical constain parameters
    MaxSLG=4000
    MaxSTG=50#This parameter restricts the maximum length of of the longitudinal and transverse distance between track segments.
    MinHitsTrack=2
    MaxTrainSampleSize=50000
    MaxValSampleSize=100000
    MaxDoca=50
    MinAngle=0 #Seed Opening Angle (Magnitude) in radians
    MaxAngle=1 #Seed Opening Angle (Magnitude) in radians

    ##Model parameters
    pre_acceptance=0.5
    post_acceptance=0.5
    #pre_vx_acceptance=0.662
    resolution=50
    MaxX=2000.0
    MaxY=500.0
    MaxZ=20000.0
    Pre_CNN_Model_Name='1T_50_SHIP_PREFIT_1_model'
    Post_CNN_Model_Name='1T_50_SHIP_POSTFIT_1_model'
    ModelArchitecture=[[6, 4, 1, 2, 2, 2, 2], [], [],[], [], [1, 4, 2], [], [], [], [], [7, 1, 1, 4]]
