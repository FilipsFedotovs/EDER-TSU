#This simple script prepares data for CNN
########################################    Import libraries    #############################################
#import csv
import Utility_Functions as UF
from Utility_Functions import Track
import argparse
import pandas as pd #We use Panda for a routine data processing
import os, psutil #helps to monitor the memory
import gc  #Helps to clear memory
import pickle
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--Set',help="Set Number", default='1')
parser.add_argument('--SubSet',help="SubSet Number", default='1')
parser.add_argument('--Fraction',help="Fraction", default='1')
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
parser.add_argument('--resolution',help="Resolution in microns per pixel", default='50')
parser.add_argument('--acceptance',help="CNN fit minimum acceptance", default='0.5')
parser.add_argument('--MaxX',help="Image size in microns along the x-axis", default='2000.0')
parser.add_argument('--MaxY',help="Image size in microns along the y-axis", default='500.0')
parser.add_argument('--MaxZ',help="Image size in microns along the z-axis", default='20000.0')
parser.add_argument('--ModelName',help="Name of the CNN model", default='1T_50_SHIP_PREFIT_1_model')
parser.add_argument('--MotherPDGList', help="Target Mother PDGs", nargs='+', type=int, default='22')
########################################     Main body functions    #########################################
args = parser.parse_args()
Set=args.Set
SubSet=args.SubSet
fraction=args.Fraction
AFS_DIR=args.AFS
EOS_DIR=args.EOS
MotherPDGList = args.MotherPDGList
# in case there is only one  Mother PDG needed
if type(MotherPDGList)== int :
    MotherPDGList = [MotherPDGList]



input_segment_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/bM1_TRACK_SEGMENTS.csv'
output_track_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/bM3_M3_RawImages_'+Set+'_'+SubSet+'_'+fraction+'.pkl'
print(UF.TimeStamp(),'Loading the data')


segments=pd.read_csv(input_segment_file_location)
print(UF.TimeStamp(),'Analysing the data')

segments["x"] = pd.to_numeric(segments["x"],downcast='float')
segments["y"] = pd.to_numeric(segments["y"],downcast='float')
segments["z"] = pd.to_numeric(segments["z"],downcast='float')
tracks = segments.drop(columns=["x","y","z"])
tracks = tracks.drop_duplicates()
segments = segments.values.tolist() #Convirting the result to List data type
tracks = tracks.values.tolist() #Convirting the result to List data type

gc.collect()

track_counter=0
print(UF.TimeStamp(),bcolors.OKGREEN+'Data has been successfully loaded and prepared..'+bcolors.ENDC)
#create seeds
GoodTracks=[]
print(UF.TimeStamp(),'Beginning the image generation part...')
limit = len(tracks)

for s in range(0,limit):
    track=tracks.pop(0)

    label=(track[1] in MotherPDGList)
    # for test
    #print(track[0], track[1], label)
    track=Track(track[0])
    if label:
        num_label = 1
    else:
        num_label = 0
    track.MCtruthClassifyTrack(num_label)
    GoodTracks.append(track)
    #track.DecorateSegments(segments) 
    #try:
    #  track.DecorateTrackGeoInfo()
    #except:
    #  continue
    #track.TrackQualityCheck(MaxDOCA,MaxSLG,MaxSTG, MaxAngle)
    #if track.GeoFit and PreFit:
    #           track.PrepareTrackPrint(MaxX,MaxY,MaxZ,resolution,True)
    #           TrackImage=UF.LoadRenderImages([track],1,1)[0]
    #           track.UnloadTrackPrint()
    #           track.CNNFitTrack(model.predict(TrackImage)[0][1])
    #           if track.Track_CNN_Fit>=acceptance:
    #              GoodTracks.append(track)
    #elif track.GeoFit:
    #       GoodTracks.append(track)
    #else:
   # del track
   # continue
print(UF.TimeStamp(),bcolors.OKGREEN+'The raw image generation has been completed..'+bcolors.ENDC)
del tracks
del segments
gc.collect()
print(UF.TimeStamp(),'Saving the results..')
open_file = open(output_track_file_location, "wb")
pickle.dump(GoodTracks, open_file)
open_file.close()
exit()
