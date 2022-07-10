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
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
parser.add_argument('--MaxFitTracksPerJob',help="Max tracks per job", default='10000')
########################################     Main body functions    #########################################
args = parser.parse_args()
Set=args.Set
AFS_DIR=args.AFS
EOS_DIR=args.EOS
MaxFitTracksPerJob=int(args.MaxFitTracksPerJob)
start_index = (int(Set)-1)*MaxFitTracksPerJob
end_index = (int(Set))*MaxFitTracksPerJob 

input_segment_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R7_TRACK_SEGMENTS.csv'
input_track_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R7_R8_TRACK_HEADERS.csv'
output_track_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R8_R8_PREPARED_TRACKS_'+Set+'.pkl'
print(UF.TimeStamp(),'Loading the data')
tracks=pd.read_csv(input_track_file_location)
tracks = tracks[start_index:min(end_index,len(tracks))]
segments=pd.read_csv(input_segment_file_location)
print(segments)
print(tracks)
exit()
print(UF.TimeStamp(),'Analysing the data')
segments=pd.merge(segments, tracks, how="inner", on=["FEDRA_Seg_ID"]) #Shrinking the Track data so just a star hit for each segment is present.
segments["x"] = pd.to_numeric(segments["x"],downcast='float')
segments["y"] = pd.to_numeric(segments["y"],downcast='float')
segments["z"] = pd.to_numeric(segments["z"],downcast='float')
segments=segments[['x','y','z','FEDRA_Seg_ID']]
segments = segments.values.tolist() #Convirting the result to List data type
tracks = tracks.values.tolist() #Convirting the result to List data type
del tracks_1
del tracks_2
del track_list
gc.collect()
limit=len(tracks)
track_counter=0
print(UF.TimeStamp(),bcolors.OKGREEN+'Data has been successfully loaded and prepared..'+bcolors.ENDC)
#create seeds
GoodTracks=[]
print(UF.TimeStamp(),'Beginning the image generation part...')
for s in range(0,limit):
    track=tracks.pop(0)
    track=Track(track[:2])
    track.DecorateSegments(segments)
    try:
      track.DecorateTrackGeoInfo()
    except:
      continue
    track.TrackQualityCheck(MaxDOCA,MaxSLG,MaxSTG, MaxAngle)
    if track.GeoFit:
           GoodTracks.append(track)
    else:
        del track
        continue
print(UF.TimeStamp(),bcolors.OKGREEN+'The track decoration has been completed..'+bcolors.ENDC)
del tracks
del segments
gc.collect()
print(UF.TimeStamp(),'Saving the results..')
open_file = open(output_track_file_location, "wb")
pickle.dump(GoodTracks, open_file)
open_file.close()
exit()
