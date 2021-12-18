#This simple script prepares data for CNN
########################################    Import libraries    #############################################
#import csv
import Utility_Functions as UF
from Utility_Functions import Track
import argparse
import pandas as pd #We use Panda for a routine data processing

import os, psutil #helps to monitor the memory
import gc  #Helps to clear memory

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
parser.add_argument('--MaxDOCA',help="Maximum DOCA allowed", default='1')
parser.add_argument('--MaxAngle',help="Maximum magnitude of angle allowed", default='1')
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
########################################     Main body functions    #########################################
args = parser.parse_args()
Set=args.Set
SubSet=args.SubSet
fraction=args.Fraction
AFS_DIR=args.AFS
EOS_DIR=args.EOS

input_segment_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R1_TRACK_SEGMENTS.csv'
input_track_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E5_E6_RawTracks_'+Set+'_'+SubSet+'_'+fraction+'.csv'
output_track_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E6_DEC_FAKE_SEEDS_'+Set+'_'+SubSet+'_'+fraction+'.csv'
print(UF.TimeStamp(),'Loading the data')
tracks=pd.read_csv(input_track_file_location)
tracks_1=tracks.drop(['Segment_2'],axis=1)
tracks_1=tracks_1.rename(columns={"Segment_1": "FEDRA_Seg_ID"})
tracks_2=tracks.drop(['Segment_1'],axis=1)
tracks_2=tracks_2.rename(columns={"Segment_2": "FEDRA_Seg_ID"})
track_list=result = pd.concat([tracks_1,tracks_2])
track_list=track_list.sort_values(['FEDRA_Seg_ID'])
track_list.drop_duplicates(subset="FEDRA_Seg_ID",keep='first',inplace=True)
segments=pd.read_csv(input_segment_file_location)
print(UF.TimeStamp(),'Analysing the data')
segments=pd.merge(segments, track_list, how="inner", on=["FEDRA_Seg_ID"]) #Shrinking the Track data so just a star hit for each segment is present.
segments["x"] = pd.to_numeric(segments["x"],downcast='float')
segments["y"] = pd.to_numeric(segments["y"],downcast='float')
segments["z"] = pd.to_numeric(segments["z"],downcast='float')
segments = segments.values.tolist() #Convirting the result to List data type
tracks = tracks.values.tolist() #Convirting the result to List data type
del tracks_1
del tracks_2
del track_list
gc.collect()

limit=len(tracks)
track_counter=0
print(UF.TimeStamp(),bcolors.OKGREEN+'Data has been successfully loaded and prepared..'+bcolors.ENDC)
#create tracks
GoodTracks=[]
Header=['Track_1','Track_2','DOCA','Seg_Lon_Gap','Seg_Transv_Gap','angle']
GoodTracks.append(Header)
print(UF.TimeStamp(),'Beginning the vertexing part...')
for s in range(0,limit):
    track=Track(tracks.pop(0))
    track.DecorateTracks(segments)
    try:
      track.DecorateTrackGeoInfo()
      new_track=[track.TrackHeader[0],track.TrackHeader[1],track.DOCA,track.V_Tr[0],track.V_Tr[1],track.Tr_Tr,track.angle]
    except:
      new_track=[track.TrackHeader[0],track.TrackHeader[1],'Fail','Fail','Fail','Fail','Fail','Fail','Fail','Fail']
    GoodTracks.append(new_track)
print(UF.TimeStamp(),bcolors.OKGREEN+'The fake track decoration has been completed..'+bcolors.ENDC)
del segments
del tracks
gc.collect()
print(UF.TimeStamp(),'Saving the results..')
UF.LogOperations(output_track_file_location,'StartLog',GoodTracks)
exit()
