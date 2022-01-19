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
parser.add_argument('--MaxDOCA',help="Maximum DOCA allowed", default='50')
parser.add_argument('--MaxAngle',help="Maximum magnitude of angle allowed", default='1')
parser.add_argument('--MaxSTG',help="Maximum Segment Transverse gap per SLG", default='50')
parser.add_argument('--MaxSLG',help="Maximum Segment Longitudinal Gap", default='4000')
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
parser.add_argument('--PreFit',help="Would you like to prefit training track segment seeds by a light CNN model?", default='N')
parser.add_argument('--resolution',help="Resolution in microns per pixel", default='50')
parser.add_argument('--acceptance',help="CNN fit minimum acceptance", default='0.5')
parser.add_argument('--MaxX',help="Image size in microns along the x-axis", default='2000.0')
parser.add_argument('--MaxY',help="Image size in microns along the y-axis", default='500.0')
parser.add_argument('--MaxZ',help="Image size in microns along the z-axis", default='20000.0')
parser.add_argument('--ModelName',help="Name of the CNN model", default='1T_50_SHIP_PREFIT_1_model')
########################################     Main body functions    #########################################
args = parser.parse_args()
Set=args.Set
SubSet=args.SubSet
fraction=args.Fraction
AFS_DIR=args.AFS
EOS_DIR=args.EOS
PreFit=args.PreFit=='Y'
if PreFit:
    resolution=float(args.resolution)
    acceptance=float(args.acceptance)
    MaxX=float(args.MaxX)
    MaxY=float(args.MaxY)
    MaxZ=float(args.MaxZ)
    print(UF.TimeStamp(),'Loading the model...')
    #Load the model
    import tensorflow as tf
    from tensorflow import keras
    model_name=EOS_DIR+'/EDER-TSU/Models/'+args.ModelName
    model=tf.keras.models.load_model(model_name)
MaxDOCA=float(args.MaxDOCA)
MaxSTG=float(args.MaxSTG)
MaxSLG=float(args.MaxSLG)
MaxAngle=float(args.MaxAngle)
input_segment_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M1_TRACK_SEGMENTS.csv'
input_track_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M2_M3_RawTracks_'+Set+'_'+SubSet+'_'+fraction+'.csv'
output_track_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M3_M3_RawImages_'+Set+'_'+SubSet+'_'+fraction+'.pkl'
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
#create seeds
GoodTracks=[]
print(UF.TimeStamp(),'Beginning the image generation part...')
for s in range(0,limit):
    track=tracks.pop(0)
    label=track[2]
    track=Track(track[:2])
    if label:
        num_label = 1
    else:
        num_label = 0
    track.MCtruthClassifyTrack(num_label)
    track.DecorateSegments(segments)
    try:
      track.DecorateTrackGeoInfo()
    except:
      continue
    track.TrackQualityCheck(MaxDOCA,MaxSLG,MaxSTG, MaxAngle)
    if track.GeoFit and PreFit:
               track.PrepareTrackPrint(MaxX,MaxY,MaxZ,resolution,True)
               TrackImage=UF.LoadRenderImages([track],1,1)[0]
               track.UnloadTrackPrint()
               track.CNNFitTrack(model.predict(TrackImage)[0][1])
               if track.Track_CNN_Fit>=acceptance:
                  GoodTracks.append(track)
    elif track.GeoFit:
           GoodTracks.append(track)
    else:
        del track
        continue
print(UF.TimeStamp(),bcolors.OKGREEN+'The raw image generation has been completed..'+bcolors.ENDC)
del tracks
del segments
gc.collect()
print(UF.TimeStamp(),'Saving the results..')
open_file = open(output_track_file_location, "wb")
pickle.dump(GoodTracks, open_file)
open_file.close()
exit()
