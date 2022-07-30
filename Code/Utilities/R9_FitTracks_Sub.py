#This simple script prepares data for CNN
########################################    Import libraries    #############################################
#import csv
import Utility_Functions as UF
import argparse
import pandas as pd #We use Panda for a routine data processing
import pickle
import tensorflow as tf
from tensorflow import keras

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
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
parser.add_argument('--resolution',help="Resolution in microns per pixel", default='100')
parser.add_argument('--bg_acceptance',help="CNN classifier minimum acceptance", default='0.5')
parser.add_argument('--MaxX',help="Image size in microns along the x-axis", default='2000.0')
parser.add_argument('--MaxY',help="Image size in microns along the y-axis", default='500.0')
parser.add_argument('--MaxZ',help="Image size in microns along the z-axis", default='20000.0')
parser.add_argument('--ClassifierModelName',help="Name of the CNN model", default='1T_50_SHIP_POSTFIT_1_model')
########################################     Main body functions    #########################################
args = parser.parse_args()
Set=args.Set
resolution=float(args.resolution)
# acceptance=float(args.bg_acceptance)
#Maximum bounds on the image size in microns
MaxX=float(args.MaxX)
MaxY=float(args.MaxY)
MaxZ=float(args.MaxZ)
numClasses = 3
#Converting image size bounds in line with resolution settings
AFS_DIR=args.AFS
EOS_DIR=args.EOS
input_track_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R8_R9_PREPARED_TRACKS_'+Set+'.pkl'
output_track_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R9_R9_CNN_Classified_Tracks_'+Set+'.pkl'
print(UF.TimeStamp(),'Analysing the data')
tracks_file=open(input_track_file_location,'rb')
tracks=pickle.load(tracks_file)
tracks_file.close()
limit=len(tracks)
track_counter=0
print(UF.TimeStamp(),bcolors.OKGREEN+'Data has been successfully loaded and prepared..'+bcolors.ENDC)
print(UF.TimeStamp(),'Loading the model...')
#Load the model
model_name=EOS_DIR+'/EDER-TSU/Models/'+args.ClassifierModelName
model=tf.keras.models.load_model(model_name)

#union tracks
GoodTracks=[]
print(UF.TimeStamp(),'Beginning the classify part...')
for s in range(0,limit):
    track=tracks.pop(0)
    track.PrepareRawTrackPrint(MaxX,MaxY,MaxZ,resolution,False)
    TrackImage=UF.LoadRenderImages([track],1,1,numClasses)[0]
    track.UnloadTrackPrint()
    track.CNNClassifyTrack(model.predict(TrackImage))
    GoodTracks.append(track)
print(UF.TimeStamp(),bcolors.OKGREEN+'The track segment CNN fit has been completed..'+bcolors.ENDC)
del tracks
gc.collect()
print(UF.TimeStamp(),'Saving the results..')
open_file = open(output_track_file_location, "wb")
pickle.dump(GoodTracks, open_file)
open_file.close()
exit()
