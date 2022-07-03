#This simple script prepares data for CNN
########################################    Import libraries    #############################################
#import csv
import Utility_Functions as UF
import argparse
import pickle
import os, psutil #helps to monitor the memory

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
parser.add_argument('--Fraction',help="Fraction", default='1')
parser.add_argument('--EOS',help="EOS location", default='')
parser.add_argument('--AFS',help="AFS location", default='')
parser.add_argument('--resolution',help="Resolution in microns per pixel", default='100')
parser.add_argument('--MaxX',help="Image size in microns along the x-axis", default='3500.0')
parser.add_argument('--MaxY',help="Image size in microns along the y-axis", default='1000.0')
parser.add_argument('--MaxZ',help="Image size in microns along the z-axis", default='20000.0')
parser.add_argument('--SetType',help="Val/Train", default='Val')
########################################     Main body functions    #########################################
args = parser.parse_args()
fraction=int(args.Fraction)+1
resolution=float(args.resolution)
#Maximum bounds on the image size in microns
MaxX=float(args.MaxX)
MaxY=float(args.MaxY)
MaxZ=float(args.MaxZ)
#Converting image size bounds in line with resolution settings
AFS_DIR=args.AFS
EOS_DIR=args.EOS
if args.SetType=='Val':
 input_seed_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M8_Validation_Set.pkl'
 output_seed_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M8_M9_VALIDATION_SET.pkl'
if args.SetType=='Train':
 input_seed_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M8_Train_Set_'+str(fraction)+'.pkl'
 output_seed_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M8_M9_TRAIN_SET_'+str(fraction)+'.pkl'
print(UF.TimeStamp(),'Analysing the data')
image_file=open(input_seed_file_location,'rb')
images=pickle.load(image_file)
image_file.close()
print(UF.TimeStamp(),bcolors.OKGREEN+'Data has been successfully loaded and prepared..'+bcolors.ENDC)
print(UF.TimeStamp(),'Beginning the rendering part...')
for im in images:
    im.PrepareRawTrackPrint(MaxX,MaxY,MaxZ,resolution,False)
print(UF.TimeStamp(),bcolors.OKGREEN+'The image rendering has been completed..'+bcolors.ENDC)
print(UF.TimeStamp(),'Saving the results..')
print(output_seed_file_location)
print(len(images))
open_file = open(output_seed_file_location, "wb")
pickle.dump(images, open_file)
open_file.close()
exit()
