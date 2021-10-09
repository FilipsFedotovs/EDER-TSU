#This simple script prepares 2-Track seeds for the initial CNN vertexing
# Part of EDER-VIANN package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
import pickle
import os


class bcolors:   #We use it for the interface
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script takes refined 2-track seed candidates from previous step and perfromes a vertex fit by using pre-trained CNN model.')
parser.add_argument('--Mode',help="Running Mode: Reset(R)/Continue(C)", default='C')

######################################## Set variables  #############################################################
args = parser.parse_args()
Mode=args.Mode




#Loading Directory locations
csv_reader=open('../config',"r")
config = list(csv.reader(csv_reader))
for c in config:
    if c[0]=='AFS_DIR':
        AFS_DIR=c[1]
    if c[0]=='EOS_DIR':
        EOS_DIR=c[1]
csv_reader.close()
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import Utility_Functions as UF #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters
########################################     Preset framework parameters    #########################################
resolution=PM.resolution
MaxX=PM.MaxX
MaxY=PM.MaxY
MaxZ=PM.MaxZ
 #The Separation bound is the maximum Euclidean distance that is allowed between hits in the beggining of Seed tracks.
MaxTracksPerJob = PM.MaxTracksPerJob
MaxSeedsPerJob = PM.MaxSeedsPerJob


print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"###################     Initialising EDER-VIANN Image rendering module          ########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)

if Mode=='R':
   print(UF.TimeStamp(),bcolors.WARNING+'Warning! You are running the script with the "Mode R" option which means that you want to render the seeds from the scratch'+bcolors.ENDC)
   UserAnswer=input(bcolors.BOLD+"Would you like to continue (Y/N)? \n"+bcolors.ENDC)
   if UserAnswer=='N':
         Mode='C'
         print(UF.TimeStamp(),'OK, continuing then...')

   if UserAnswer=='Y':
      print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
      UF.RecCleanUp(AFS_DIR, EOS_DIR, 'M4', ['M4_M5'], "SoftUsed == \"EDER-VIANN-M4\"")
      print(UF.TimeStamp(),'Submitting jobs... ',bcolors.ENDC)
      val_file=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_M4_Validation_Set.pkl'
      if os.path.isfile(val_file)==False:
          print(UF.TimeStamp(),bcolors.FAIL+'Critical fail!', val_file, 'is missing. Please make sure that the previous script M3_GenerateImages.py has finished correctly '+bcolors.ENDC)
          exit()
      else:
          job_details=['Val',1,resolution,MaxX,MaxY,MaxZ,AFS_DIR,EOS_DIR]
          UF.BSubmitRenderImageJobsCondor(job_details)
          f_counter=0
          for f in range(1,100):
             train_file=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_M4_Train_Set_'+str(f)+'.pkl'
             if os.path.isfile(train_file):
              f_counter=f
          job_details=['Train',f_counter,resolution,MaxX,MaxY,MaxZ,AFS_DIR,EOS_DIR]
          UF.BSubmitRenderImageJobsCondor(job_details)
      print(UF.TimeStamp(), bcolors.OKGREEN+'All jobs have been submitted, please rerun this script with "--Mode C" in few hours'+bcolors.ENDC)
if Mode=='C':
   print(UF.TimeStamp(),'Checking results... ',bcolors.ENDC)
   bad_pop=[]
   print(UF.TimeStamp(),'Checking jobs... ',bcolors.ENDC)
   val_file=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M4_M5_VALIDATION_SET.pkl'
   if os.path.isfile(val_file)==False:
         job_details=['Val',1,resolution,MaxX,MaxY,MaxZ,AFS_DIR,EOS_DIR]
         bad_pop.append(job_details)
   for f in range(1,100):
              train_file=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_M4_Train_Set_'+str(f)+'.pkl'
              req_train_file=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M4_M5_TRAIN_SET_'+str(f)+'.pkl'
              job_details=['Train',f,resolution,MaxX,MaxY,MaxZ,AFS_DIR,EOS_DIR]
              if os.path.isfile(req_train_file)!=True  and os.path.isfile(train_file):
                 bad_pop.append(job_details)
   if len(bad_pop)>0:
     print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
     print(bcolors.BOLD+'If you would like to wait and try again later please enter W'+bcolors.ENDC)
     print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
     UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
     if UserAnswer=='W':
         print(UF.TimeStamp(),'OK, exiting now then')
         exit()
     if UserAnswer=='R':
        for bp in bad_pop:
             UF.SubmitRenderImageJobsCondor(bp)
        print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
        print(bcolors.BOLD+"Please check them in few hours"+bcolors.ENDC)
        exit()
   else:

       print(UF.TimeStamp(),bcolors.OKGREEN+'All HTCondor Image Rendering jobs have finished'+bcolors.ENDC)
       print(UF.TimeStamp(),'Cleaning up the work space... ',bcolors.ENDC)
       print(bcolors.BOLD+'Would you like to delete un-rendered images?'+bcolors.ENDC)
       UserAnswer=input(bcolors.BOLD+"Please, enter your option Y/N \n"+bcolors.ENDC)
       if UserAnswer=='Y':
           UF.RecCleanUp(AFS_DIR, EOS_DIR, 'M4', ['M3_M4'], "SoftUsed == \"EDER-VIANN-M4\"")
       else:
        print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(), bcolors.OKGREEN+"Image rendering is completed"+bcolors.ENDC)
        print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#End of the script



