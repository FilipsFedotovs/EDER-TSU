#This simple script prepares 2-Track seeds for the initial CNN vertexing
# Part of EDER-TSU package
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

print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################     Initialising EDER-TSU Rendering module               ########################"+bcolors.ENDC)
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
      UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'M8', ['M8_M9'], "SoftUsed == \"EDER-TSU-M8\"")
      print(UF.TimeStamp(),'Submitting jobs... ',bcolors.ENDC)
      val_file=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M8_Validation_Set.pkl'
      if os.path.isfile(val_file)==False:
          print(UF.TimeStamp(),bcolors.FAIL+'Critical fail!', val_file, 'is missing. Please make sure that the previous script M7_GenerateImages.py has finished correctly '+bcolors.ENDC)
          exit()
      else:
          OptionHeader = [' --SetType ', ' --Fraction ', ' --EOS ', " --AFS ", " --resolution ",
                          " --MaxX ", " --MaxY ", " --MaxZ "]
          OptionLine = ['Val', 0, EOS_DIR, AFS_DIR, resolution, MaxX, MaxY, MaxZ]
          SHName = AFS_DIR + '/HTCondor/SH/SH_M8_Val.sh'
          SUBName = AFS_DIR + '/HTCondor/SUB/SUB_M8_Val.sub'
          MSGName = AFS_DIR + '/HTCondor/MSG/MSG_M8_Val'
          ScriptName = AFS_DIR + '/Code/Utilities/M8_RenderImages_Sub.py '
          UF.SubmitJobs2Condor(
              [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-M8', True,
               False])
          f_counter=0
          for f in range(1,100):
             train_file=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M8_Train_Set_'+str(f)+'.pkl'
             if os.path.isfile(train_file):
              f_counter=f
          OptionHeader = [' --SetType ', ' --Fraction ', ' --EOS ', " --AFS ", " --resolution ",
                          " --MaxX ", " --MaxY ", " --MaxZ "]
          OptionLine = ['Train', '$1', EOS_DIR, AFS_DIR, resolution, MaxX, MaxY, MaxZ]
          SHName = AFS_DIR + '/HTCondor/SH/SH_M8.sh'
          SUBName = AFS_DIR + '/HTCondor/SUB/SUB_M8.sub'
          MSGName = AFS_DIR + '/HTCondor/MSG/MSG_M8'
          ScriptName = AFS_DIR + '/Code/Utilities/M8_RenderImages_Sub.py '
          UF.SubmitJobs2Condor(
              [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, f_counter, 'EDER-TSU-M8', True,
               False])
      print(UF.TimeStamp(), bcolors.OKGREEN+'All jobs have been submitted, please rerun this script with "--Mode C" in few hours'+bcolors.ENDC)
if Mode=='C':
   print(UF.TimeStamp(),'Checking results... ',bcolors.ENDC)
   bad_pop=[]
   print(UF.TimeStamp(),'Checking jobs... ',bcolors.ENDC)
   val_file=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M8_M9_VALIDATION_SET.pkl'
   if os.path.isfile(val_file)==False:
           OptionHeader = [' --SetType ', ' --Fraction ', ' --EOS ', " --AFS ", " --resolution ",
                           " --MaxX ", " --MaxY ", " --MaxZ "]
           OptionLine = ['Val', 0, EOS_DIR, AFS_DIR, resolution, MaxX, MaxY, MaxZ]
           SHName = AFS_DIR + '/HTCondor/SH/SH_M8.sh_Val'
           SUBName = AFS_DIR + '/HTCondor/SUB/SUB_M8_Val.sub'
           MSGName = AFS_DIR + '/HTCondor/MSG/MSG_M8_Val'
           ScriptName = AFS_DIR + '/Code/Utilities/M8_RenderImages_Sub.py '
           bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-M8', False,
               False])
   for f in range(1,100):
              train_file=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M8_Train_Set_'+str(f)+'.pkl'
              req_train_file=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M8_M9_TRAIN_SET_'+str(f)+'.pkl'
              OptionHeader = [' --SetType ', ' --Fraction ', ' --EOS ', " --AFS ", " --resolution ",
                              " --MaxX ", " --MaxY ", " --MaxZ "]
              OptionLine = ['Train', f-1, EOS_DIR, AFS_DIR, resolution, MaxX, MaxY, MaxZ]
              SHName = AFS_DIR + '/HTCondor/SH/SH_M8_'+str(f-1)+'.sh'
              SUBName = AFS_DIR + '/HTCondor/SUB/SUB_M8_'+str(f-1)+'.sub'
              MSGName = AFS_DIR + '/HTCondor/MSG/MSG_M8'+str(f-1)
              ScriptName = AFS_DIR + '/Code/Utilities/M8_RenderImages_Sub.py '
              if os.path.isfile(req_train_file)!=True  and os.path.isfile(train_file):
                 bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-M8', False,
                   False])
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
            UF.SubmitJobs2Condor(bp)
        print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
        print(bcolors.BOLD+"Please check them in few hours"+bcolors.ENDC)
        exit()
   else:

       print(UF.TimeStamp(),bcolors.OKGREEN+'All HTCondor Image Rendering jobs have finished'+bcolors.ENDC)
       print(UF.TimeStamp(),'Cleaning up the work space... ',bcolors.ENDC)
       print(bcolors.BOLD+'Would you like to delete un-rendered images?'+bcolors.ENDC)
       UserAnswer=input(bcolors.BOLD+"Please, enter your option Y/N \n"+bcolors.ENDC)
       if UserAnswer=='Y':
           UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'M8', ['M7_M8'], "SoftUsed == \"EDER-TSU-M8\"")
       else:
        print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(), bcolors.OKGREEN+"Image rendering is completed"+bcolors.ENDC)
        print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#End of the script



