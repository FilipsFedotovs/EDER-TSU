#Belogs to: EDER-VIANN
#Purpose: To setup all working directories for the new user

########################################    Import libraries    #############################################
import csv
import os, shutil
import argparse
from shutil import copyfile
import subprocess
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
######################################## Work out whether user wants to uninstall the config ################
parser = argparse.ArgumentParser(description='Setup creation parameters')
parser.add_argument('--REMOVE',help="If you want to uninstall please enter Y", default='N')
args = parser.parse_args()
UserChoice=args.REMOVE
######################################## Create some functions to simplify the code   #######################
def FolderCreate(DIR):
    try:
      os.mkdir(DIR)
      print(bcolors.OKBLUE+DIR+bcolors.ENDC, bcolors.OKGREEN+'folder has been created...'+bcolors.ENDC)
    except:
      print(bcolors.FAIL+'Problem creating '+bcolors.ENDC,bcolors.OKBLUE+DIR+bcolors.ENDC, bcolors.FAIL+'folder, probably it already exists...'+bcolors.ENDC)

if UserChoice=='Y':
   csv_reader=open('config',"r")
   dir_reader = csv.reader(csv_reader)
   for row in dir_reader:
     if row[0]=='EOS_DIR':
       DEL_DIR=row[1]
       DEL_DIR+='/'+'EDER-VIANN'
       shutil.rmtree(DEL_DIR)
       break
   shutil.rmtree('HTCondor')
   shutil.rmtree('Code')
   os.remove('config')
   print(bcolors.OKGREEN+'Uninstallation complete, you can delete setup.py and its parent directory manually if you wish'+bcolors.ENDC)
   exit()

########################################     Work out and registering the current directory    #########################################
CurrDir=os.getcwd()
print('Current directory is set as:', bcolors.OKBLUE+os.getcwd()+bcolors.ENDC)
#Writing this directory into csv file so the software knows where to look data/code
csv_writer=open('config',"w")
dir_writer = csv.writer(csv_writer)
string_to_write=[]
string_to_write.append('AFS_DIR')
string_to_write.append(CurrDir)
dir_writer.writerow(string_to_write)
print(bcolors.OKGREEN+'Created the configuration file'+bcolors.ENDC)

########################################     Create directories for HTCondor    #########################################
FolderCreate('Code/HTCondor')
FolderCreate('Code/HTCondor/MSG')
FolderCreate('Code/HTCondor/SH')

#########################################   Workout EOS directory #################################
EOSDir=input(bcolors.BOLD+"Please enter the full path of your directory on EOS:\n"+bcolors.ENDC)
#Writing this directory into csv file so the software knows where to look data/code
csv_writer=open('config',"a")
dir_writer = csv.writer(csv_writer)
string_to_write=[]
string_to_write.append('EOS_DIR')
string_to_write.append(EOSDir)
dir_writer.writerow(string_to_write)
print(bcolors.OKGREEN+'Updated the directory mapping file with EOS location'+bcolors.ENDC)

########################################     Create sub-directories on EOS    #########################################

EOSsubDIR=EOSDir+'/'+'EDER-VIANN'
EOSsubDataDIR=EOSsubDIR+'/'+'Data'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
EOSsubTrainDIR=EOSsubDataDIR+'/'+'TRAIN_SET'
EOSsubRecDIR=EOSsubDataDIR+'/'+'REC_SET'
EOSsubTestDIR=EOSsubDataDIR+'/'+'TEST_SET'

FolderCreate(EOSsubDIR)
FolderCreate(EOSsubDataDIR)
FolderCreate(EOSsubModelDIR)
FolderCreate(EOSsubTrainDIR)
FolderCreate(EOSsubRecDIR)
FolderCreate(EOSsubTestDIR)

#########################################   Workout out training and validation files #################################
print(bcolors.BOLD+'Copying the models...'+bcolors.ENDC)
#Copying the pretrained models to the user directory
ModelOrigin='/eos/experiment/ship/data/EDER-VIANN/Models/'
src_files = os.listdir(TrainOrigin)
for file_name in src_files:
    full_file_name = os.path.join(ModelOrigin, file_name)
    if os.path.isfile(full_file_name) and (('model' in full_file_name)==True):
        print('Copying file', full_file_name, 'from ',bcolors.OKBLUE+TrainOrigin+bcolors.ENDC,'into', bcolors.OKBLUE+EOSsubTrainDIR+bcolors.ENDC)
        shutil.copy(full_file_name, EOSsubModelDIR)

print(bcolors.OKGREEN+'EDER-VIANN setup is successfully completed' +bcolors.ENDC)
exit()
