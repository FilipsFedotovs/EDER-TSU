#This simple script prepares 2-Track seeds for the initial CNN vertexing
# Part of EDER-VIANN package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
import math #We use it for data manipulation
import gc  #Helps to clear memory
import numpy as np
import os



class bcolors:   #We use it for the interface text colouring
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

parser = argparse.ArgumentParser(description='This script selects and prepares 2-track seeds that have a common Mother particle according to Monte-Carlo.')
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
########################################
# Preset framework parameters    #########################################
MaxSegmentsPerJob = PM.MaxSegmentsPerJob #These parameteres help to keep each HTCondor job size small enough to be executed without crash.
#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E1_TRACK_SEGMENTS.csv'
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"####################     Initialising EDER-TSU Truth test seed generation module   ###################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(UF.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
data=pd.read_csv(input_file_location,header=0,usecols=['FEDRA_Seg_ID'])
print(UF.TimeStamp(),'Analysing data... ',bcolors.ENDC)
data.drop_duplicates(subset="FEDRA_Seg_ID",keep='first',inplace=True)  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
Records=len(data.axes[0])
SubSets=np.ceil(Records/MaxSegmentsPerJob) #Splitting jobs into subsets
if Mode=='R':
   print(UF.TimeStamp(),bcolors.WARNING+'Warning! You are running the script with the "Mode R" option which means that you want to create the seeds from the scratch'+bcolors.ENDC)
   print(UF.TimeStamp(),bcolors.WARNING+'This option will erase all the previous Seed Creation jobs/results'+bcolors.ENDC)
   UserAnswer=input(bcolors.BOLD+"Would you like to continue (Y/N)? \n"+bcolors.ENDC)
   if UserAnswer=='N':
         Mode='C'
         print(UF.TimeStamp(),'OK, continuing then...')

   if UserAnswer=='Y':
      print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
      #Cleaning-up the folder
      UF.EvalCleanUp(AFS_DIR, EOS_DIR, 'E2', ['E2_E2','E2_E3'], "SoftUsed == \"EDER-TSU-E2\"")
      print(UF.TimeStamp(),'Submitting jobs... ',bcolors.ENDC)
      # Prepare HTCondor job submission parameters
      OptionHeader = [' --SubSet ', ' --EOS ', " --AFS ", " --MaxSegments "]
      OptionLine = ['$1', EOS_DIR, AFS_DIR, MaxSegmentsPerJob]
      SHName = AFS_DIR + '/HTCondor/SH/SH_E2.sh'
      SUBName = AFS_DIR + '/HTCondor/SUB/SUB_E2.sub'
      MSGName = AFS_DIR + '/HTCondor/MSG/MSG_E2'
      ScriptName = AFS_DIR + '/Code/Utilities/E2_GenerateEvalTracks_Sub.py '
      #Submit the HTCondor jobs to HTCondor
      UF.SubmitJobs2Condor(
              [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, int(SubSets), 'EDER-TSU-E2', False,
               False])
      print(UF.TimeStamp(), bcolors.OKGREEN+'All jobs have been submitted, please rerun this script with "--Mode C" in few hours'+bcolors.ENDC)
if Mode=='C':
   bad_pop=[]
   print(UF.TimeStamp(),'Checking jobs... ',bcolors.ENDC)
   for sj in range(0,int(SubSets)):
           #Prepare HTCondor job submission parameters
           OptionHeader = [' --SubSet ', ' --EOS ', " --AFS ", " --MaxSegments "]
           OptionLine = [sj, EOS_DIR, AFS_DIR, MaxSegmentsPerJob]
           SHName = AFS_DIR + '/HTCondor/SH/SH_E2_'+str(sj)+'.sh'
           SUBName = AFS_DIR + '/HTCondor/SUB/SUB_E2_'+str(sj)+'.sub'
           MSGName = AFS_DIR + '/HTCondor/MSG/MSG_E2_'+str(sj)
           ScriptName = AFS_DIR + '/Code/Utilities/E2_GenerateEvalTracks_Sub.py '
           job_details=[OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-E2', False,False]
           output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E2_E2_RawTracks_'+str(sj)+'.csv'
           output_result_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E2_E2_RawTracks_'+str(sj)+'_RES.csv'
           if os.path.isfile(output_result_location)==False:
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
             #Resubmit HTCondor jobs that have failed at the first attempt.
             UF.SubmitJobs2Condor(bp)
        print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
        print(bcolors.BOLD+"Please check them in few hours"+bcolors.ENDC)
        exit()
   else:
       print(UF.TimeStamp(),bcolors.OKGREEN+'All HTCondor truth seed generation jobs have finished'+bcolors.ENDC)
       print(UF.TimeStamp(),'Collating the results...')
       for sj in range(0,int(SubSets)):
           output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E2_E2_RawTracks_'+str(sj)+'.csv'
           result=pd.read_csv(output_file_location,names = ['Segment_1','Segment_2'])
           Records=len(result.axes[0])
           print(UF.TimeStamp(),'Subset', str(sj), 'contains', Records, 'seeds',bcolors.ENDC)
           #Removing duplicates.
           result["Track_ID"]= ['-'.join(sorted(tup)) for tup in zip(result['Segment_1'], result['Segment_2'])]
           result.drop_duplicates(subset="Track_ID",keep='first',inplace=True)
           result.drop(result.index[result['Segment_1'] == result['Segment_2']], inplace = True)
           result.drop(["Track_ID"],axis=1,inplace=True)
           Records_After_Compression=len(result.axes[0])
           if Records>0:
              Compression_Ratio=int((Records_After_Compression/Records)*100)
           else:
              CompressionRatio=0
           print(UF.TimeStamp(),'Subset', str(sj), 'compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC) #Compression ratio = Deduplicated Set/Not deduplicated set
           fractions=int(math.ceil(Records_After_Compression/MaxSegmentsPerJob))
           for f in range(0,fractions):
             new_output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E2_E3_RawTracks_'+str(sj)+'_'+str(f)+'.csv'
             result[(f*MaxSegmentsPerJob):min(Records_After_Compression,((f+1)*MaxSegmentsPerJob))].to_csv(new_output_file_location,index=False) #Splitting sets for the next script

       print(UF.TimeStamp(),'Cleaning up the work space... ',bcolors.ENDC)
       UF.EvalCleanUp(AFS_DIR, EOS_DIR, 'E2', ['E2_E2'], "SoftUsed == \"EDER-TSU-E2\"") #Cleaning up the EOS directory and HCondor logs
       print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
       print(UF.TimeStamp(), bcolors.OKGREEN+"Truth seed generation is completed"+bcolors.ENDC)
       print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)

#End of the script



