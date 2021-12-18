#This simple script prepares 2-Track seeds for the initial CNN vertexing
# Part of EDER-TSU package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
import math #We use it for data manipulation
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
parser = argparse.ArgumentParser(description='This script takes preselected truth 2-track seeds and decorates them with additional information such as DOCA and opening angle.')
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
MaxEvalSegmentsPerJob = PM.MaxEvalSegmentsPerJob
MaxTracksPerJob = PM.MaxTracksPerJob
#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E1_TRACK_SEGMENTS.csv'
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"####################  Initialising EDER-TSU MC Evaluation truth seed decoration module ###############"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(UF.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
data=pd.read_csv(input_file_location,header=0,usecols=['FEDRA_Seg_ID'])

print(UF.TimeStamp(),'Analysing data... ',bcolors.ENDC)
data.drop_duplicates(subset="FEDRA_Seg_ID",keep='first',inplace=True)  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
Records=len(data.axes[0])
SubSets=math.ceil(Records/MaxEvalSegmentsPerJob)
if Mode=='R':
   print(UF.TimeStamp(),bcolors.WARNING+'Warning! You are running the script with the "Mode R" option which means that you want to create the seeds from the scratch'+bcolors.ENDC)
   print(UF.TimeStamp(),bcolors.WARNING+'This option will erase all the previous Seed Creation jobs/results'+bcolors.ENDC)
   UserAnswer=input(bcolors.BOLD+"Would you like to continue (Y/N)? \n"+bcolors.ENDC)
   if UserAnswer=='N':
         Mode='C'
         print(UF.TimeStamp(),'OK, continuing then...')

   if UserAnswer=='Y':
      print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
      UF.EvalCleanUp(AFS_DIR, EOS_DIR, 'E3', ['E3_E3','E3_TRUTH'], "SoftUsed == \"EDER-TSU-E3\"")
      print(UF.TimeStamp(),'Submitting jobs... ',bcolors.ENDC)
      for sj in range(0,int(SubSets)):
            f_counter=0
            for f in range(0,1000):
             new_output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E2_E3_RawTracks_'+str(sj)+'_'+str(f)+'.csv'
             if os.path.isfile(new_output_file_location):
                 f_counter=f
            OptionHeader = [' --SubSet ', ' --EOS ', " --AFS ", " --Fraction "]
            OptionLine = [sj, EOS_DIR, AFS_DIR, '$1']
            SHName = AFS_DIR + '/HTCondor/SH/SH_E3_'+str(sj)+'.sh'
            SUBName = AFS_DIR + '/HTCondor/SUB/SUB_E3_'+str(sj)+'.sub'
            MSGName = AFS_DIR + '/HTCondor/MSG/MSG_E3_'+str(sj)
            ScriptName = AFS_DIR + '/Code/Utilities/E3_DecorateEvalTracks_Sub.py '
            UF.SubmitJobs2Condor(
                [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, f_counter+1, 'EDER-TSU-E3', False,
                 False])
      print(UF.TimeStamp(), bcolors.OKGREEN+'All jobs have been submitted, please rerun this script with "--Mode C" in few hours'+bcolors.ENDC)
if Mode=='C':
   print(UF.TimeStamp(),'Checking results... ',bcolors.ENDC)
   test_file=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E3_TRUTH_TRACKS.csv'
   if os.path.isfile(test_file):
       print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
       print(UF.TimeStamp(), bcolors.OKGREEN+"The process has been completed before, if you want to restart, please rerun with '--Mode R' option"+bcolors.ENDC)
       exit()
   bad_pop=[]
   print(UF.TimeStamp(),'Checking jobs... ',bcolors.ENDC)

   for sj in range(0,int(SubSets)):
           for f in range(0,1000):
              new_output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E2_E3_RawTracks_'+str(sj)+'_'+str(f)+'.csv'
              required_output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E3_E3_DecoratedTracks_'+str(sj)+'_'+str(f)+'.csv'
              OptionHeader = [' --SubSet ', ' --EOS ', " --AFS ", " --Fraction "]
              OptionLine = [sj, EOS_DIR, AFS_DIR, f]
              SHName = AFS_DIR + '/HTCondor/SH/SH_E3_' + str(sj) + '_'+str(f)+'.sh'
              SUBName = AFS_DIR + '/HTCondor/SUB/SUB_E3_' + str(sj) + '_'+str(f)+'.sub'
              MSGName = AFS_DIR + '/HTCondor/MSG/MSG_E3_' + str(sj)+ '_' +str(f)
              ScriptName = AFS_DIR + '/Code/Utilities/E3_DecorateEvalTracks_Sub.py '
              job_details=[OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-E3', False,
               False]
              if os.path.isfile(required_output_file_location)!=True and os.path.isfile(new_output_file_location):
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
             UF.SubmitJobs2Condor(bp)
        print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
        print(bcolors.BOLD+"Please check them in few hours"+bcolors.ENDC)
        exit()
   else:
       print(UF.TimeStamp(),bcolors.OKGREEN+'All HTCondor truth track decoration jobs have finished'+bcolors.ENDC)
       for sj in range(0,int(SubSets)):
           for f in range(0,1000):
             progress=int(round((float(sj)/float(int(SubSets)))*100,0))
             print("Collating the results, progress is ",progress,' %', end="\r", flush=True)
             new_output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E2_E3_RawTracks_'+str(sj)+'_'+str(f)+'.csv'
             required_output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E3_E3_DecoratedTracks_'+str(sj)+'_'+str(f)+'.csv'
             if os.path.isfile(required_output_file_location)!=True and os.path.isfile(new_output_file_location):
                 print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",required_output_file_location,'is missing, please restart the script with the option "--Mode R"'+bcolors.ENDC)
             elif os.path.isfile(required_output_file_location):
                 if sj==f==0:
                    base_data=pd.read_csv(required_output_file_location,names=['Segment_1','Segment_2','DOCA','Seg_Lon_Gap','Seg_Transv_Gap','angle'])
                 else:
                    new_data=pd.read_csv(required_output_file_location,names=['Segment_1','Segment_2','DOCA','Seg_Lon_Gap','Seg_Transv_Gap','angle'])
                    frames=[base_data,new_data]
                    base_data=pd.concat(frames)
       Records=len(base_data.axes[0])
       print(UF.TimeStamp(),'Set contains', Records, '2-track vertices',bcolors.ENDC)
       output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E3_TRUTH_TRACKS.csv'
       base_data["Track_ID"]= ['-'.join(sorted(tup)) for tup in zip(base_data['Segment_1'], base_data['Segment_2'])]
       base_data.drop_duplicates(subset="Track_ID",keep='first',inplace=True)
       base_data.drop(base_data.index[base_data['Segment_1'] == base_data['Segment_2']], inplace = True)
       base_data.drop(["Track_ID"],axis=1,inplace=True)
       Records_After_Compression=len(base_data.axes[0])
       if Records>0:
              Compression_Ratio=int((Records_After_Compression/Records)*100)
       else:
              CompressionRatio=0
       print(UF.TimeStamp(),'Set compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)
       base_data.to_csv(output_file_location,index=False)
       print(UF.TimeStamp(),'Cleaning up the work space... ',bcolors.ENDC)
       UF.EvalCleanUp(AFS_DIR, EOS_DIR, 'E3', ['E3_E3','E2_E3'], "SoftUsed == \"EDER-TSU-E3\"")
       print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
       print(UF.TimeStamp(), bcolors.OKGREEN+"2-track MC truth evaluation set ",bcolors.OKBLUE+output_file_location+bcolors.ENDC," is ready"+bcolors.ENDC)
       print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#End of the script



