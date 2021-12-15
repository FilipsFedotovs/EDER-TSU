#This simple script prepares 2-segment strack for fake track studies
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
parser = argparse.ArgumentParser(description='This script selects and prepares 2-segment tracks that do not have a common Mother particle.')
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
MaxSLG=PM.MaxSLG
MaxSTG=PM.MaxSTG #These parameters restricts the maximum length of of the longitudinal and transverse distance between track segments.
MaxSegmentsPerJob = PM.MaxSegmentsPerJob
MaxTracksPerJob = PM.MaxTracksPerJob
#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R1_TRACK_SEGMENTS.csv'
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"####################     Initialising EDER-TSU Fake Track generation module          ###################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(UF.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
data=pd.read_csv(input_file_location,header=0,usecols=['FEDRA_Seg_ID','z'])
print(UF.TimeStamp(),'Analysing data... ',bcolors.ENDC)

data = data.groupby('FEDRA_Seg_ID')['z'].min()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
data=data.reset_index()
data = data.groupby('z')['FEDRA_Seg_ID'].count()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
data=data.reset_index()
data=data.sort_values(['z'],ascending=True)
data['Sub_Sets']=np.ceil(data['FEDRA_Seg_ID']/MaxSegmentsPerJob)
data['Sub_Sets'] = data['Sub_Sets'].astype(int)
data = data.values.tolist() #Convirting the result to List data type
if Mode=='R':
   print(UF.TimeStamp(),bcolors.WARNING+'Warning! You are running the script with the "Mode R" option which means that you want to create the seeds from the scratch'+bcolors.ENDC)
   print(UF.TimeStamp(),bcolors.WARNING+'This option will erase all the previous track Creation jobs/results'+bcolors.ENDC)
   UserAnswer=input(bcolors.BOLD+"Would you like to continue (Y/N)? \n"+bcolors.ENDC)
   if UserAnswer=='N':
         Mode='C'
         print(UF.TimeStamp(),'OK, continuing then...')

   if UserAnswer=='Y':
      print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
      UF.EvalCleanUp(AFS_DIR, EOS_DIR, 'E5', ['E5_E5','E5_E6'], "SoftUsed == \"EDER-TSU-E5\"")
      print(UF.TimeStamp(),'Submitting jobs... ',bcolors.ENDC)
      for j in range(0,len(data)):
          OptionHeader = [' --Set ', ' --Subset ', ' --EOS ', " --AFS ", " --PlateZ ", " --MaxSegments ", " --MaxSLG ", " --MaxSTG "]
          OptionLine = [j, '$1', EOS_DIR, AFS_DIR, int(data[j][0]), MaxTracksPerJob, MaxSLG, MaxSTG]
          SHName = AFS_DIR + '/HTCondor/SH/SH_E5_' + str(j) + '.sh'
          SUBName = AFS_DIR + '/HTCondor/SUB/SUB_E5_' + str(j) + '.sub'
          MSGName = AFS_DIR + '/HTCondor/MSG/MSG_E5_' + str(j)
          ScriptName = AFS_DIR + '/Code/Utilities/E5_GenerateFakeTracks_Sub.py '
          UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, int(data[j][2]), 'EDER-VIANN-E5', False,False])
      print(UF.TimeStamp(), bcolors.OKGREEN+'All jobs have been submitted, please rerun this script with "--Mode C" in few hours'+bcolors.ENDC)
if Mode=='C':
   bad_pop=[]
   print(UF.TimeStamp(),'Checking jobs... ',bcolors.ENDC)
   for j in range(0,len(data)):
       for sj in range(0,int(data[j][2])):
           OptionHeader = [' --Set ', ' --Subset ', ' --EOS ', " --AFS ", " --PlateZ ", " --MaxTracks "," --MaxSLG ", " --MaxSTG "]
           OptionLine = [j, sj, EOS_DIR, AFS_DIR, int(data[j][0]), MaxSegmentsPerJob,MaxSLG, MaxSTG]
           SHName = AFS_DIR + '/HTCondor/SH/SH_E5_' + str(j) + '_' + str(sj) + '.sh'
           SUBName = AFS_DIR + '/HTCondor/SUB/SUB_E5_' + str(j) + '_' + str(sj) + '.sub'
           MSGName = AFS_DIR + '/HTCondor/MSG/MSG_E5_' + str(j) + '_' + str(sj)
           ScriptName = AFS_DIR + '/Code/Utilities/E5_GenerateFakeTracks_Sub.py '
           job_details = [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-E5', False,False]
           output_file_location=EOS_DIR+'/EDER-VIANN/Data/TEST_SET/E5_E5_RawTracks_'+str(j)+'_'+str(sj)+'.csv'
           output_result_location=EOS_DIR+'/EDER-VIANN/Data/TEST_SET/E5_E5_RawTracks_'+str(j)+'_'+str(sj)+'_RES.csv'
           if os.path.isfile(output_result_location) == False:
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
       print(UF.TimeStamp(),bcolors.OKGREEN+'All HTCondor track creation jobs have finished'+bcolors.ENDC)
       print(UF.TimeStamp(),'Collating the results...')
       input_file_location=EOS_DIR+'/EDER-VIANN/Data/TEST_SET/E1_TRACK_SEGMENTS.csv'
       eval_data=pd.read_csv(input_file_location,header=0,usecols=['FEDRA_Seg_ID','MC_Mother_Track_ID'])
       eval_data.drop_duplicates(subset=['FEDRA_Seg_ID','MC_Mother_Track_ID'],keep='first',inplace=True)
       eval_data=eval_data.rename(columns={'FEDRA_Seg_ID': 'Track_1'})
       eval_data=eval_data.rename(columns={'MC_Mother_Track_ID': 'Mother_1'})
       eval_data2=eval_data.rename(columns={'Track_1': 'Track_2'})
       eval_data2=eval_data2.rename(columns={'Mother_1': 'Mother_2'})
       for j in range(0,len(data)):
        for sj in range(0,int(data[j][2])):
           output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E5_E5_RawTracks_'+str(j)+'_'+str(sj)+'.csv'
           if os.path.isfile(output_file_location):
            result=pd.read_csv(output_file_location,names = ['Track_1','Track_2'])
            Records=len(result.axes[0])
            print(UF.TimeStamp(),'Set',str(j),'and subset', str(sj), 'contains', Records, 'tracks',bcolors.ENDC)
            result["Track_ID"]= ['-'.join(sorted(tup)) for tup in zip(result['Track_1'], result['Track_2'])]
            result.drop_duplicates(subset="Track_ID",keep='first',inplace=True)
            result.drop(result.index[result['Track_1'] == result['Track_2']], inplace = True)
            result.drop(["Track_ID"],axis=1,inplace=True)
            result=pd.merge(result, eval_data, how="left", on=['Track_1'])
            result=pd.merge(result, eval_data2, how="left", on=['Track_2'])
            result.drop(result.index[(result['Mother_1'] == result['Mother_2']) & (result['Mother_1']!='') & (result['Mother_2']!='')], inplace = True)
            result.drop(['Mother_1'],axis=1,inplace=True)
            result.drop(['Mother_2'],axis=1,inplace=True)
            Records_After_Compression=len(result.axes[0])
            if Records>0:
              Compression_Ratio=int((Records_After_Compression/Records)*100)
            else:
              CompressionRatio=0
            print(UF.TimeStamp(),'Set',str(j),'and subset', str(sj), 'compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)
            fractions=int(math.ceil(Records_After_Compression/MaxTracksPerJob))
            for f in range(0,fractions):
             new_output_file_location=EOS_DIR+'/EDER-VIANN/Data/TEST_SET/E5_E6_RawTracks_'+str(j)+'_'+str(sj)+'_'+str(f)+'.csv'
             result[(f*MaxTracksPerJob):min(Records_After_Compression,((f+1)*MaxTracksPerJob))].to_csv(new_output_file_location,index=False)
            os.unlink(output_file_location)
       print(UF.TimeStamp(),'Cleaning up the work space... ',bcolors.ENDC)
       UF.EvalCleanUp(AFS_DIR, EOS_DIR, 'E5', ['E5_E5'], "SoftUsed == \"EDER-TSU-E5\"")
       print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
       print(UF.TimeStamp(), bcolors.OKGREEN+"Fake track generation is completed."+bcolors.ENDC)
       print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#End of the script

