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
parser = argparse.ArgumentParser(description='This script takes preselected 2-track seeds from the previous step and decorates them with additional information such as DOCA and opening angle.')
parser.add_argument('--Mode',help="Running Mode: Reset(R)/Continue(C)", default='C')
parser.add_argument('--MaxDOCA',help="Maximum DOCA allowed", default='500')
parser.add_argument('--MaxAngle',help="Maximum magnitude of angle allowed", default='1.5')
parser.add_argument('--MaxSTG',help="Maximum Segment Transverse gap per SLG", default='1000')
parser.add_argument('--MaxSLG',help="Maximum Segment Longitudinal Gap", default='6000')
parser.add_argument('--DOCABin',help="The size of bins for DOCA values", default='100')
parser.add_argument('--AngleBin',help="The size of bins for Angle values", default='0.1')
parser.add_argument('--STGBin',help="The size of bins for STG values", default='100')
parser.add_argument('--SLGBin',help="The size of bins for SLG values", default='100')
######################################## Set variables  #############################################################
args = parser.parse_args()
Mode=args.Mode
MaxDOCA=float(args.MaxDOCA)
MaxSTG=float(args.MaxSTG)
MaxSLG=float(args.MaxSLG)
MaxAngle=float(args.MaxAngle)

DOCABin=float(args.DOCABin)
STGBin=float(args.STGBin)
SLGBin=float(args.SLGBin)
AngleBin=float(args.AngleBin)
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
 #The Separation bound is the maximum Euclidean distance that is allowed between hits in the beggining of Seed tracks.
MaxSegmentsPerJob = PM.MaxSegmentsPerJob
MaxTracksPerJob = PM.MaxTracksPerJob
#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R1_TRACK_SEGMENTS.csv'
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################    Initialising EDER-TSU fake track decoration module    ########################"+bcolors.ENDC)
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
   print(UF.TimeStamp(),bcolors.WARNING+'This option will erase all the previous Seed Creation jobs/results'+bcolors.ENDC)
   UserAnswer=input(bcolors.BOLD+"Would you like to continue (Y/N)? \n"+bcolors.ENDC)
   if UserAnswer=='N':
         Mode='C'
         print(UF.TimeStamp(),'OK, continuing then...')
   if UserAnswer=='Y':
      print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
      UF.EvalCleanUp(AFS_DIR, EOS_DIR, 'E6', ['E6_E6'], "SoftUsed == \"EDER-TSU-E6\"")
      print(UF.TimeStamp(),'Submitting jobs... ',bcolors.ENDC)
      for j in range(0,len(data)):
        for sj in range(0,int(data[j][2])):
            f_counter=0
            for f in range(0,1000):
              new_output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E5_E6_RawTracks_'+str(j)+'_'+str(sj)+'_'+str(f)+'.csv'
              if os.path.isfile(new_output_file_location):
               f_counter=f
            OptionHeader = [' --Set ',' --SubSet ', ' --EOS ', " --AFS ", " --Fraction ",' --MaxDOCA ', ' --MaxAngle ', ' --MaxSTG ', ' --MaxSLG ']
            OptionLine = [j,sj, EOS_DIR, AFS_DIR, '$1', MaxDOCA,MaxAngle,MaxSTG,MaxSLG]
            SHName = AFS_DIR + '/HTCondor/SH/SH_E6_' + str(j) + '_' + str(sj) + '.sh'
            SUBName = AFS_DIR + '/HTCondor/SUB/SUB_E6_' + str(j) + '_' + str(sj) + '.sub'
            MSGName = AFS_DIR + '/HTCondor/MSG/MSG_E6_' + str(j) + '_' + str(sj)
            ScriptName = AFS_DIR + '/Code/Utilities/E6_DecorateFakeTracks_Sub.py '
            UF.SubmitJobs2Condor(
                [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, f_counter + 1, 'EDER-TSU-E6', False,
                 False])
      print(UF.TimeStamp(), bcolors.OKGREEN+'All jobs have been submitted, please rerun this script with "--Mode C" in few hours'+bcolors.ENDC)
if Mode=='C':
   print(UF.TimeStamp(),'Checking results... ',bcolors.ENDC)
   test_file=EOS_DIR+'/EDER-TSU/Data/REC_SET/E6_FAKE_GENUINE_TRACKS.csv'
   if os.path.isfile(test_file):
       print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
       print(UF.TimeStamp(), bcolors.OKGREEN+"The process has been completed before, if you want to restart, please rerun with '--Mode R' option"+bcolors.ENDC)
       exit()
   bad_pop=[]
   print(UF.TimeStamp(),'Checking jobs... ',bcolors.ENDC)
   for j in range(0,len(data)):
        for sj in range(0,int(data[j][2])):
            for f in range (0,1000):
               new_output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E5_E6_RawTracks_'+str(j)+'_'+str(sj)+'_'+str(f)+'.csv'
               required_output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E6_E6_Dec_Fake_Tracks_'+str(j)+'_'+str(sj)+'_'+str(f)+'.csv'
               OptionHeader = [' --Set ', ' --SubSet ', ' --EOS ', " --AFS ", " --Fraction ",' --MaxDOCA ', ' --MaxAngle ', ' --MaxSTG ', ' --MaxSLG ']
               OptionLine = [j, sj, EOS_DIR, AFS_DIR, f,MaxDOCA,MaxAngle,MaxSTG,MaxSLG]
               SHName = AFS_DIR + '/HTCondor/SH/SH_E6_' + str(j) + '_' + str(sj) + '_' + str(f)+'.sh'
               SUBName = AFS_DIR + '/HTCondor/SUB/SUB_E6_' + str(j) + '_' + str(sj) + '_' + str(f)+'.sub'
               MSGName = AFS_DIR + '/HTCondor/MSG/MSG_E6_' + str(j) + '_' + str(sj)+'_' + str(f)
               ScriptName = AFS_DIR + '/Code/Utilities/E6_DecorateFakeTracks_Sub.py '
               job_details=[OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-E6', False,
                  False]
               if os.path.isfile(required_output_file_location)!=True and os.path.isfile(new_output_file_location):
                  bad_pop.append(job_details)
               else:
                continue
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
       print(UF.TimeStamp(),bcolors.OKGREEN+'All HTCondor Seed Creation jobs have finished'+bcolors.ENDC)
       created_file=False
       for j in range(0,len(data)):
           progress=round((float(j)/float(len(data)))*100,2)
           print(UF.TimeStamp(),'Compressing output, progress is ',progress,' %', end="\r", flush=True) #Progress display
           for sj in range(0,int(data[j][2])):
                for f in range (0,1000):
                   new_output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E5_E6_RawTracks_'+str(j)+'_'+str(sj)+'_'+str(f)+'.csv'
                   required_output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E6_E6_Dec_Fake_Tracks_'+str(j)+'_'+str(sj)+'_'+str(f)+'.csv'
                   if os.path.isfile(new_output_file_location):
                       if created_file==False:
                          track_data=pd.read_csv(required_output_file_location,usecols=['DOCA','Seg_Lon_Gap','Seg_Transv_Gap','angle'])
                          track_data['DOCA'] = track_data['DOCA'].astype(float)
                          track_data['DOCA']=(track_data['DOCA']/DOCABin)
                          track_data['DOCA']=track_data['DOCA'].apply(np.ceil)
                          track_data['DOCA']=track_data['DOCA']*DOCABin

                          track_data['Seg_Lon_Gap'] = track_data['Seg_Lon_Gap'].astype(float)
                          track_data['Seg_Lon_Gap']=(track_data['Seg_Lon_Gap']/SLGBin)
                          track_data['Seg_Lon_Gap']=track_data['Seg_Lon_Gap'].apply(np.ceil)
                          track_data['Seg_Lon_Gap']=track_data['Seg_Lon_Gap']*SLGBin

                          track_data['Seg_Transv_Gap'] = track_data['Seg_Transv_Gap'].astype(float)
                          track_data['Seg_Transv_Gap']=(track_data['Seg_Transv_Gap']/STGBin)
                          track_data['Seg_Transv_Gap']=track_data['Seg_Transv_Gap'].apply(np.ceil)
                          track_data['Seg_Transv_Gap']=track_data['Seg_Transv_Gap']*STGBin

                          track_data['angle'] = track_data['angle'].astype(float)
                          track_data['angle']=(track_data['angle']/AngleBin)
                          track_data['angle']=track_data['angle'].apply(np.ceil)
                          track_data['angle']=track_data['angle']*AngleBin
                          track_data['angle']=track_data['angle'].abs()
                          track_data['tracks']=1
                          track_data=track_data.groupby(['DOCA','Seg_Lon_Gap','Seg_Transv_Gap','angle'])['tracks'].sum().reset_index()
                          track_data['track_type']='Fake'
                          created_file=True
                       else:
                          new_track_data=pd.read_csv(required_output_file_location,usecols=['DOCA','Seg_Lon_Gap','Seg_Transv_Gap','angle'])
                          new_track_data['DOCA'] = new_track_data['DOCA'].astype(float)
                          new_track_data['DOCA']=(new_track_data['DOCA']/DOCABin)
                          new_track_data['DOCA']=new_track_data['DOCA'].apply(np.ceil)
                          new_track_data['DOCA']=new_track_data['DOCA']*DOCABin

                          new_track_data['Seg_Lon_Gap'] = new_track_data['Seg_Lon_Gap'].astype(float)
                          new_track_data['Seg_Lon_Gap']=(new_track_data['Seg_Lon_Gap']/SLGBin)
                          new_track_data['Seg_Lon_Gap']=new_track_data['Seg_Lon_Gap'].apply(np.ceil)
                          new_track_data['Seg_Lon_Gap']=new_track_data['Seg_Lon_Gap']*SLGBin

                          new_track_data['Seg_Transv_Gap'] = new_track_data['Seg_Transv_Gap'].astype(float)
                          new_track_data['Seg_Transv_Gap']=(new_track_data['Seg_Transv_Gap']/STGBin)
                          new_track_data['Seg_Transv_Gap']=new_track_data['Seg_Transv_Gap'].apply(np.ceil)
                          new_track_data['Seg_Transv_Gap']=new_track_data['Seg_Transv_Gap']*STGBin

                          new_track_data['angle']=new_track_data['angle'].abs()
                          new_track_data['angle'] = new_track_data['angle'].astype(float)
                          new_track_data['angle']=(new_track_data['angle']/AngleBin)
                          new_track_data['angle']=new_track_data['angle'].apply(np.ceil)
                          new_track_data['angle']=new_track_data['angle']*AngleBin

                          new_track_data['tracks']=1
                          new_track_data=new_track_data.groupby(['DOCA','Seg_Lon_Gap','Seg_Transv_Gap','angle'])['tracks'].sum().reset_index()
                          new_track_data['track_type']='Fake'

                          combo_track_data = [track_data,new_track_data]

                          track_data = pd.concat(combo_track_data)
                          track_data=track_data.groupby(['DOCA','Seg_Lon_Gap','Seg_Transv_Gap','angle','track_type'])['tracks'].sum().reset_index()
       print(UF.TimeStamp(),'Adding genuine data from', bcolors.OKBLUE+EOS_DIR+'/EDER-TSU/Data/TEST_SET/E3_TRUTH_TRACKS.csv'+bcolors.ENDC)
       truth_track_data=pd.read_csv(EOS_DIR+'/EDER-TSU/Data/TEST_SET/E3_TRUTH_TRACKS.csv',usecols=['DOCA','Seg_Lon_Gap','Seg_Transv_Gap','angle'])

       truth_track_data['DOCA'] = truth_track_data['DOCA'].astype(float)
       truth_track_data['DOCA']=(truth_track_data['DOCA']/DOCABin)
       truth_track_data['DOCA']=truth_track_data['DOCA'].apply(np.ceil)
       truth_track_data['DOCA']=truth_track_data['DOCA']*DOCABin

       truth_track_data['Seg_Lon_Gap'] = truth_track_data['Seg_Lon_Gap'].astype(float)
       truth_track_data['Seg_Lon_Gap']=(truth_track_data['Seg_Lon_Gap']/SLGBin)
       truth_track_data['Seg_Lon_Gap']=truth_track_data['Seg_Lon_Gap'].apply(np.ceil)
       truth_track_data['Seg_Lon_Gap']=truth_track_data['Seg_Lon_Gap']*SLGBin

       truth_track_data['Seg_Transv_Gap'] = truth_track_data['Seg_Transv_Gap'].astype(float)
       truth_track_data['Seg_Transv_Gap']=(truth_track_data['Seg_Transv_Gap']/STGBin)
       truth_track_data['Seg_Transv_Gap']=truth_track_data['Seg_Transv_Gap'].apply(np.ceil)
       truth_track_data['Seg_Transv_Gap']=truth_track_data['Seg_Transv_Gap']*STGBin

       truth_track_data['angle']=truth_track_data['angle'].abs()
       truth_track_data['angle'] = truth_track_data['angle'].astype(float)
       truth_track_data['angle']=(truth_track_data['angle']/AngleBin)
       truth_track_data['angle']=truth_track_data['angle'].apply(np.ceil)
       truth_track_data['angle']=truth_track_data['angle']*AngleBin

       truth_track_data['tracks']=1
       truth_track_data=truth_track_data.groupby(['DOCA','Seg_Lon_Gap','Seg_Transv_Gap','angle'])['tracks'].sum().reset_index()
       truth_track_data['track_type']='Genuine'
       combo_track_data = [track_data,truth_track_data]
       track_data = pd.concat(combo_track_data)
       output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E6_FAKE_GENUINE_TRACKS.csv'
       print(UF.TimeStamp(),'Writing combined data into', bcolors.OKBLUE+output_file_location+bcolors.ENDC)
       track_data.to_csv(output_file_location,index=False)
       print(bcolors.BOLD+'Would you like to delete filtered seeds data?'+bcolors.ENDC)
       UserAnswer=input(bcolors.BOLD+"Please, enter your option Y/N \n"+bcolors.ENDC)
       if UserAnswer=='Y':
            print(UF.TimeStamp(),'Cleaning up the work space... ',bcolors.ENDC)
            UF.EvalCleanUp(AFS_DIR, EOS_DIR, 'E6', ['E5_E6','E6_E6'], "SoftUsed == \"EDER-TSU-E6\"")
       print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
       print(UF.TimeStamp(), bcolors.OKGREEN+"Fake 2-track decoration is completed"+bcolors.ENDC)
       print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
       #End of the script



