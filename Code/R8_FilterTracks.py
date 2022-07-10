#This simple script prepares 2-Track seeds for the initial CNN vertexing
# Part of EDER-TSU package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
import math #We use it for data manipulation
import numpy as np
import os
import pickle


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
parser = argparse.ArgumentParser(description='This script takes preselected 2-track seed candidates from previous step and refines them by applying additional cuts on the parameters such as DOCA, fiducial cute and distance to the possible vertex origin.')
parser.add_argument('--Mode',help="Running Mode: Reset(R)/Continue(C)", default='C')
parser.add_argument('--Log',help="Would you like to enable logging? Please make sure you have file generated by E3 process", default='N')
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
MaxDoca=PM.MaxDoca
MaxSLG=PM.MaxSLG
MaxSTG=PM.MaxSTG
MinAngle=PM.MinAngle
MaxAngle=PM.MaxAngle
 #The Separation bound is the maximum Euclidean distance that is allowed between hits in the beggining of Seed tracks.
MaxFitTracksPerJob=PM.MaxFitTracksPerJob
#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R7_R8_TRACK_HEADERS.csv'
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################     Initialising EDER-TSU Filter Tracks module           ########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(UF.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
data=pd.read_csv(input_file_location,header=0)
print(UF.TimeStamp(),'Analysing data... ',bcolors.ENDC)
NoJobs=np.ceil(len(data))/MaxFitTracksPerJob
print(len(data))
print(NoJobs)
exit()
if Mode=='R':
   print(UF.TimeStamp(),bcolors.WARNING+'Warning! You are running the script with the "Mode R" option which means that you want to vertex the seeds from the scratch'+bcolors.ENDC)
   print(UF.TimeStamp(),bcolors.WARNING+'This option will erase all the previous Seed vertexing jobs/results'+bcolors.ENDC)
   UserAnswer=input(bcolors.BOLD+"Would you like to continue (Y/N)? \n"+bcolors.ENDC)
   if UserAnswer=='N':
         Mode='C'
         print(UF.TimeStamp(),'OK, continuing then...')

   if UserAnswer=='Y':
      print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
      UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R8', ['R8_R8','R8_R9'], "SoftUsed == \"EDER-TSU-R8\"")
      print(UF.TimeStamp(),'Submitting jobs... ',bcolors.ENDC)
      OptionHeader = [' --Set ', ' --MaxFitTracksPerJobt ', ' --EOS ', " --AFS "]
            OptionLine = ['$1',MaxFitTracksPerJob, EOS_DIR, AFS_DIR]
            SHName = AFS_DIR + '/HTCondor/SH/SH_R8.sh'
            SUBName = AFS_DIR + '/HTCondor/SUB/SUB_R8.sub'
            MSGName = AFS_DIR + '/HTCondor/MSG/MSG_R8'
            ScriptName = AFS_DIR + '/Code/Utilities/R8_FilterTracks_Sub.py '
            UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, NoJobs, 'EDER-TSU-R8', False,False])
      print(UF.TimeStamp(), bcolors.OKGREEN+'All jobs have been submitted, please rerun this script with "--Mode C" in few hours'+bcolors.ENDC)
if Mode=='C':
   bad_pop=[]
   print(UF.TimeStamp(),'Checking jobs... ',bcolors.ENDC)
   for j in range(0,len(data)):
       for sj in range(0,int(data[j][2])):
           for f in range(0,1000):
              new_output_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R7_R8_RawTracks_'+str(j)+'_'+str(sj)+'_'+str(f)+'.csv'
              required_output_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R8_R8_FilteredTracks_'+str(j)+'_'+str(sj)+'_'+str(f)+'.pkl'
              OptionHeader = [' --Set ', ' --SubSet ', ' --Fraction ', ' --EOS ', " --AFS ", " --MaxSTG ", " --MaxSLG ", " --MaxDOCA ", " --MaxAngle "]
              OptionLine = [j, sj, '$1', EOS_DIR, AFS_DIR, MaxSTG, MaxSLG, MaxDoca, MaxAngle]
              SHName = AFS_DIR +'/HTCondor/SH/SH_R8_'+str(j)+'_'+str(sj) + '_'+ str(f) +'.sh'
              SUBName = AFS_DIR + '/HTCondor/SUB/SUB_R8_' + str(j) + '_' + str(sj) + '_'+ str(f) + '.sub'
              MSGName = AFS_DIR + '/HTCondor/MSG/MSG_R8_' + str(j) + '_' + str(sj) + '_'+ str(f)
              ScriptName = AFS_DIR + '/Code/Utilities/R8_FilterTracks_Sub.py '
              job_details=[OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-R8', False,
                 False]
              if os.path.isfile(required_output_file_location)!=True  and os.path.isfile(new_output_file_location):
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
       print(UF.TimeStamp(),bcolors.OKGREEN+'All HTCondor track Creation jobs have finished'+bcolors.ENDC)
       print(UF.TimeStamp(),'Collating the results...')
       trigger=False
       for j in range(0,len(data)):
        for sj in range(0,int(data[j][2])):
           for f in range(0,1000):
              new_output_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R7_R8_RawTracks_'+str(j)+'_'+str(sj)+'_'+str(f)+'.csv'
              required_output_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R8_R8_FilteredTracks_'+str(j)+'_'+str(sj)+'_'+str(f)+'.pkl'
              if os.path.isfile(required_output_file_location)!=True and os.path.isfile(new_output_file_location):
                 print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",required_output_file_location,'is missing, please restart the script with the option "--Mode R"'+bcolors.ENDC)
              elif os.path.isfile(required_output_file_location):
                 if trigger==False:
                    base_data_file=open(required_output_file_location,'rb')
                    base_data=pickle.load(base_data_file)
                    base_data_file.close()
                    trigger=True
                    continue
                 else:
                    new_data_file=open(required_output_file_location,'rb')
                    new_data=pickle.load(new_data_file)
                    new_data_file.close()
                    base_data+=new_data
        Records=len(base_data)
        print(UF.TimeStamp(),'Set',str(j),'contains', Records, 'selected track seed candidates for CNN fit...',bcolors.ENDC)

        base_data=list(set(base_data))
        Records_After_Compression=len(base_data)
        fractions=int(math.ceil(Records_After_Compression/MaxFitTracksPerJob))
        for f in range(0,fractions):
             output_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R8_R4_FilteredTracks_'+str(j)+'_'+str(f)+'.pkl'
             open_file = open(output_file_location, "wb")
             pickle.dump(base_data[(f*MaxFitTracksPerJob):min(Records_After_Compression,((f+1)*MaxFitTracksPerJob))], open_file)
             open_file.close()
        if Records>0:
              Compression_Ratio=int((Records_After_Compression/Records)*100)
        else:
              CompressionRatio=0
        print(UF.TimeStamp(),'Set',str(j+1),'compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)
       if args.Log=='Y':
         try:
             print(UF.TimeStamp(),'Initiating the logging...')
             eval_data_file=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E3_TRUTH_TRACKS.csv'
             eval_data=pd.read_csv(eval_data_file,header=0,usecols=['Segment_1','Segment_2'])
             eval_data["Track_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data['Segment_1'], eval_data['Segment_2'])]
             eval_data.drop(['Segment_1'],axis=1,inplace=True)
             eval_data.drop(['Segment_2'],axis=1,inplace=True)
             rec_no=0
             eval_no=0
             for j in range(0,len(data)):
                    for f in range(0,1000):
                      new_input_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R8_R4_FilteredTracks_'+str(j)+'_'+str(f)+'.pkl'
                      if os.path.isfile(new_input_file_location)==False:
                            break
                      else:
                         progress=round((float(j)/float(len(data)))*100,0)
                         print(UF.TimeStamp(),'progress is ',progress,' %', end="\r", flush=True) #Progress display
                         rec_data_file=open(new_input_file_location,'rb')
                         rec_data=pickle.load(rec_data_file)
                         rec_data_file.close()
                         rec_list=[]
                         for rd in rec_data:
                             rec_list.append([rd.SegmentHeader[0],rd.SegmentHeader[1]])
                         rec = pd.DataFrame(rec_list, columns = ['Segment_1','Segment_2'])
                         rec["Track_ID"]= ['-'.join(sorted(tup)) for tup in zip(rec['Segment_1'], rec['Segment_2'])]
                         rec.drop(['Segment_1'],axis=1,inplace=True)
                         rec.drop(['Segment_2'],axis=1,inplace=True)
                         rec_eval=pd.merge(eval_data, rec, how="inner", on=['Track_ID'])
                         eval_no+=len(rec_eval)
                         rec_no+=(len(rec)-len(rec_eval))           
             UF.LogOperations(EOS_DIR+'/EDER-TSU/Data/REC_SET/R_LOG.csv', 'UpdateLog', [[3,'DOCA and angle cuts',rec_no,eval_no,eval_no/(rec_no+eval_no),eval_no/len(eval_data)]])
             print(UF.TimeStamp(), bcolors.OKGREEN+"The log data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+EOS_DIR+'/EDER-TSU/Data/REC_SET/R_LOG.csv'+bcolors.ENDC)
         except:
             print(UF.TimeStamp(), bcolors.WARNING+'Log creation has failed'+bcolors.ENDC)
       print(UF.TimeStamp(),'Cleaning up the work space... ',bcolors.ENDC)
       UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R8', ['R7_R8','R8_R8'], "SoftUsed == \"EDER-TSU-R8\"")
       print(UF.TimeStamp(), bcolors.OKGREEN+"Track filtering is completed, you can perform CNN fit on them now..."+bcolors.ENDC)
       print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#End of the script



