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
parser = argparse.ArgumentParser(description='This script takes refined 2-track seed candidates from previous step and perfromes a fit by using pre-trained CNN model.')
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
acceptance=PM.acceptance
MaxX=PM.MaxX
MaxY=PM.MaxY
MaxZ=PM.MaxZ
ModelName=PM.Pre_CNN_Model_Name
 #The Separation bound is the maximum Euclidean distance that is allowed between hits in the beggining of Seed tracks.
MaxSegmentsPerJob = PM.MaxSegmentsPerJob
MaxTracksPerJob = PM.MaxTracksPerJob
#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R1_TRACK_SEGMENTS.csv'
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################     Initialising EDER-TSU Vertexing module               ########################"+bcolors.ENDC)
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
data = data.values.tolist() #Convirting the result to List data type
if Mode=='R':
   print(UF.TimeStamp(),bcolors.WARNING+'Warning! You are running the script with the "Mode R" option which means that you want to glue the tracks from the scratch'+bcolors.ENDC)
   print(UF.TimeStamp(),bcolors.WARNING+'This option will erase all the previous track seed fit jobs/results'+bcolors.ENDC)
   UserAnswer=input(bcolors.BOLD+"Would you like to continue (Y/N)? \n"+bcolors.ENDC)
   if UserAnswer=='N':
         Mode='C'
         print(UF.TimeStamp(),'OK, continuing then...')

   if UserAnswer=='Y':
      print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
      UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R4', ['R4_R4','R4_REC'], "SoftUsed == \"EDER-TSU-R4\"")
      print(UF.TimeStamp(),'Submitting jobs... ',bcolors.ENDC)
      for j in range(0,len(data)):
            f_counter=0
            for f in range(0,1000):
             new_output_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R3_R4_FilteredTracks_'+str(j)+'_'+str(f)+'.pkl'
             if os.path.isfile(new_output_file_location):
              f_counter=f
            OptionHeader = [' --Set ', ' --Fraction ', ' --EOS ', " --AFS ", " --resolution ", " --acceptance "," --MaxX ", " --MaxY ", " --MaxZ ", " --ModelName "]
            OptionLine = [(j), '$1', EOS_DIR, AFS_DIR, resolution,acceptance,MaxX,MaxY,MaxZ,ModelName]
            SHName = AFS_DIR + '/HTCondor/SH/SH_R4_' + str(j) + '.sh'
            SUBName = AFS_DIR + '/HTCondor/SUB/SUB_R4_' + str(j) + '.sub'
            MSGName = AFS_DIR + '/HTCondor/MSG/MSG_R4_' + str(j)
            ScriptName = AFS_DIR + '/Code/Utilities/R4_FitTracks_Sub.py '
            UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, f_counter+1, 'EDER-TSU-R4', False,False])
      print(UF.TimeStamp(), bcolors.OKGREEN+'All jobs have been submitted, please rerun this script with "--Mode C" in few hours'+bcolors.ENDC)
if Mode=='C':
   bad_pop=[]
   print(UF.TimeStamp(),'Checking jobs... ',bcolors.ENDC)
   for j in range(0,len(data)):
           for f in range(0,1000):
              new_output_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R3_R4_FilteredTracks_'+str(j)+'_'+str(f)+'.pkl'
              required_output_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R4_R4_CNN_Fit_Tracks_'+str(j)+'_'+str(f)+'.pkl'
              OptionHeader = [' --Set ', ' --Fraction ', ' --EOS ', " --AFS ", " --resolution ", " --acceptance ",
                              " --MaxX ", " --MaxY ", " --MaxZ ", " --ModelName "]
              OptionLine = [(j), f, EOS_DIR, AFS_DIR, resolution, acceptance, MaxX, MaxY, MaxZ, ModelName]
              SHName = AFS_DIR + '/HTCondor/SH/SH_R4_' + str(j) + '_'+str(f)+'.sh'
              SUBName = AFS_DIR + '/HTCondor/SUB/SUB_R4_' + str(j) +'_'+str(f)+'.sub'
              MSGName = AFS_DIR + '/HTCondor/MSG/MSG_R4_' + str(j) +'_'+str(f)
              ScriptName = AFS_DIR + '/Code/Utilities/R4_FitTracks_Sub.py '
              
              job_details = [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-R4', False,
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

       print(UF.TimeStamp(),bcolors.OKGREEN+'All HTCondor Track Creation jobs have finished'+bcolors.ENDC)
       print(UF.TimeStamp(),'Collating the results...')
       for j in range(0,len(data)):
           for f in range(0,1000):
              new_output_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R3_R4_FilteredTracks_'+str(j)+'_'+str(f)+'.pkl'
              required_output_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R4_R4_CNN_Fit_Tracks_'+str(j)+'_'+str(f)+'.pkl'
              progress=round((float(j)/float(len(data)))*100,2)
              print(UF.TimeStamp(),'progress is ',progress,' %', end="\r", flush=True) #Progress display
              if os.path.isfile(required_output_file_location)!=True and os.path.isfile(new_output_file_location):
                 print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",required_output_file_location,'is missing, please restart the script with the option "--Mode R"'+bcolors.ENDC)
              elif os.path.isfile(required_output_file_location):
                 if (j)==(f)==0:
                    base_data_file=open(required_output_file_location,'rb')
                    base_data=pickle.load(base_data_file)
                    base_data_file.close()
                 else:
                    new_data_file=open(required_output_file_location,'rb')
                    new_data=pickle.load(new_data_file)
                    new_data_file.close()
                    base_data+=new_data

       Records=len(base_data)
       print(UF.TimeStamp(),'Final reconstructed set contains', Records, '2-segment glued tracks',bcolors.ENDC)
       output_file_eval_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R4_E4_CNN_FIT_TRACK_SEGMENTS.csv'
       output_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R4_R5_CNN_Fit_Track_Segments.pkl'
       base_data=list(set(base_data))
       Records_After_Compression=len(base_data)
       if Records>0:
              Compression_Ratio=int((Records_After_Compression/Records)*100)
       else:
              CompressionRatio=0
       print(UF.TimeStamp(),'Final reconstructed set compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)
       print(UF.TimeStamp(), 'Saving the object file... ')
       base_data_file=open(output_file_location,'wb')
       pickle.dump(base_data,base_data_file)
       base_data_file.close()
       eval_tracks=[]
       eval_tracks.append(['Segment_1','Segment_2','Track_CNN_Fit'])
       print(UF.TimeStamp(), 'Saving the csv file... ')
       for sd in base_data:
           eval_tracks.append([sd.SegmentHeader[0],sd.SegmentHeader[1],sd.Track_CNN_Fit])
       del base_data
       UF.LogOperations(output_file_eval_location,'StartLog', eval_tracks)
       print(UF.TimeStamp(),'Cleaning up the work space... ',bcolors.ENDC)
       UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R4', ['R4_R4'], "SoftUsed == \"EDER-TSU-R4\"")
       print(bcolors.BOLD+'Would you like to delete filtered track seed data?'+bcolors.ENDC)
       UserAnswer=input(bcolors.BOLD+"Please, enter your option Y/N \n"+bcolors.ENDC)
       if UserAnswer=='Y':
           UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R4', ['R3_R4'], "SoftUsed == \"EDER-TSU-R4\"")
       else:
        print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
        print(UF.TimeStamp(), bcolors.OKGREEN+"2-segment gluing is completed"+bcolors.ENDC)
        print(UF.TimeStamp(), bcolors.OKGREEN+"The results are saved in"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC, 'and in '+bcolors.ENDC, bcolors.OKBLUE+output_file_eval_location+bcolors.ENDC)
        print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#End of the script



