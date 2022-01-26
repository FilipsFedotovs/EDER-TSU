#This simple merges 2-Track seeds to produce the final result
# Part of EDER-TSU package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import math #We use it for data manipulation
import gc  #Helps to clear memory
import numpy as np
import os
import pickle
import random


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
parser = argparse.ArgumentParser(description='This script takes vertex-fitted 2-track seed candidates from previous step and merges them if seeds have a common track')
parser.add_argument('--Acceptance',help="Minimum acceptance for the seed", default='0.5')
parser.add_argument('--DataCut',help="In how many chunks would you like to split data?", default='30')
parser.add_argument('--Mode',help="Restart (R) or Continue (C)", default='C')
######################################## Set variables  #############################################################
args = parser.parse_args()

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
from Utility_Functions import Track
import Parameters as PM #This is where we keep framework global parameters
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################     Initialising EDER-TSU Track gluing module            ########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
MaxTracksPerTrPool=PM.MaxTracksPerTrPool
if args.Mode=='R':
    UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R5', ['R5_R5'], "SoftUsed == \"EDER-TSU-R5\"")
    Acceptance=float(args.Acceptance)
if args.Mode=='R':
    print(UF.TimeStamp(), "Starting the script from the scratch")
    input_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R4_R5_CNN_Fit_Track_Segments.pkl'
    print(UF.TimeStamp(), "Loading fit track seeds from the file",bcolors.OKBLUE+input_file_location+bcolors.ENDC)
    data_file=open(input_file_location,'rb')
    base_data=pickle.load(data_file)
    data_file.close()
    print(UF.TimeStamp(), bcolors.OKGREEN+"Loading is successful, there are "+str(len(base_data))+" fit seeds..."+bcolors.ENDC)
    print(UF.TimeStamp(), "Stripping off the seeds with low acceptance...")
    base_data=[tr for tr in base_data if tr.Track_CNN_Fit >= Acceptance]
    print(UF.TimeStamp(), bcolors.OKGREEN+"The refining was successful, "+str(len(base_data))+" track seeds remain..."+bcolors.ENDC)
    output_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R4_R5_CNN_Fit_Track_Segments_Refined.pkl'
    open_file = open(output_file_location, "wb")
    pickle.dump(base_data, open_file)
    open_file.close()
    no_iter=int(math.ceil(float(len(base_data)/float(MaxTracksPerTrPool))))
    print(UF.TimeStamp(), "Submitting jobs to HTCondor...")
    OptionHeader = [' --f ',' --Set ', ' --EOS ', " --AFS ", ' --MaxPoolTracks ']
    OptionLine = [output_file_location,'$1', EOS_DIR, AFS_DIR, MaxTracksPerTrPool]
    SHName = AFS_DIR + '/HTCondor/SH/SH_R5.sh'
    SUBName = AFS_DIR + '/HTCondor/SUB/SUB_R5.sub'
    MSGName = AFS_DIR + '/HTCondor/MSG/MSG_R5'
    ScriptName = AFS_DIR + '/Code/Utilities/R5_GlueTrackSegments_Sub.py '
    UF.SubmitJobs2Condor(
        [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, no_iter, 'EDER-TSU-R5', True,
         False])
    print(UF.TimeStamp(), bcolors.OKGREEN + "All ",no_iter," jobs have been submitted to HTCondor successfully..." + bcolors.ENDC)
if args.Mode=='C':
    print(UF.TimeStamp(), "Continuing the merging of seeds into multi-track vertices")
    input_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R5_R5_Glued_Segments_Temp.pkl'
    if os.path.isfile(input_file_location)==False:
        input_file_location = EOS_DIR+'/EDER-TSU/Data/REC_SET/R4_R5_CNN_Fit_Track_Segments_Refined.pkl'
    if os.path.isfile(input_file_location) == False:
        print(UF.TimeStamp(), bcolors.FAIL + "Critical fail, no",input_file_location,"file is present, please restart with the option --Mode R" + bcolors.ENDC)
        exit()
    print(UF.TimeStamp(), "Checking jobs")
    data_file = open(input_file_location, 'rb')
    base_data = pickle.load(data_file)
    data_file.close()
    original_data_seeds=len(base_data)
    del base_data
    no_iter = int(math.ceil(float(original_data_seeds / float(MaxTracksPerTrPool))))
    print(UF.TimeStamp(), "Submitting jobs to HTCondor...")
    bad_pop=[]
    for i in range(no_iter):
        required_file_location = EOS_DIR + '/EDER-TSU/Data/REC_SET/R5_R5_Temp_Glued_Segments_' + str(i) + '.pkl'
        if os.path.isfile(required_file_location) == False:
            OptionHeader = [' --f ', ' --Set ', ' --EOS ', " --AFS ", ' --MaxPoolTracks ']
            OptionLine = [input_file_location, i, EOS_DIR, AFS_DIR, MaxTracksPerTrPool]
            SHName = AFS_DIR + '/HTCondor/SH/SH_R5_'+str(i)+'.sh'
            SUBName = AFS_DIR + '/HTCondor/SUB/SUB_R5_'+str(i)+'.sub'
            MSGName = AFS_DIR + '/HTCondor/MSG/MSG_R5_'+str(i)
            ScriptName = AFS_DIR + '/Code/Utilities/R5_GlueTrackSegments_Sub.py '
            bad_pop.append(
                [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-R5', True,
                 False])
    if len(bad_pop)>0:
        print(UF.TimeStamp(), bcolors.WARNING + 'Warning, there are still', len(bad_pop),
              'HTCondor jobs remaining' + bcolors.ENDC)
        print(bcolors.BOLD + 'If you would like to wait and try again later please enter W' + bcolors.ENDC)
        print(bcolors.BOLD + 'If you would like to resubmit please enter R' + bcolors.ENDC)
        UserAnswer = input(bcolors.BOLD + "Please, enter your option\n" + bcolors.ENDC)
        if UserAnswer == 'W':
            print(UF.TimeStamp(), 'OK, exiting now then')
            exit()
        if UserAnswer == 'R':
            for bp in bad_pop:
                UF.SubmitJobs2Condor(bp)
            print(UF.TimeStamp(), bcolors.OKGREEN + "All jobs have been resubmitted" + bcolors.ENDC)
            print(bcolors.BOLD + "Please check them in few hours" + bcolors.ENDC)
            exit()
    else:
        print(UF.TimeStamp(), bcolors.OKGREEN + 'All HTCondor Seed Creation jobs have finished' + bcolors.ENDC)
        print(UF.TimeStamp(), 'Collating the results...')
        VertexPool=[]
        for i in range(no_iter):
            progress = round((float(i) / float(no_iter)) * 100, 2)
            print(UF.TimeStamp(), 'progress is ', progress, ' %', end="\r", flush=True)
            required_file_location = EOS_DIR + '/EDER-TSU/Data/REC_SET/R5_R5_Temp_Glued_Segments_' + str(i) + '.pkl'
            data_file = open(required_file_location, 'rb')
            NewData=pickle.load(data_file)
            data_file.close()
            VertexPool+=NewData
        print(UF.TimeStamp(), 'As a result of the previous operation',str(original_data_seeds),'track segment seeds were merged into',str(len(VertexPool)),'tracks...')
        comp_ratio = round((float(len(VertexPool)) / float(original_data_seeds)) * 100, 2)
        print(UF.TimeStamp(), 'The compression ratio is',comp_ratio, '%...')
        print(bcolors.BOLD + 'If you would like to reiterate the merging operation one more time please enter C' + bcolors.ENDC)
        print(bcolors.BOLD + 'If you would like to merge all remaining vertices then enter F' + bcolors.ENDC)
        UserAnswer = input(bcolors.BOLD + "Please, enter your option\n" + bcolors.ENDC)
        if UserAnswer == 'C':
            print(UF.TimeStamp(), 'OK, shuffling and preparing the set...')
            random.shuffle(VertexPool)
            output_file_location = EOS_DIR+'/EDER-TSU/Data/REC_SET/R5_R5_Glued_Segments_Temp.pkl'
            open_file = open(output_file_location, "wb")
            pickle.dump(VertexPool, open_file)
            open_file.close()
            print(UF.TimeStamp(), "Saving the temporarily file into", bcolors.OKBLUE + output_file_location + bcolors.ENDC)
            UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R5', ['R5_R5_Temp'], "SoftUsed == \"EDER-TSU-R5\"")
            no_iter = int(math.ceil(float(len(VertexPool) / float(MaxTracksPerTrPool))))
            print(UF.TimeStamp(), "Submitting jobs to HTCondor...")
            OptionHeader = [' --f ', ' --Set ', ' --EOS ', " --AFS ", ' --MaxPoolTracks ']
            OptionLine = [output_file_location, '$1', EOS_DIR, AFS_DIR, MaxTracksPerTrPool]
            SHName = AFS_DIR + '/HTCondor/SH/SH_R5.sh'
            SUBName = AFS_DIR + '/HTCondor/SUB/SUB_R5.sub'
            MSGName = AFS_DIR + '/HTCondor/MSG/MSG_R5'
            ScriptName = AFS_DIR + '/Code/Utilities/R5_GlueTrackSegments_Sub.py '
            UF.SubmitJobs2Condor(
                [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, no_iter, 'EDER-TSU-R5', False,
                 False])
            print(UF.TimeStamp(), bcolors.OKGREEN + "All ", no_iter,
                  " jobs have been submitted to HTCondor successfully..." + bcolors.ENDC)
            exit()
        if UserAnswer == 'F':
                 print(UF.TimeStamp(), 'Ok starting the final merging of the remained vertices')

                 InitialDataLength=len(VertexPool)
                 SeedCounter=0
                 SeedCounterContinue=True
                 while SeedCounterContinue:
                     if SeedCounter==len(VertexPool):
                                       SeedCounterContinue=False
                                       break
                     progress=round((float(SeedCounter)/float(len(VertexPool)))*100,0)
                     print(UF.TimeStamp(),'progress is ',progress,' %', end="\r", flush=True) #Progress display
                     SubjectSeed=VertexPool[SeedCounter]


                     for ObjectSeed in VertexPool[SeedCounter+1:]:
                              if (('3-349540' in SubjectSeed.SegmentHeader) or ('3-404564' in SubjectSeed.SegmentHeader) or ('3-440684' in SubjectSeed.SegmentHeader) or ('3-349540' in ObjectSeed.SegmentHeader) or ('3-404564' in ObjectSeed.SegmentHeader) or ('3-440684' in ObjectSeed.SegmentHeader)):
                                 print(ObjectSeed.SegmentHeader)
                                 print(ObjectSeed.SegmentHits)
                                 print(ObjectSeed.TR_CNN_FIT)
                                 print(SubjectSeed.SegmentHeader)
                                 print(SubjectSeed.SegmentHits)
                                 print(SubjectSeed.TR_CNN_FIT)
                                 if SubjectSeed.InjectTrack(ObjectSeed):
                                             VertexPool.pop(VertexPool.index(ObjectSeed))
                     SeedCounter+=1
                 print(str(InitialDataLength), "vertices from different files were merged into", str(len(VertexPool)), 'vertices with higher multiplicity...')
                 for v in range(0,len(VertexPool)):
                     VertexPool[v].AssignCNNTrId(v)
                 output_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R5_GLUED_TRACKS.pkl'
                 output_csv_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R5_GLUED_TRACKS.csv'
                 csv_out=[['Old_Track_ID','New_Track_Quarter','New_Track_ID']]
                 for Tr in VertexPool:
                     for TH in Tr.SegmentHeader:
                         csv_out.append([TH,'TSU',TR_CNN_ID])
                 print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
                 print(UF.TimeStamp(), "Saving the results into the file",bcolors.OKBLUE+output_csv_location+bcolors.ENDC)
                 UF.LogOperations(output_csv_location,'StartLog', csv_out)
                 print(UF.TimeStamp(), "Saving the results into the file",bcolors.OKBLUE+output_file_location+bcolors.ENDC)
                 open_file = open(output_file_location, "wb")
                 pickle.dump(VertexPool, open_file)
                 open_file.close()
                 UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R5', ['R5_R5'], "SoftUsed == \"EDER-TSU-R5\"")
                 print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
                 print(UF.TimeStamp(),bcolors.OKGREEN+'The vertex merging has been completed..'+bcolors.ENDC)
                 print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
