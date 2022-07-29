#This simple script prepares 2-Track seeds for the initial CNN vertexing
# Part of EDER-TSU package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
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
parser = argparse.ArgumentParser(description='This script takes the output from the previous step and decorates the mwith track hit information that can be used to render the seed image. This script creates teraining and validation samples.')
parser.add_argument('--Mode',help="Running Mode: Reset(R)/Continue(C)", default='C')
parser.add_argument('--Samples',help="How many samples? Please enter the number or ALL if you want to use all data", default='ALL')
parser.add_argument('--ValidationSize',help="What is the proportion of Validation Images?", default='0.1')
parser.add_argument('--LabelMix',help="What is the desired proportion of genuine vertices in the training/validation sets", default='0.5')
parser.add_argument('--MotherPDGList', help="Target Mother PDGs", nargs='+', type=int, default='22')
######################################## Set variables  #############################################################
args = parser.parse_args()
Mode=args.Mode
MotherPDGList = args.MotherPDGList
if type(MotherPDGList)== int :
    MotherPDGList = [MotherPDGList]
MotherPDGList = str(MotherPDGList).strip('[').strip(']').replace(',','')





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
#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M6_TRACK_SEGMENTS.csv'
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################     Initialising EDER-TSU Image Generation module        ########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(UF.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
data=pd.read_csv(input_file_location,header=0,usecols=['FEDRA_Seg_ID'])
print(UF.TimeStamp(),'Analysing data... ',bcolors.ENDC)
data = data.drop_duplicates()

trackCnt = int(np.ceil(len(data)/MaxSegmentsPerJob))

if Mode=='R':
    print(UF.TimeStamp(),bcolors.WARNING+'Warning! You are running the script with the "Mode R" option which means that you want to create the seeds from the scratch'+bcolors.ENDC)
    print(UF.TimeStamp(),bcolors.WARNING+'This option will erase all the previous Seed Creation jobs/results'+bcolors.ENDC)
    UserAnswer=input(bcolors.BOLD+"Would you like to continue (Y/N)? \n"+bcolors.ENDC)
    if UserAnswer=='N':
        Mode='C'
        print(UF.TimeStamp(),'OK, continuing then...')

    if UserAnswer=='Y':
        print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
        UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'M7', ['M7_M7','M7_M8'], "SoftUsed == \"EDER-TSU-M7\"")
        print(UF.TimeStamp(),'Submitting jobs... ',bcolors.ENDC)
        OptionHeader = [' --Set ', ' --EOS ', " --AFS ", " --MotherPDGList ", " --MaxSegmentsPerJob "]
        OptionLine = ['$1', EOS_DIR, AFS_DIR, MotherPDGList, MaxSegmentsPerJob]
        SHName = AFS_DIR + '/HTCondor/SH/SH_M7.sh'
        SUBName = AFS_DIR + '/HTCondor/SUB/SUB_M7.sub'
        MSGName = AFS_DIR + '/HTCondor/MSG/MSG_M7' 
        ScriptName = AFS_DIR + '/Code/Utilities/M7_GenerateImages_Sub.py '
        UF.SubmitJobs2Condor(
        [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, trackCnt, 'EDER-TSU-M7', False,
                False])
        print(UF.TimeStamp(), bcolors.OKGREEN+'All jobs have been submitted, please rerun this script with "--Mode C" in few hours'+bcolors.ENDC)
if Mode=='C':

    ProcessStatus=1
    bad_pop=[]
    MaxJ=0
    for j in range(trackCnt):
        new_output_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M6_M7_RawTracks_'+str(j)+'.csv'
        required_output_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M7_RawImages_'+str(j)+'.pkl'       
        OptionHeader = [' --Set ', ' --EOS ', " --AFS "," --MotherPDGList"]
        OptionLine = ['$1', EOS_DIR, AFS_DIR, MotherPDGList]
        SHName = AFS_DIR + '/HTCondor/SH/SH_M7_'+str(j)+'.sh'
        SUBName = AFS_DIR + '/HTCondor/SUB/SUB_M7_'+str(j)+'.sub'
        MSGName = AFS_DIR + '/HTCondor/MSG/MSG_M7_'+str(j) 
        ScriptName = AFS_DIR + '/Code/Utilities/M7_GenerateImages_Sub.py '
        job_details = [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-M7', False,
                        False]
        if os.path.isfile(required_output_file_location)!=True and os.path.isfile(new_output_file_location):
            bad_pop.append(job_details)
        if os.path.isfile(required_output_file_location):
            MaxJ=j
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
        test_file=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M7_CondensedImages_'+str(MaxJ)+'.pkl'
        if os.path.isfile(test_file):
            print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
            print(UF.TimeStamp(), bcolors.OKGREEN+"The process has been ran before, continuing the image generation"+bcolors.ENDC)
            ProcessStatus=2
        test_file=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M7_SamplesCondensedImages_'+str(MaxJ)+'.pkl'
        if os.path.isfile(test_file):
            print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
            print(UF.TimeStamp(), bcolors.OKGREEN+"The process has been ran before and image sampling has begun"+bcolors.ENDC)
            ProcessStatus=3
        if ProcessStatus==1:
            UF.LogOperations(EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M7_Temp_Stats.csv','StartLog', [[0,0,0,0]])
            print(UF.TimeStamp(),bcolors.OKGREEN+'All HTCondor Seed Creation jobs have finished'+bcolors.ENDC)
            print(UF.TimeStamp(),'Collating the results...')
            for j in range(trackCnt):
                output_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M7_CondensedImages_'+str(j)+'.pkl'
                if os.path.isfile(output_file_location)==False:
                    Temp_Stats=UF.LogOperations(EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M7_Temp_Stats.csv','ReadLog', '_')
                    TotalImages=int(Temp_Stats[0][0])
                    Seeds0 =int(Temp_Stats[0][1])
                    Seeds1 =int(Temp_Stats[0][2])
                    Seeds2 =int(Temp_Stats[0][3])

                    new_output_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M6_M7_RawTracks_'+str(j)+'.csv'
                    required_output_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M7_RawImages_'+str(j)+'.pkl'
                    if os.path.isfile(required_output_file_location)!=True and os.path.isfile(new_output_file_location):
                        print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",required_output_file_location,'is missing, please restart the script with the option "--Mode R"'+bcolors.ENDC)
                    elif os.path.isfile(required_output_file_location):
                        base_data_file=open(required_output_file_location,'rb')
                        base_data=pickle.load(base_data_file)
                        base_data_file.close()
                    #try:
                    Records=len(base_data)
                    print(UF.TimeStamp(),'Set',str(j),'contains', Records, 'raw images',bcolors.ENDC)
                    base_data=list(set(base_data))
                    Records_After_Compression=len(base_data)
                    if Records>0:
                        Compression_Ratio=int((Records_After_Compression/Records)*100)
                    else:
                        CompressionRatio=0
                    TotalImages+=Records_After_Compression
                    Seeds0+=sum(1 for im in base_data if im.MC_truth_label == 0)
                    Seeds1+=sum(1 for im in base_data if im.MC_truth_label == 1)
                    Seeds2+=sum(1 for im in base_data if im.MC_truth_label == 2)


                    print(UF.TimeStamp(),'Set',str(j),'compression ratio is ', Compression_Ratio, ' %',bcolors.ENDC)
                    open_file = open(output_file_location, "wb")
                    pickle.dump(base_data, open_file)
                    open_file.close()
                    #except:
                        #continue
#                del new_data
                    UF.LogOperations(EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M7_Temp_Stats.csv','StartLog', [[TotalImages,Seeds0,Seeds1,Seeds2]])
            ProcessStatus=2


        ####Stage 2
        if ProcessStatus==2:
            print(UF.TimeStamp(),'Sampling the required number of seeds',bcolors.ENDC)
            Temp_Stats=UF.LogOperations(EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M7_Temp_Stats.csv','ReadLog', '_')
            TotalImages=int(Temp_Stats[0][0])
            Seeds0 =int(Temp_Stats[0][1])
            Seeds1 =int(Temp_Stats[0][2])
            TrueSeeds =int(Temp_Stats[0][3])
            if args.Samples=='ALL':
                if TrueSeeds<=(float(args.LabelMix)*TotalImages):
                    RequiredTrueSeeds=TrueSeeds
                    RequiredFakeSeeds=int(round((RequiredTrueSeeds/float(args.LabelMix))-RequiredTrueSeeds,0))
                else:
                    RequiredFakeSeeds=TotalImages-TrueSeeds
                    RequiredTrueSeeds=int(round((RequiredFakeSeeds/(1.0-float(args.LabelMix)))-RequiredFakeSeeds,0))
            else:
                NormalisedTotSamples=int(args.Samples)
                if TrueSeeds<=(float(args.LabelMix)*NormalisedTotSamples):
                    RequiredTrueSeeds=TrueSeeds
                    RequiredFakeSeeds=int(round((RequiredTrueSeeds/float(args.LabelMix))-RequiredTrueSeeds,0))
                    
                else:
                    RequiredFakeSeeds = NormalisedTotSamples*(1.0-float(args.LabelMix))
                    # signals
                    RequiredTrueSeeds=int(round((RequiredFakeSeeds/(1.0-float(args.LabelMix)))-RequiredFakeSeeds,0))
            if TrueSeeds==0:
                TrueSeedCorrection=0
            else:
                TrueSeedCorrection=RequiredTrueSeed/TrueSeeds
            FakeSeedCorrection=RequiredFakeSeeds/(TotalImages-TrueSeeds)
            for j in range(trackCnt):
                req_file=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M7_SamplesCondensedImages_'+str(j)+'.pkl'
                output_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M7_CondensedImages_'+str(j)+'.pkl'
                if os.path.isfile(req_file)==False and os.path.isfile(output_file_location):
                    progress=int( round( (float(j)/float(trackCnt)*100),0)  )
                    print(UF.TimeStamp(),"Sampling image from the collated data, progress is ",progress,' % of seeds generated',end="\r", flush=True)
                    base_data_file=open(output_file_location,'rb')
                    base_data=pickle.load(base_data_file)
                    base_data_file.close()

                    
                    Extracted0=[im for im in base_data if im.MC_truth_label ==0]
                    Extracted1=[im for im in base_data if im.MC_truth_label ==1]
                    Extracted2=[im for im in base_data if im.MC_truth_label ==2]
                    del base_data
                    gc.collect()
                    Extracted0=random.sample(Extracted0,int(round(FakeSeedsCorrection*len(Extracted0)/2,0)))
                    Extracted1=random.sample(Extracted1,int(round(FakeSeedsCorrection*len(Extracted1)/2,0)))
                    Extracted2=random.sample(Extracted1,int(round(TrueSeedCorrection*len(Extracted2),0)))

                    TotalData=[]

                    print(len(Extracted0))
                    print(len(Extracted1))
                    print(len(Extracted2))

                    TotalData=Extracted0+Extracted1+Extracted2
                    write_data_file=open(req_file,'wb')
                    pickle.dump(TotalData, write_data_file)
                    write_data_file.close()
                    del TotalData
                    del Extracted0
                    del Extracted1
                    del Extracted2
                    gc.collect()
                    ProcessStatus=3


        if ProcessStatus==3:
            TotalData=[]
            for j in range(trackCnt):
                output_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M7_SamplesCondensedImages_'+str(j)+'.pkl'
                if os.path.isfile(output_file_location):
                    progress=int( round( (float(j)/float(trackCnt)*100),0)  )
                    print(UF.TimeStamp(),"Re-sampling image from the collated data, progress is ",progress,' % of seeds generated',end="\r", flush=True)
                    base_data_file=open(output_file_location,'rb')
                    base_data=pickle.load(base_data_file)
                    base_data_file.close()
                    TotalData+=base_data
            del base_data
            gc.collect()
            ValidationSampleSize=int(round(min((len(TotalData)*float(args.ValidationSize)),PM.MaxValSampleSize),0))
            random.shuffle(TotalData)
            output_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M8_Validation_Set.pkl'
            ValExtracted_file = open(output_file_location, "wb")
            pickle.dump(TotalData[:ValidationSampleSize], ValExtracted_file)
            ValExtracted_file.close()
            TotalData=TotalData[ValidationSampleSize:]
            print(UF.TimeStamp(), bcolors.OKGREEN+"Validation Set has been saved at ",bcolors.OKBLUE+output_file_location+bcolors.ENDC,bcolors.OKGREEN+'file...'+bcolors.ENDC)
            No_Train_Files=int(math.ceil(len(TotalData)/PM.MaxTrainSampleSize))
            for SC in range(0,No_Train_Files):
                output_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M7_M8_Train_Set_'+str(SC+1)+'.pkl'
                OldExtracted_file = open(output_file_location, "wb")
                pickle.dump(TotalData[(SC*PM.MaxTrainSampleSize):min(len(TotalData),((SC+1)*PM.MaxTrainSampleSize))], OldExtracted_file)
                OldExtracted_file.close()
                print(UF.TimeStamp(), bcolors.OKGREEN+"Train Set", str(SC+1) ," has been saved at ",bcolors.OKBLUE+output_file_location+bcolors.ENDC,bcolors.OKGREEN+'file...'+bcolors.ENDC)
            UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'M7', ['M7_M7_SamplesCondensedImages','M7_M7_CondensedImages'], "SoftUsed == \"EDER-TSU-M7\"")
            print(bcolors.BOLD+'Would you like to delete track seeds data?'+bcolors.ENDC)
            UserAnswer=input(bcolors.BOLD+"Please, enter your option Y/N \n"+bcolors.ENDC)
            if UserAnswer=='Y':
                UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'M7', ['M6_M7','M7_M7'], "SoftUsed == \"EDER-TSU-M7\"")
            else:
                print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
                print(UF.TimeStamp(), bcolors.OKGREEN+"Training and Validation data has been created: you can render them now..."+bcolors.ENDC)
                print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
                exit()
#End of the script



