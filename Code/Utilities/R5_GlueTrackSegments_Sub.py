#This simple merges 2-Track verCtices to produce the final result
# Part of EDER-VIANN package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################

import argparse
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
parser = argparse.ArgumentParser(description='This script takes vertex-fitted 2-track seed candidates from previous step and merges them if seeds have a common track')
parser.add_argument('--f',help="File with seeds", default='')
parser.add_argument('--Set',help="What set?", default='1')
parser.add_argument('--AFS',help="AFS storage", default='')
parser.add_argument('--EOS',help="EOS storage", default='')
parser.add_argument('--MaxPoolSeeds',help="How many seeds?", default='20000')
######################################## Set variables  #############################################################
args = parser.parse_args()
AFS_DIR=args.AFS
EOS_DIR=args.EOS
Set=int(args.Set)
input_file_location=args.f
MaxPoolSeeds=int(args.MaxPoolSeeds)

#Loading Directory locations

import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import Utility_Functions as UF #This is where we keep routine utility functions
from Utility_Functions import Seed
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(UF.TimeStamp(), "Loading vertexed seeds from the file",bcolors.OKBLUE+input_file_location+bcolors.ENDC)
data_file=open(input_file_location,'rb')
base_data=pickle.load(data_file)
data_file.close()
print(UF.TimeStamp(), bcolors.OKGREEN+"Loading is successful, there are total of "+str(len(base_data))+" vertexed seeds..."+bcolors.ENDC)
base_data=base_data[(Set*MaxPoolSeeds):min(((Set+1)*MaxPoolSeeds),len(base_data))]
print(UF.TimeStamp(), bcolors.OKGREEN+"Out of these only "+str(len(base_data))+" vertexed seeds will be considered here..."+bcolors.ENDC)
output_file_location=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R6_R6_Temp_Merged_Seeds_'+str(Set)+'.pkl'
print(UF.TimeStamp(), "Initiating the seed merging...")
InitialDataLength=len(base_data)
SeedCounter=0
SeedCounterContinue=True
while SeedCounterContinue:
    if SeedCounter>=len(base_data):
       SeedCounterContinue=False
       break
    progress=round(float(SeedCounter)/float(len(base_data))*100,0)
    print(UF.TimeStamp(),'progress is ',progress,' %', end="\r", flush=True) #Progress display
    SubjectSeed=base_data[SeedCounter]
    for ObjectSeed in base_data[SeedCounter+1:]:
        if SubjectSeed.InjectSeed(ObjectSeed):
           base_data.pop(base_data.index(ObjectSeed))
    SeedCounter+=1
print(str(InitialDataLength), "2-track vertices were merged into", str(len(base_data)), 'vertices with higher multiplicity...')
open_file = open(output_file_location, "wb")
pickle.dump(base_data, open_file)
open_file.close()
print(UF.TimeStamp(), "Saving the results into the file",bcolors.OKBLUE+output_file_location+bcolors.ENDC)

