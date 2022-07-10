#This simple script prepares 2-Track seeds for the initial CNN vertexing
# Part of EDER-VIANN package
#Made by Filips Fedotovs
#Current version 1.0

########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
from pandas import DataFrame as df
import math #We use it for data manipulation
import gc  #Helps to clear memory
import numpy as np

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--Set',help="Set number", default='1')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--MaxSegments',help="A maximum number of track combinations that will be used in a particular HTCondor job for this script", default='20000')
######################################## Set variables  #############################################################
args = parser.parse_args()
Set=args.Set   #This is just used to name the output file
########################################     Preset framework parameters    #########################################
MaxSegments=int(args.MaxSegments)

#Loading Directory locations
EOS_DIR=args.EOS
AFS_DIR=args.AFS

#import sys
#sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import Utility_Functions as UF #This is where we keep routine utility functions

#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R7_TRACK_SEGMENTS.csv'
output_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R8_R8_RawTracks_'+Set+'.csv'
print(UF.TimeStamp(), "Modules Have been imported successfully...")
print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)


#What section of data will we cut?
StartDataCut=(int(Set)-1)*MaxSegments
EndDataCut=(int(Set))*MaxSegments
data=pd.read_csv(input_file_location)[StartDataCut:EndDataCut]

data=data.rename(columns={"FEDRA_Seg_ID": "Segment_1"})
data.drop(['y','z','x'],axis=1,inplace=True) #Removing the information that we don't need anymore
data = data.drop_duplicates()
result_list=data.values.tolist() #Convirting the result to List data type

UF.LogOperations(output_file_location,'StartLog',result_list) #Writing the remaining data into the csv
print(UF.TimeStamp(), "Fake track generation is finished...")
#End of the script



