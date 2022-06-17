#This simple script prepares the evaluation data for vertexing procedure

########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
parser = argparse.ArgumentParser(description='This script selects and prepares 2-segment seeds that have either a common track (True label) or do not have a common track (False label)')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/eos/user/a/aiuliano/public/sims_fedra/CH1_pot_03_02_20/b000001/b000001_withvertices.csv')
parser.add_argument('--Xmin',help="This option excludes data events that have tracks with hits x-coordinates that are above this value", default='0')
parser.add_argument('--Xmax',help="This option excludes data events that have tracks with hits x-coordinates that are below this value", default='0')
parser.add_argument('--Ymin',help="This option excludes data events that have tracks with hits y-coordinates that are above this value", default='0')
parser.add_argument('--Ymax',help="This option excludes data events that have tracks with hits y-coordinates that are below this value", default='0')
########################################     Main body functions    #########################################
args = parser.parse_args()
input_file_location=args.f
Xmin=float(args.Xmin)
Xmax=float(args.Xmax)
Ymin=float(args.Ymin)
Ymax=float(args.Ymax)
SliceData=max(Xmin,Xmax,Ymin,Ymax)>0
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
import Utility_Functions as UF
import Parameters as PM
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"####################  Initialising EDER-TSU training data preparation module         ###################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules have been imported successfully..."+bcolors.ENDC)
#fetching_test_data
print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
data=pd.read_csv(input_file_location,
            header=0,
            usecols=[PM.FEDRA_Track_ID,PM.FEDRA_Track_QUADRANT,PM.x,PM.y,PM.z,PM.MC_Track_ID,PM.MC_Event_ID,PM.MC_Mother_PDG])

total_rows=len(data.axes[0])
print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
print(UF.TimeStamp(),'Removing unreconstructed hits...')
data=data.dropna()
final_rows=len(data.axes[0])
print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
# convert data types

data[PM.MC_Event_ID] = data[PM.MC_Event_ID].astype(str)
data[PM.MC_Track_ID] = data[PM.MC_Track_ID].astype(str)
data[PM.FEDRA_Track_ID] = data[PM.FEDRA_Track_ID].astype(int)
data[PM.FEDRA_Track_ID] = data[PM.FEDRA_Track_ID].astype(str)
try:
    data[PM.FEDRA_Track_QUADRANT] = data[PM.FEDRA_Track_QUADRANT].astype(int)
except:
    print(UF.TimeStamp(), bcolors.WARNING+"Failed to convert quadrant to integer..."+bcolors.ENDC)
data[PM.FEDRA_Track_QUADRANT] = data[PM.FEDRA_Track_QUADRANT].astype(str)
data['FEDRA_Seg_ID'] = data[PM.FEDRA_Track_QUADRANT] + '-' + data[PM.FEDRA_Track_ID]
data['MC_Mother_Track_ID'] = data[PM.MC_Event_ID] + '-' + data[PM.MC_Track_ID]
data['MC_Mother_PDG'] =data[PM.MC_Mother_PDG]
# drop useless columns 
data=data.drop(columns=[PM.FEDRA_Track_ID, 
                PM.FEDRA_Track_QUADRANT,
                PM.MC_Event_ID,
                PM.MC_Track_ID,
                PM.MC_Mother_PDG
                ])
compress_data=data.drop([PM.x,PM.y,PM.z],axis=1)
compress_data['MC_Mother_Track_No']= compress_data['MC_Mother_Track_ID']
compress_data=compress_data.groupby(by=['FEDRA_Seg_ID','MC_Mother_Track_ID','MC_Mother_PDG'])['MC_Mother_Track_No'].count().reset_index()
compress_data=compress_data.sort_values(['FEDRA_Seg_ID','MC_Mother_Track_No'],ascending=[1,0])
# the majority of hits in FEDRA_Seg are from MC_Track 
compress_data.drop_duplicates(subset='FEDRA_Seg_ID',keep='first',inplace=True)
# compress_data gives the FEDRA_SEG - MC_TRACK correspondance
# each FEDRA_SEG has only one MC_TRACK_ID now
data=data.drop(['MC_Mother_Track_ID','MC_Mother_PDG'],axis=1)
compress_data=compress_data.drop(['MC_Mother_Track_No'],axis=1)
data=pd.merge(data, compress_data, how="left", on=['FEDRA_Seg_ID'])
if SliceData:
     print(UF.TimeStamp(),'Slicing the data...')
     ValidEvents=data.drop(data.index[(data[PM.x] > Xmax) | (data[PM.x] < Xmin) | (data[PM.y] > Ymax) | (data[PM.y] < Ymin)])
     ValidEvents.drop([PM.x,PM.y,PM.z,'MC_Mother_Track_ID'],axis=1,inplace=True)
     ValidEvents.drop_duplicates(subset='FEDRA_Seg_ID',keep='first',inplace=True)
     data=pd.merge(data, ValidEvents, how="inner", on=['FEDRA_Seg_ID'])
     final_rows=len(data.axes[0])
     print(UF.TimeStamp(),'The sliced data has ',final_rows,' hits')
output_file_location=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/bM1_TRACK_SEGMENTS.csv'
print(UF.TimeStamp(),'Removing tracks which have less than',PM.MinHitsTrack,'hits...')
track_no_data=data.groupby(['MC_Mother_Track_ID','FEDRA_Seg_ID','MC_Mother_PDG'],as_index=False).count()
track_no_data=track_no_data.drop([PM.y,PM.z],axis=1)
# how many FEDRA_Seg does each MC_Track have 
track_no_data=track_no_data.rename(columns={PM.x: "FEDRA_Seg_No"})
new_combined_data=pd.merge(data, track_no_data, how="left", on=['FEDRA_Seg_ID','MC_Mother_Track_ID','MC_Mother_PDG'])
new_combined_data = new_combined_data[new_combined_data.FEDRA_Seg_No >= PM.MinHitsTrack]
new_combined_data = new_combined_data.drop(["FEDRA_Seg_No"],axis=1)
# drop MC_Track_ID at last
new_combined_data = new_combined_data.drop(["MC_Mother_Track_ID"],axis=1)
new_combined_data=new_combined_data.sort_values(['FEDRA_Seg_ID',PM.x],ascending=[1,1])
grand_final_rows=len(new_combined_data.axes[0])
print(UF.TimeStamp(),'The cleaned data has ',grand_final_rows,' hits')
new_combined_data=new_combined_data.rename(columns={PM.x: "x"})
new_combined_data=new_combined_data.rename(columns={PM.y: "y"})
new_combined_data=new_combined_data.rename(columns={PM.z: "z"})
new_combined_data.to_csv(output_file_location,index=False)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"The track segment data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
exit()
