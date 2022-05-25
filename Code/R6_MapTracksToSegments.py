#This simple script maps the glued tracks to FEDRA reconstructed track segments

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

########################################     Main body functions    #########################################


#Loading Directory locations
csv_reader=open('../config',"r")
config = list(csv.reader(csv_reader))
for c in config:
    if c[0]=='AFS_DIR':
        AFS_DIR=c[1]
    if c[0]=='EOS_DIR':
        EOS_DIR=c[1]
csv_reader.close()
parser = argparse.ArgumentParser(description='This script prepares the reconstruction data for EDER-TSU track desegmentation routines by using the custom file with track resonstruction data')
parser.add_argument('--f',help="Please enter the full path to the file with track reconstruction", default='/eos/user/a/aiuliano/public/sims_fedra/CH1_pot_03_02_20/b000001/b000001_withvertices.csv')
parser.add_argument('--o',help="Please enter the full path to the output file with track reconstruction and gluing", default=EOS_DIR+'/EDER-TSU/Data/REC_SET/R6_REC_AND_GLUED_TRACKS.csv')
args = parser.parse_args()
input_file_location=args.f
output_file_location=args.o
input_map_file_location=EOS_DIR+'/EDER-TSU/Data/REC_SET/R5_GLUED_TRACKS.csv'
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import Utility_Functions as UF
import Parameters as PM
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"####################                  EDER-TSU data mapping module                   ###################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules have been imported successfully..."+bcolors.ENDC)
#fetching_test_data
print(UF.TimeStamp(),'Loading raw data from',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
data=pd.read_csv(input_file_location,header=0)
print(UF.TimeStamp(),'Loading mapped data from',bcolors.OKBLUE+input_map_file_location+bcolors.ENDC)
map_data=pd.read_csv(input_map_file_location,header=0)
total_rows=len(data.axes[0])
print(UF.TimeStamp(),'The raw data has ',total_rows,' hits')
print(UF.TimeStamp(),'Removing unreconstructed hits...')
data.dropna(subset=[PM.FEDRA_Track_ID],inplace=True)
final_rows=len(data.axes[0])
print(UF.TimeStamp(),'The cleaned data has ',final_rows,' hits')
data[PM.FEDRA_Track_ID] = data[PM.FEDRA_Track_ID].astype(int)
data[PM.FEDRA_Track_ID] = data[PM.FEDRA_Track_ID].astype(str)
data[PM.FEDRA_Track_QUADRANT] = data[PM.FEDRA_Track_QUADRANT].astype(int)
data[PM.FEDRA_Track_QUADRANT] = data[PM.FEDRA_Track_QUADRANT].astype(str)
data['FEDRA_Seg_ID'] = data[PM.FEDRA_Track_QUADRANT] + '-' + data[PM.FEDRA_Track_ID]
print(UF.TimeStamp(),'Mapping data...')
new_combined_data=pd.merge(data, map_data, how="outer", left_on=["FEDRA_Seg_ID"], right_on=['Old_Track_ID'])
# need to modify later
print(new_combined_data)
exit()
new_combined_data[PM.FEDRA_Track_QUADRANT] = np.where(new_combined_data['New_Track_Quarter'].isnull(), new_combined_data[PM.FEDRA_Track_QUADRANT], new_combined_data['New_Track_Quarter'])
new_combined_data[PM.FEDRA_Track_ID] = np.where(new_combined_data['New_Track_ID'].isnull(), new_combined_data[PM.FEDRA_Track_ID], new_combined_data['New_Track_ID'])
print(UF.TimeStamp(),'Mapping data...')
new_combined_data=new_combined_data.drop(['FEDRA_Seg_ID'],axis=1)
new_combined_data=new_combined_data.drop(['Old_Track_ID'],axis=1)
new_combined_data=new_combined_data.drop(['New_Track_Quarter'],axis=1)
new_combined_data=new_combined_data.drop(['New_Track_ID'],axis=1)
new_combined_data.drop_duplicates(subset=[PM.FEDRA_Track_QUADRANT,PM.FEDRA_Track_ID,PM.z],keep='first',inplace=True)
new_combined_data.to_csv(output_file_location,index=False)
print(UF.TimeStamp(), bcolors.OKGREEN+"The re-glued track data has been created successfully and written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
exit()
