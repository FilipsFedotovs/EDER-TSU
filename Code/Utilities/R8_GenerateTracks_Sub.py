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
Set=args.Set    #This is just used to name the output file
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
data=pd.read_csv(input_file_location)
print(data)
exit()


print(UF.TimeStamp(),'Creating segment combinations... ')
data_header = data.groupby('FEDRA_Seg_ID')['z'].min()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)

#What section of data will we cut?
StartDataCut=Subset*MaxSegments
EndDataCut=(Subset+1)*MaxSegments

#Specifying the right join

r_data=data.rename(columns={"FEDRA_Seg_ID": "Segment_2"})
r_data.drop(r_data.index[r_data['z'] != PlateZ], inplace = True)

Records=len(r_data.axes[0])
print(UF.TimeStamp(),'There are  ', Records, 'segments in the starting plate')

r_data=r_data.iloc[StartDataCut:min(EndDataCut,Records)]


Records=len(r_data.axes[0])
print(UF.TimeStamp(),'However we will only attempt  ', Records, 'track segments in the starting plate')
r_data.drop(['y'],axis=1,inplace=True)
r_data.drop(['x'],axis=1,inplace=True)
r_data.drop(['z'],axis=1,inplace=True)
data.drop(['e_y'],axis=1,inplace=True)
data.drop(['e_x'],axis=1,inplace=True)
data.drop(['e_z'],axis=1,inplace=True)
data.drop(data.index[data['z'] <= PlateZ], inplace = True)
data=data.rename(columns={"FEDRA_Seg_ID": "Segment_1"})

data['join_key'] = 'join_key'
r_data['join_key'] = 'join_key'

result_list=[]  #We will keep the result in list rather then Panda Dataframe to save memory

#Downcasting Panda Data frame data types in order to save memory
data["x"] = pd.to_numeric(data["x"],downcast='float')
data["y"] = pd.to_numeric(data["y"],downcast='float')
data["z"] = pd.to_numeric(data["z"],downcast='float')

r_data["e_x"] = pd.to_numeric(r_data["e_x"],downcast='float')
r_data["e_y"] = pd.to_numeric(r_data["e_y"],downcast='float')
r_data["e_z"] = pd.to_numeric(r_data["e_z"],downcast='float')

#Cleaning memory
del data_header
gc.collect()

#Creating csv file for the results
UF.LogOperations(output_file_location,'StartLog',result_list)
#This is where we start

for i in range(0,Steps):
  r_temp_data=r_data.iloc[0:min(Cut,len(r_data.axes[0]))] #Taking a small slice of the data
  r_data.drop(r_data.index[0:min(Cut,len(r_data.axes[0]))],inplace=True) #Shrinking the right join dataframe
  merged_data=pd.merge(data, r_temp_data, how="inner", on=['join_key']) #Merging Tracks to check whether they could form a seed
  merged_data['SLG']=merged_data['z']-merged_data['e_z'] #Calculating the Euclidean distance between Track start hits
  merged_data['STG']=np.sqrt((merged_data['x']-merged_data['e_x'])**2+((merged_data['y']-merged_data['e_y'])**2)) #Calculating the Euclidean distance between Track start hits
  merged_data['DynamicCut']=MaxSTG+(merged_data['SLG']*0.96)
  merged_data.drop(merged_data.index[merged_data['SLG'] < 0], inplace = True) #Dropping the Seeds that are too far apart
  merged_data.drop(merged_data.index[merged_data['SLG'] > MaxSLG], inplace = True) #Dropping the track segment combinations where the length of the gap between segments is too large
  merged_data.drop(merged_data.index[merged_data['STG'] > merged_data['DynamicCut']], inplace = True)
  merged_data.drop(['y','z','x','e_x','e_y','e_z','join_key','STG','SLG','DynamicCut'],axis=1,inplace=True) #Removing the information that we don't need anymore
  if merged_data.empty==False:
    merged_data.drop(merged_data.index[merged_data['Segment_1'] == merged_data['Segment_2']], inplace = True) #Removing the cases where Seed tracks are the same
    merged_list = merged_data.values.tolist() #Convirting the result to List data type
    result_list+=merged_list #Adding the result to the list
  if len(result_list)>=2000000: #Once the list gets too big we dump the results into csv to save memory
      UF.LogOperations(output_file_location,'UpdateLog',result_list) #Write to the csv
      #Clearing the memory
      del result_list
      result_list=[]
      gc.collect()
UF.LogOperations(output_file_location,'UpdateLog',result_list) #Writing the remaining data into the csv
UF.LogOperations(output_result_location,'StartLog',[])
print(UF.TimeStamp(), "Fake track generation is finished...")
#End of the script



