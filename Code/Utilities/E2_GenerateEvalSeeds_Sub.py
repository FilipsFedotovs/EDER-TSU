#This simple script generates Truth 2-Track seeds for the reconstruction efficiency tests
# Part of EDER-TSU package
#Made by Filips Fedotovs
#Current version 1.0

########################################    Import libraries    #############################################
import argparse
import pandas as pd #We use Panda for a routine data processing
from pandas import DataFrame as df
import math #We use it for data manipulation
import gc  #Helps to clear memory

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--SubSet',help="Subset number", default='1')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--MaxTracks',help="A maximum number of track combinations that will be used in a particular HTCondor job for this script", default='20000')


######################################## Set variables  #############################################################
args = parser.parse_args()


Subset=int(args.SubSet)  #The subset helps to determine what portion of the track list is used to create the Seeds

########################################     Preset framework parameters    #########################################
MaxRecords=10000000 #A set parameter that helps to manage memory load of this script (Please do not exceed 10000000)
MaxTracks=int(args.MaxTracks)

#Loading Directory locations
EOS_DIR=args.EOS
AFS_DIR=args.AFS

import Utility_Functions as UF #This is where we keep routine utility functions

#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET//E1_TRACK_SEGMENTS.csv'
output_file_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E2_E2_RawTracks_'+str(Subset)+'.csv'
output_result_location=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E2_E2_RawTracks_'+str(Subset)+'_RES.csv'
print(UF.TimeStamp(), "Modules Have been imported successfully...")
print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)
data=pd.read_csv(input_file_location)



print(UF.TimeStamp(),'Creating seeds... ')
data_header = data.groupby('FEDRA_Seg_ID')['z'].min()  #Keeping only starting hits for the each track record (we do not require the full information about track in this script)
data_header=data_header.reset_index()
Records=len(data_header.axes[0])
print(UF.TimeStamp(),'There are total of ', Records, 'tracks in the data set')
Cut=math.ceil(MaxRecords/Records) #Even if use only a max of 20000 track on the right join we cannot perform the full outer join due to the memory limitations, we do it in a small 'cuts'
Steps=math.ceil(MaxTracks/Cut)  #Calculating number of cuts
data=pd.merge(data, data_header, how="inner", on=["FEDRA_Seg_ID","z"]) #Shrinking the Track data so just a star hit for each track is present.

#What section of data will we cut?
StartDataCut=Subset*MaxTracks
EndDataCut=(Subset+1)*MaxTracks

#Specifying the right join
r_data=data.rename(columns={"x": "r_x"})
#r_data.drop(r_data.index[r_data['z'] != PlateZ], inplace = True)
Records=len(r_data.axes[0])
print(UF.TimeStamp(),'There are  ', Records, 'tracks in the starting plate')
r_data=r_data.iloc[StartDataCut:min(EndDataCut,Records)]
Records=len(r_data.axes[0])
print(UF.TimeStamp(),'However we will only attempt  ', Records, 'tracks in the starting plate')
r_data=r_data.rename(columns={"y": "r_y"})
r_data=r_data.rename(columns={"z": "r_z"})
r_data=r_data.rename(columns={"FEDRA_Seg_ID": "Track_2"})
data=data.rename(columns={"FEDRA_Seg_ID": "Track_1"})

result_list=[]  #We will keep the result in list rather then Panda Dataframe to save memory

#Downcasting Panda Data frame data types in order to save memory
data["x"] = pd.to_numeric(data["x"],downcast='float')
data["y"] = pd.to_numeric(data["y"],downcast='float')
data["z"] = pd.to_numeric(data["z"],downcast='float')
r_data["r_x"] = pd.to_numeric(r_data["r_x"],downcast='float')
r_data["r_y"] = pd.to_numeric(r_data["r_y"],downcast='float')
r_data["r_z"] = pd.to_numeric(r_data["r_z"],downcast='float')

#Cleaning memory
del data_header
gc.collect()

#Creating csv file for the results
UF.LogOperations(output_file_location,'StartLog',result_list)

#This is where we start
for i in range(0,Steps):
  r_temp_data=r_data.iloc[0:min(Cut,len(r_data.axes[0]))] #Taking a small slice of the data
  r_data.drop(r_data.index[0:min(Cut,len(r_data.axes[0]))],inplace=True) #Shrinking the right join dataframe
  merged_data=pd.merge(data, r_temp_data, how="inner", on=['MC_Mother_Track_ID']) #Merging Tracks to check whether they could form a seed
  merged_data.drop(['y','z','x','r_x','r_y','r_z','MC_Mother_Track_ID'],axis=1,inplace=True) #Removing the information that we don't need anymore
  merged_data.drop(merged_data.index[merged_data['Track_1'] == merged_data['Track_2']], inplace = True) #Removing the cases where Seed tracks are the same
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
print(UF.TimeStamp(), "Evaluation Seed generation is finished...")
#End of the script



