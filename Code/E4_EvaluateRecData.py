#This simple script prepares 2-Track seeds for the initial CNN vertexing
# Part of EDER-VIANN package
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

class bcolors:   #We use it for the interface
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'






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
from Utility_Functions import Track
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script compares the ouput of the previous step with the output of EDER-VIANN reconstructed data to calculate reconstruction perfromance.')
parser.add_argument('--Acceptance',help="What is the mininimum acceptance", default='0.5')
parser.add_argument('--LinkAcceptance',help="What is the mininimum acceptance", default='N')
parser.add_argument('--sf',help="Please choose the input file", default=EOS_DIR+'/EDER-VIANN/Data/REC_SET/R5_E4_LINK_FIT_SEEDS.csv')
parser.add_argument('--of',help="Please choose the evaluation file (has to match the same geometrical domain and type of the track as the subject", default=EOS_DIR+'/EDER-VIANN/Data/TEST_SET/E3_TRUTH_SEEDS.csv')
parser.add_argument('--TypeOfAnalysis',help="What type of analysis? Test CNN: 'CNN'. Test FEDRA Track Reconstruction quality: 'FEDRA'. All: 'All", default='CNN')
parser.add_argument('--rf',help="Please choose the input file", default=EOS_DIR+'/EDER-TSU/Data/REC_SET/R1_TRACK_SEGMENTS.csv')
parser.add_argument('--ef',help="Please choose the input evaluation track file (has to match the same geometrical domain and type of the track as the subject", default=EOS_DIR+'/EDER-TSU/Data/TEST_SET/E1_TRACK_SEGMENTS.csv')
######################################## Set variables  #############################################################
args = parser.parse_args()
acceptance=float(args.Acceptance)
if args.LinkAcceptance!='N':
   link_acceptance=float(args.LinkAcceptance)
########################################     Preset framework parameters    #########################################
 #The Separation bound is the maximum Euclidean distance that is allowed between hits in the beggining of Seed tracks.
MaxTracksPerJob = PM.MaxTracksPerJob
#Specifying the full path to input/output files
input_rec_file_location=args.rf
input_eval_file_location=args.ef
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################     Initialising EDER-TSU Evaluation module              ########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
if args.TypeOfAnalysis == 'ALL' or args.TypeOfAnalysis == 'CNN':
    print(UF.TimeStamp(),'Analysing evaluation data... ',bcolors.ENDC)
    if os.path.isfile(input_eval_file_location)!=True:
                     print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",input_eval_file_location,'is missing, please restart the evaluation sequence scripts'+bcolors.ENDC)
    eval_data=pd.read_csv(input_eval_file_location,header=0,usecols=['Track_1','Track_2'])
    eval_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(eval_data['Track_1'], eval_data['Track_2'])]
    eval_data.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
    eval_data.drop(eval_data.index[eval_data['Track_1'] == eval_data['Track_2']], inplace = True)
    eval_data.drop(["Track_1"],axis=1,inplace=True)
    eval_data.drop(["Track_2"],axis=1,inplace=True)
    TotalMCVertices=len(eval_data.axes[0])
    TotalRecVertices=0
    MatchedVertices=0
    FakeVertices=0
    print(UF.TimeStamp(),'Evaluating reconstructed set ',bcolors.ENDC)
    test_file_location=args.sf
    rec_file_location=args.vf
    if os.path.isfile(test_file_location)!=True:
        print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",test_file_location,'is missing, please restart the reconstruction sequence scripts'+bcolors.ENDC)
        exit()
    if args.LinkAcceptance!='N':
          test_data=pd.read_csv(test_file_location,header=0,usecols=['Track_1','Track_2','Seed_CNN_Fit', 'Seed_Link_Fit'])
    else:
        test_data = pd.read_csv(test_file_location, header=0,
                                usecols=['Track_1', 'Track_2', 'Seed_CNN_Fit'])
    test_data["Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(test_data['Track_1'], test_data['Track_2'])]
    test_data.drop_duplicates(subset="Seed_ID",keep='first',inplace=True)
    test_data.drop(test_data.index[test_data['Track_1'] == test_data['Track_2']], inplace = True)
    test_data.drop(["Track_1"],axis=1,inplace=True)
    test_data.drop(["Track_2"],axis=1,inplace=True)
    test_data.drop(test_data.index[test_data['Seed_CNN_Fit']<acceptance], inplace = True)
    if args.LinkAcceptance!='N':
         test_data.drop(test_data.index[test_data['Seed_Link_Fit']<link_acceptance], inplace = True)
    test_data.drop(["Seed_CNN_Fit"],axis=1,inplace=True)
    CurrentRecVertices=len(test_data.axes[0])
    TotalRecVertices+=CurrentRecVertices
    test_data=pd.merge(test_data, eval_data, how="inner", on=["Seed_ID"])
    RemainingRecVertices=len(test_data.axes[0])
    MatchedVertices+=RemainingRecVertices
    FakeVertices+=(CurrentRecVertices-RemainingRecVertices)
    Recall=round((float(MatchedVertices)/float(TotalMCVertices))*100,2)
    Precision=round((float(MatchedVertices)/float(TotalRecVertices))*100,2)
    if (Recall+Precision)==0:
        F1_Score=0
    else:
        F1_Score=round(2*((Recall*Precision)/(Recall+Precision)),2)
    print(UF.TimeStamp(), bcolors.OKGREEN+'Evaluation has been finished'+bcolors.ENDC)

    print(bcolors.HEADER+"#########################################  Results  #########################################"+bcolors.ENDC)
    print('Total 2-track combinations are expected according to Monte Carlo:',TotalMCVertices)
    print('Total 2-track combinations were reconstructed by EDER-VIANN:',TotalRecVertices)
    print('EDER-VIANN correct combinations were reconstructed:',MatchedVertices)
    print('Therefore the recall of the current model is',bcolors.BOLD+str(Recall), '%'+bcolors.ENDC)
    print('And the precision of the current model is',bcolors.BOLD+str(Precision), '%'+bcolors.ENDC)
    print('The F1 score of the current model is',bcolors.BOLD+str(F1_Score), '%'+bcolors.ENDC)

elif args.TypeOfAnalysis == 'ALL' or args.TypeOfAnalysis == 'FEDRA':
    print(UF.TimeStamp(), 'Evaluating FEDRA tracking reconstruction performance')
    eval_data=pd.read_csv(input_eval_file_location,header=0,usecols=['FEDRA_Seg_ID','MC_Mother_Track_ID'])
    eval_data.drop_duplicates(keep='first',inplace=True)
    rec_data=pd.read_csv(input_rec_file_location,header=0)
    rec_data=pd.merge(rec_data, eval_data, how="left", on=['FEDRA_Seg_ID'])
    seg_data=rec_data.drop(['x','y','z'],axis=1)
    print(seg_data)
    seg_data['FEDRA_Seg_No']=seg_data['FEDRA_Seg_ID']
    seg_data=seg_data.groupby(by=['MC_Mother_Track_ID','FEDRA_Seg_ID'])['FEDRA_Seg_No'].count().reset_index()
    print(seg_data)
    seg_data_segm_kpi=seg_data.drop(seg_data.index[seg_data['FEDRA_Seg_No'] < 2])
    print(seg_data_segm_kpi)
    exit()
    #ev_file=open(rec_file_location,'rb')
    #ev_data=pickle.load(ev_file)
    #ev_file.close()
    new_ev_data=[]
    for obj in ev_data:
        for  TH in obj.TrackHeader:
            new_ev_data.append([TH,str(obj.VX_CNN_ID)])
    del ev_data
    eval_data=pd.read_csv(input_eval_vx_location,header=0,usecols=['Track_ID','Mother_ID'])
    eval_data=eval_data.drop_duplicates()
    eval_data_vx_count=eval_data.groupby(by=['Mother_ID'])['Track_ID'].count().reset_index()
    eval_data_vx_count=eval_data_vx_count.rename(columns={'Track_ID': "MC_Multiplicity"})
    eval_data_vx_count=eval_data_vx_count.drop(eval_data_vx_count.index[eval_data_vx_count['MC_Multiplicity'] < 2])
    TotalFullVx=len(eval_data_vx_count)
    eval_data=pd.merge(eval_data, eval_data_vx_count, on=["Mother_ID"])
    new_ev_data = pd.DataFrame (new_ev_data, columns = ['Track_ID', 'Mother_ID'])
    new_ev_data=new_ev_data.drop_duplicates()
    new_ev_data_vx_count=new_ev_data.groupby(by=['Mother_ID'])['Track_ID'].count().reset_index()
    new_ev_data_vx_count=new_ev_data_vx_count.rename(columns={'Track_ID': "CNN_Multiplicity"})
    new_ev_data_vx_count=new_ev_data_vx_count.drop(new_ev_data_vx_count.index[new_ev_data_vx_count['CNN_Multiplicity'] < 2])
    TotalFullFedraVx=len(new_ev_data_vx_count)
    new_ev_data=pd.merge(new_ev_data, new_ev_data_vx_count, on=["Mother_ID"])
    new_ev_data=new_ev_data.rename(columns={'Mother_ID': "Vertex_ID"})
    combo_set=pd.merge(new_ev_data, eval_data, on=["Track_ID"])
    new_ev_data_map=combo_set.groupby(by=['Mother_ID','Vertex_ID','CNN_Multiplicity',"MC_Multiplicity"])['Track_ID'].count().reset_index()
    new_ev_data_map=new_ev_data_map.sort_values(['Vertex_ID','Mother_ID','Track_ID'],ascending=[1,1,0])
    new_ev_data_map=new_ev_data_map.rename(columns={'Track_ID': "Overlap"})
    new_ev_data_map.drop_duplicates(subset="Vertex_ID",keep='first',inplace=True)
    new_ev_data_map=new_ev_data_map.drop(new_ev_data_map.index[new_ev_data_map['Overlap'] < 2])
    new_ev_data_map['row_recall']=new_ev_data_map['Overlap']/new_ev_data_map['MC_Multiplicity']
    new_ev_data_map['row_precision']=new_ev_data_map['Overlap']/new_ev_data_map['CNN_Multiplicity']
    overall_recall=round((len(new_ev_data_map)/TotalFullVx)*100,2)
    overall_precision=round((len(new_ev_data_map)/TotalFullFedraVx)*100,2)
    recall_table=new_ev_data_map.groupby(by=["MC_Multiplicity"])['row_recall'].mean().reset_index()
    precision_table=new_ev_data_map.groupby(by=["MC_Multiplicity"])['row_precision'].mean().reset_index()
    count_table=new_ev_data_map.groupby(by=["MC_Multiplicity"])['row_precision'].count().reset_index()
    count_table=count_table.rename(columns={'row_precision': "No_Vx"})
    table=pd.merge(count_table, recall_table, how="inner", on=["MC_Multiplicity"])
    table=pd.merge(table, precision_table, how="inner", on=["MC_Multiplicity"])
    print('Multi-track vertices expected according to Monte Carlo:',TotalFullVx)
    print('Multi-track vertices reconstructed by EDER-VIANN:',TotalFullFedraVx)
    print('Multi-track vertices that were at least partially correctly reconstructed by EDER-VIANN:',len(new_ev_data_map))
    print('Therefore the overall recall of EDER-VIANN vertex reconstruction is',bcolors.BOLD+str(overall_recall), '%'+bcolors.ENDC)
    print('And the overall precision of the EDER-VIANN is',bcolors.BOLD+str(overall_precision), '%'+bcolors.ENDC)
    print('The average matched vertex reconstruction completeness is ',bcolors.BOLD+str(round(new_ev_data_map['row_recall'].mean(),4)*100), '%'+bcolors.ENDC)
    print('And the average vertex purity is',bcolors.BOLD+str(round(new_ev_data_map['row_precision'].mean(),4)*100), '%'+bcolors.ENDC)
    output_file_location = EOS_DIR + '/EDER-VIANN/Data/TEST_SET/E4_CNN_VX_REC_STATS.csv'
    table.to_csv(output_file_location,index=False)
    print(UF.TimeStamp(), bcolors.OKGREEN+"Stats have on FEDRA vertex reconstruction by Multiplicity has been written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
    print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
    #End of the script



