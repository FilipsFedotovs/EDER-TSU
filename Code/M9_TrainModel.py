#import libraries
import numpy as np
from numpy.random import randint
import argparse
import ast
import csv
import os
from os import path
import random as rnd
from random import random
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#Set the parsing module
parser = argparse.ArgumentParser(description='Enter training job parameters')
parser.add_argument('--Mode',help="Please enter the running mode: 'R' for reset, 'C' for continuing the training", default='C')
parser.add_argument('--ModelName',help="Which model would you like to use as a base for training (please enter N if you want to train a new model from scratch)", default='Default')
parser.add_argument('--ModelNewName',help="Would you like to save your pretrained model as a separate one", default='Default')
parser.add_argument('--LR',help="Would you like to modify the model Learning Rate, If yes please enter it here eg: 0.01 ", default='Default')
args = parser.parse_args()
#setting main learning parameters
mode=args.Mode
_=0
#Loading Directory locations
csv_reader=open('../config',"r")
config = list(csv.reader(csv_reader))
for c in config:
    if c[0]=='AFS_DIR':
        AFS_DIR=c[1]
    if c[0]=='EOS_DIR':
        EOS_DIR=c[1]
csv_reader.close()

#Loading Data configurations
EOSsubDIR=EOS_DIR+'/'+'EDER-TSU'
EOSsubDataDIR=EOSsubDIR+'/'+'Data'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import Utility_Functions as UF
import Parameters as PM
if args.ModelName=='Default':
    ModelName=PM.Pre_CNN_Model_Name
else:
    ModelName=args.ModelName

print(bcolors.HEADER+"####################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################  Initialising EDER-TSU model training module     #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################            Written by Filips Fedotovs            #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################               PhD Student at UCL                 #########################"+bcolors.ENDC)
print(bcolors.HEADER+"###################### For troubleshooting please contact filips.fedotovs@cern.ch ##################"+bcolors.ENDC)
print(bcolors.HEADER+"####################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
#This code fragment covers the Algorithm logic on the first run
if mode=='R' and args.ModelName=='N':
 UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'M9', ['M9_M9','M9_PERFORMANCE_'], "SoftUsed == \"EDER-TSU-M9\"")
 job=[]
 job.append(1)
 job.append(1)
 job.append(PM.ModelArchitecturePlus)
 job.append(args.LR)
 job.append(ModelName)
 DNA=PM.ModelArchitecturePlus
 if args.ModelNewName=='Default':
     job.append(ModelName)
     OptionLine = ['Create', 1, EOS_DIR, AFS_DIR, DNA, args.LR, 1, ModelName, ModelName]
 else:
     try:
         import pickle
         train_file=open(EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/M8_M9_VALIDATION_SET.pkl','rb')
         TrainImages=pickle.load(train_file)
         import logging
         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
         logging.getLogger('tensorflow').setLevel(logging.FATAL)
         import warnings
         warnings.simplefilter(action='ignore', category=FutureWarning)
         import tensorflow as tf
         from tensorflow import keras
         from keras.models import Sequential
         from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
         from keras.optimizers import adam
         from keras import callbacks
         from keras import backend as K

         HiddenLayerDNA=[]
         FullyConnectedDNA=[]
         OutputDNA=[]
         act_fun_list=['N/A','linear','exponential','elu','relu', 'selu','sigmoid','softmax','softplus','softsign','tanh']
         for gene in DNA:
                if DNA.index(gene)<=4 and len(gene)>0:
                    HiddenLayerDNA.append(gene)
                elif DNA.index(gene)<=9 and len(gene)>0:
                    FullyConnectedDNA.append(gene)
                elif DNA.index(gene)>9 and len(gene)>0:
                    OutputDNA.append(gene)
         print(HiddenLayerDNA)
         model = Sequential()
         if args.LR=='Default':
          LR=10**(-int(OutputDNA[0][3]))
          opt = adam(learning_rate=10**(-int(OutputDNA[0][3])))
         else:
          LR=float(args.LR)
          opt = adam(learning_rate=float(args.LR))
         for HL in HiddenLayerDNA:
                 Nodes=HL[0]*16
                 KS=(np.array(HL[2])*2)+1
                 PS=HL[3]
                 DR=float(HL[6]-1)/10.0
                 if HiddenLayerDNA.index(HL)==0:
                    print(KS)
                    model.add(Conv3D(Nodes, activation=act_fun_list[HL[1]],kernel_size=(KS[0],KS[1],KS[2]),kernel_initializer='he_uniform', input_shape=(TrainImages[0].H,TrainImages[0].W,TrainImages[0].L,1)))
                 else:
                    print(KS)
                    model.add(Conv3D(Nodes, activation=act_fun_list[HL[1]],kernel_size=(KS[0],KS[1],KS[2]),kernel_initializer='he_uniform'))
                 if PS[0]>1 or PS[1]>1 or PS[2]>1:
                    print(PS)
                    model.add(MaxPooling3D(pool_size=(PS[0], PS[1], PS[2])))
                 model.add(BatchNormalization(center=HL[4]>1, scale=HL[5]>1))
                 model.add(Dropout(DR))
         model.add(Flatten())
         for FC in FullyConnectedDNA:
                     Nodes=4**FC[0]
                     DR=float(FC[2]-1)/10.0
                     model.add(Dense(Nodes, activation=act_fun_list[FC[1]], kernel_initializer='he_uniform'))
                     model.add(Dropout(DR))
         model.add(Dense(2, activation=act_fun_list[OutputDNA[0][0]]))
 # Compile the model
         model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
         model.summary()
         print(model.optimizer.get_config())
         print(UF.TimeStamp(),bcolors.OKGREEN+'Model configuration is valid...'+bcolors.ENDC)
         job.append(args.ModelNewName)
         DNA = '"' + str(PM.ModelArchitecturePlus) + '"'
         OptionLine = ['Create', 1, EOS_DIR, AFS_DIR, DNA, args.LR, 1, ModelName, args.ModelNewName]
     except:
        print(UF.TimeStamp(),bcolors.FAIL+'Model configuration is invalid, exiting now...'+bcolors.ENDC)
        exit()
 print(UF.TimeStamp(),bcolors.OKGREEN+'Job description has been created'+bcolors.ENDC)
 PerformanceHeader=[['Epochs','Set','Training Samples','Train Loss','Train Accuracy','Validation Loss','Validation Accuracy']]
 UF.LogOperations(EOSsubModelDIR+'/M9_PERFORMANCE_'+job[5]+'.csv','StartLog',PerformanceHeader)
 OptionHeader = [' --Mode ', ' --ImageSet ', ' --EOS ', " --AFS ", " --DNA ",
                 " --LR ", " --Epoch ", " --ModelName ", " --ModelNewName "]
 SHName = AFS_DIR + '/HTCondor/SH/SH_M9.sh'
 SUBName = AFS_DIR + '/HTCondor/SUB/SUB_M9.sub'
 MSGName = AFS_DIR + '/HTCondor/MSG/MSG_M9'
 ScriptName = AFS_DIR + '/Code/Utilities/M9_TrainModel_Sub.py '
 UF.SubmitJobs2Condor(
     [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-M9', True,
      True])
 job[4]=job[5]
 UF.LogOperations(EOSsubModelDIR+'/M9_M9_JobTask.csv','StartLog',[job])
 print(bcolors.BOLD+"Please the job completion in few hours by running this script with the option C"+bcolors.ENDC)
elif mode=='R':
 UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'M9', ['M9_M9','M9_PERFORMANCE_'], "SoftUsed == \"EDER-TSU-M9\"")
 job=[]
 job.append(1)
 job.append(1)
 job.append(PM.ModelArchitecturePlus)
 job.append(args.LR)
 job.append(ModelName)
 DNA = '"' + str(PM.ModelArchitecturePlus) + '"'
 if args.ModelNewName=='Default':
     job.append(ModelName)
     OptionLine = ['Train', 1, EOS_DIR, AFS_DIR, DNA, args.LR, 1,  ModelName, ModelName]
 else:
     job.append(args.ModelNewName)
     OptionLine = ['Train', 1, EOS_DIR, AFS_DIR, DNA, args.LR, 1, ModelName, args.ModelNewName]
 print(UF.TimeStamp(),bcolors.OKGREEN+'Job description has been created'+bcolors.ENDC)
 PerformanceHeader=[['Epochs','Set','Training Samples','Train Loss','Train Accuracy','Validation Loss','Validation Accuracy']]
 UF.LogOperations(EOSsubModelDIR+'/M9_PERFORMANCE_'+job[5]+'.csv','StartLog',PerformanceHeader)
 OptionHeader = [' --Mode ', ' --ImageSet ', ' --EOS ', " --AFS ", " --DNA ",
                 " --LR ", " --Epoch ", " --ModelName ", " --ModelNewName "]
 SHName = AFS_DIR + '/HTCondor/SH/SH_M9.sh'
 SUBName = AFS_DIR + '/HTCondor/SUB/SUB_M9.sub'
 MSGName = AFS_DIR + '/HTCondor/MSG/MSG_M9'
 ScriptName = AFS_DIR + '/Code/Utilities/M9_TrainModel_Sub.py '
 UF.SubmitJobs2Condor(
     [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-M9', True,
      True])
 job[4]=job[5]
 UF.LogOperations(EOSsubModelDIR+'/M9_M9_JobTask.csv','StartLog',[job])
 print(bcolors.BOLD+"Please the job completion in few hours by running this script with the option C"+bcolors.ENDC)
if mode=='C':
   CurrentSet=0
   print(UF.TimeStamp(),'Continuing the training that has been started before')
   print(UF.TimeStamp(),'Checking the previous job completion...')
   csv_reader=open(EOSsubModelDIR+'/M9_M9_JobTask.csv',"r")
   PreviousJob = list(csv.reader(csv_reader))
   if args.LR!='Default':
       PreviousJob[0][3]=args.LR
   csv_reader.close()
   CurrentSet=int(PreviousJob[0][0])
   CurrentEpoch=int(PreviousJob[0][1])
   ###Working out the latest batch
   ###Working out the remaining jobs
   required_file_name=EOSsubModelDIR+'/M9_M9_model_train_log_'+PreviousJob[0][0]+'.csv'
   if os.path.isfile(required_file_name)==False:
     print(UF.TimeStamp(),bcolors.WARNING+'Warning, the HTCondor job is still running'+bcolors.ENDC)
     print(bcolors.BOLD+'If you would like to wait and try again later please enter W'+bcolors.ENDC)
     print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
     UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
     if UserAnswer=='W':
         print(UF.TimeStamp(),'OK, exiting now then')
         exit()
     if UserAnswer=='R':
        if CurrentSet==1:
          OptionLine = ['Create', PreviousJob[0][0], EOS_DIR, AFS_DIR, '"'+str(PreviousJob[0][2])+'"', PreviousJob[0][3], PreviousJob[0][1], PreviousJob[0][4], PreviousJob[0][5]]
        if CurrentSet>1:
          OptionLine = ['Train', PreviousJob[0][0], EOS_DIR, AFS_DIR, '"'+str(PreviousJob[0][2])+'"', PreviousJob[0][3], PreviousJob[0][1], PreviousJob[0][4], PreviousJob[0][5]]
        OptionHeader = [' --Mode ', ' --ImageSet ', ' --EOS ', " --AFS ", " --DNA ",
                        " --LR ", " --Epoch ", " --ModelName ", " --ModelNewName "]
        SHName = AFS_DIR + '/HTCondor/SH/SH_M9.sh'
        SUBName = AFS_DIR + '/HTCondor/SUB/SUB_M9.sub'
        MSGName = AFS_DIR + '/HTCondor/MSG/MSG_M9'
        ScriptName = AFS_DIR + '/Code/Utilities/M9_TrainModel_Sub.py '
        UF.SubmitJobs2Condor(
            [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-M9', True,
             True])
        print(UF.TimeStamp(), bcolors.OKGREEN+"The Training Job for the CurrentSet",CurrentSet,"have been resubmitted"+bcolors.ENDC)
        print(bcolors.OKGREEN+"Please check it in a few hours"+bcolors.ENDC)
        exit()
   else:
      print(UF.TimeStamp(),bcolors.OKGREEN+'The training of the model by using image set',CurrentSet,'has been completed'+bcolors.ENDC)
      print(bcolors.BOLD+'Would you like to continue training?'+bcolors.ENDC)
      UserAnswer=input(bcolors.BOLD+"Please, enter Y/N\n"+bcolors.ENDC)
      if UserAnswer=='Y':
          csv_reader=open(required_file_name,"r")
          PreviousHeader = list(csv.reader(csv_reader))
          UF.LogOperations(EOSsubModelDIR+'/M9_PERFORMANCE_'+PreviousJob[0][5]+'.csv','UpdateLog',PreviousHeader)
          os.unlink(required_file_name)
          print(UF.TimeStamp(),'Creating next batch',CurrentSet+1)
          print(bcolors.BOLD+'Image Set',CurrentSet,' is completed'+bcolors.ENDC)
          if os.path.isfile(EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/'+'M8_M9_TRAIN_SET_'+str(CurrentSet+1)+'.pkl')==False:
              print(bcolors.WARNING+'No more training files left, restarting the new epoch...'+bcolors.ENDC)
              CurrentSet=1
              CurrentEpoch+=1
              PreviousJob[0][0]=str(CurrentSet)
              PreviousJob[0][1]=str(CurrentEpoch)
              OptionLine = ['Train', PreviousJob[0][0], EOS_DIR, AFS_DIR, '"'+str(PreviousJob[0][2])+'"', PreviousJob[0][3], PreviousJob[0][1], PreviousJob[0][4], PreviousJob[0][5]]
              OptionHeader = [' --Mode ', ' --ImageSet ', ' --EOS ', " --AFS ", " --DNA ",
                              " --LR ", " --Epoch ", " --ModelName ", " --ModelNewName "]
              SHName = AFS_DIR + '/HTCondor/SH/SH_M9.sh'
              SUBName = AFS_DIR + '/HTCondor/SUB/SUB_M9.sub'
              MSGName = AFS_DIR + '/HTCondor/MSG/MSG_M9'
              ScriptName = AFS_DIR + '/Code/Utilities/M9_TrainModel_Sub.py '
              UF.SubmitJobs2Condor(
                  [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-M9', True,
                   True])
              print(UF.TimeStamp(),bcolors.OKGREEN+'The Image Set',CurrentSet,'has been submitted to HTCondor'+bcolors.ENDC)
              exit()
          print(bcolors.BOLD+'Would you like to continue training?'+bcolors.ENDC)
          UserAnswer=input(bcolors.BOLD+"Please, enter Y/N\n"+bcolors.ENDC)
          if UserAnswer=='Y':
              CurrentSet+=1
              PreviousJob[0][0]=str(CurrentSet)
              UF.LogOperations(EOSsubModelDIR+'/M9_M9_JobTask.csv','StartLog',PreviousJob)
              OptionLine = ['Train', PreviousJob[0][0], EOS_DIR, AFS_DIR, '"'+str(PreviousJob[0][2])+'"', PreviousJob[0][3], PreviousJob[0][1], PreviousJob[0][4], PreviousJob[0][5]]
              OptionHeader = [' --Mode ', ' --ImageSet ', ' --EOS ', " --AFS ", " --DNA ",
                              " --LR ", " --Epoch ", " --ModelName ", " --ModelNewName "]
              SHName = AFS_DIR + '/HTCondor/SH/SH_M9.sh'
              SUBName = AFS_DIR + '/HTCondor/SUB/SUB_M9.sub'
              MSGName = AFS_DIR + '/HTCondor/MSG/MSG_M9'
              ScriptName = AFS_DIR + '/Code/Utilities/M9_TrainModel_Sub.py '
              UF.SubmitJobs2Condor(
                  [OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-TSU-M9', True,
                   True])
              print(UF.TimeStamp(),bcolors.OKGREEN+'The next Image Set',CurrentSet,'has been submitted to HTCondor'+bcolors.ENDC)
              print(bcolors.BOLD,'Please run the script in few hours with --MODE C setting'+bcolors.ENDC)
              print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)

          if UserAnswer=='N':
              print(UF.TimeStamp(),bcolors.OKGREEN+'Training is finished then, thank you and good bye'+bcolors.ENDC)
              print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
      else:
          csv_reader=open(required_file_name,"r")
          PreviousHeader = list(csv.reader(csv_reader))
          UF.LogOperations(EOSsubModelDIR+'/M9_PERFORMANCE_'+PreviousJob[0][5]+'.csv','UpdateLog',PreviousHeader)
          os.unlink(required_file_name)
          print(UF.TimeStamp(),bcolors.OKGREEN+'Training is finished then, thank you and good bye'+bcolors.ENDC)
          print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
exit()


