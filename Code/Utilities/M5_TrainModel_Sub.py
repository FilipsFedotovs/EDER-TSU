########################################################################################################################
#######################################  This simple script prepares data for CNN  #####################################




########################################    Import libraries    ########################################################
import csv
import argparse
import math
import ast
import numpy as np
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import copy


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from keras.optimizers import adam
from keras import callbacks
from keras import backend as K

########################## Visual Formatting #################################################
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

########################## Setting the parser ################################################
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--Res',help="Please enter the scaling resolution in microns", default='100')
parser.add_argument('--Mode',help="Please enter the mode: Create/Test/Train", default='Test')
parser.add_argument('--ImageSet',help="Please enter the image set", default='1')
parser.add_argument('--DNA',help="Please enter the model dna", default='[[4, 4, 1, 2, 2, 2, 2], [5, 4, 1, 1, 2, 2, 2], [5, 4, 2, 2, 2, 2, 2], [], [], [6, 4, 2], [5, 4, 2], [4, 4, 2], [], [], [7, 1, 1, 4]]')
parser.add_argument('--AFS',help="Please enter the user afs directory", default='.')
parser.add_argument('--EOS',help="Please enter the user eos directory", default='.')
parser.add_argument('--MaxX',help="Please enter the value of MaxX", default='3500.0')
parser.add_argument('--MaxY',help="Please enter the value of MaxY", default='1000.0')
parser.add_argument('--MaxZ',help="Please enter the value of MaxZ", default='20000.0')
parser.add_argument('--LR',help="Please enter the value of learning rate", default='Default')
parser.add_argument('--Epoch',help="Please enter the epoch number", default='1')
parser.add_argument('--ModelName',help="Name of the model", default='2T_100_MC_1_model')
parser.add_argument('--ModelNewName',help="Name of the model", default='2T_100_MC_1_model')
########################################     Initialising Variables    #########################################
args = parser.parse_args()
ImageSet=args.ImageSet
resolution=float(args.Res)
Mode=args.Mode
DNA=ast.literal_eval(args.DNA)
HiddenLayerDNA=[]
FullyConnectedDNA=[]
OutputDNA=[]
for gene in DNA:
    if DNA.index(gene)<=4 and len(gene)>0:
        HiddenLayerDNA.append(gene)
    elif DNA.index(gene)<=9 and len(gene)>0:
        FullyConnectedDNA.append(gene)
    elif DNA.index(gene)>9 and len(gene)>0:
        OutputDNA.append(gene)

act_fun_list=['N/A','linear','exponential','elu','relu', 'selu','sigmoid','softmax','softplus','softsign','tanh']
#Maximum bounds on the image size in microns
MaxX=float(args.MaxX)
MaxY=float(args.MaxY)
MaxZ=float(args.MaxZ)
#Converting image size bounds in line with resolution settings
boundsX=int(round(MaxX/resolution,0))
boundsY=int(round(MaxY/resolution,0))
boundsZ=int(round(MaxZ/resolution,0))
H=boundsX*2
W=boundsY*2
L=boundsZ
ValidModel=True
Accuracy=0.0
Accuracy0=0.0
##################################   Loading Directory locations   ##################################################
AFS_DIR=args.AFS
EOS_DIR=args.EOS
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import Utility_Functions as UF
#Load data configuration
EOSsubDIR=EOS_DIR+'/'+'EDER-VIANN'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
flocation=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/'+'M3_TRAIN_SET_'+ImageSet+'.csv'
vlocation=EOS_DIR+'/EDER-VIANN/Data/TRAIN_SET/M3_VALIDATION_SET.csv'
#train_history = tf.keras.callbacks.CSVLogger(EOSsubModelDIR+'/'+'train_log_'+ImageSet+'.csv', separator=",", append=True)

##############################################################################################################################
######################################### Starting the program ################################################################
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################  Initialising     EDER-VIANN   model creation module #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+flocation+bcolors.ENDC)
#Estimate number of images in the training file
#Calculate number of batches used for this job
TrainBatchSize=(OutputDNA[0][1]*4)
TrainImages=UF.LoadAllImages(flocation)
ValImages=UF.LoadAllImages(vlocation)
print(UF.TimeStamp(), bcolors.OKGREEN+"Train data has been loaded successfully..."+bcolors.ENDC)
NTrainBatches=math.ceil(float(len(TrainImages))/float(TrainBatchSize))
NValBatches=math.ceil(float(len(ValImages))/float(TrainBatchSize))
print(UF.TimeStamp(),'This iteration will be split in',bcolors.BOLD+str(NTrainBatches)+bcolors.ENDC,str(TrainBatchSize),'-size batches')
print(UF.TimeStamp(),'Loading the model...')
##### This but has to be converted to a part that interprets DNA code  ###################################
if args.LR=='Default':
  LR=10**(-int(OutputDNA[0][3]))
  opt = adam(learning_rate=10**(-int(OutputDNA[0][3])))
else:
    LR=float(args.LR)
    opt = adam(learning_rate=float(args.LR))
if Mode=='Train':
           model_name=EOSsubModelDIR+'/'+args.ModelName
           model=tf.keras.models.load_model(model_name)
           K.set_value(model.optimizer.learning_rate, LR)
           model.summary()
           print(model.optimizer.get_config())
if Mode!='Train':
           try:
             model = Sequential()
             for HL in HiddenLayerDNA:
                 Nodes=HL[0]*16
                 KS=(HL[2]*2)+1
                 PS=HL[3]
                 DR=float(HL[6]-1)/10.0
                 if HiddenLayerDNA.index(HL)==0:
                    model.add(Conv3D(Nodes, activation=act_fun_list[HL[1]],kernel_size=(KS,KS,KS),kernel_initializer='he_uniform', input_shape=(H,W,L,1)))
                 else:
                    model.add(Conv3D(Nodes, activation=act_fun_list[HL[1]],kernel_size=(KS,KS,KS),kernel_initializer='he_uniform'))
                 if PS>1:
                    model.add(MaxPooling3D(pool_size=(PS, PS, PS)))
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
           except:
              print(UF.TimeStamp(), bcolors.FAIL+"Invalid model..."+bcolors.ENDC)
              ValidModel=False
records=[]
print(UF.TimeStamp(),'Starting the training process... ')
for ib in range(0,NTrainBatches):
    StartSeed=(ib*TrainBatchSize)+1
    EndSeed=StartSeed+TrainBatchSize-1
    BatchImages=UF.LoadRenderImages(TrainImages,resolution,MaxX,MaxY,MaxZ,StartSeed,EndSeed,True)
    model.train_on_batch(BatchImages[0],BatchImages[1])
    progress=int(round((float(ib)/float(NTrainBatches))*100,0))
    print("Training in progress ",progress,' %', end="\r", flush=True)
print(UF.TimeStamp(),'Finished with the training... ')
print(UF.TimeStamp(),'Evaluating this epoch ')
model.reset_metrics()
for ib in range(0,NTrainBatches):
    StartSeed=(ib*TrainBatchSize)+1
    EndSeed=StartSeed+TrainBatchSize-1
    BatchImages=UF.LoadRenderImages(TrainImages,resolution,MaxX,MaxY,MaxZ,StartSeed,EndSeed,True)
    t=model.test_on_batch(BatchImages[0], BatchImages[1], reset_metrics=False)
    train_loss=t[0]
    train_acc=t[1]
model.reset_metrics()
for ib in range(0,NValBatches):
    StartSeed=(ib*TrainBatchSize)+1
    EndSeed=StartSeed+TrainBatchSize-1
    BatchImages=UF.LoadRenderImages(ValImages,resolution,MaxX,MaxY,MaxZ,StartSeed,EndSeed,True)
    a=model.test_on_batch(BatchImages[0], BatchImages[1], reset_metrics=False)
    val_loss=a[0]
    val_acc=a[1]
if ValidModel:
    model_name=EOSsubModelDIR+'/'+args.ModelNewName
    model.save(model_name)
    records.append([int(args.Epoch),ImageSet,len(TrainImages),train_loss,train_acc,val_loss,val_acc])
    UF.LogOperations(EOSsubModelDIR+'/'+'M5_M5_model_train_log_'+ImageSet+'.csv','StartLog', records)

