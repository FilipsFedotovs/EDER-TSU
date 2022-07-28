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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import copy
import pickle
import torch




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
parser.add_argument('--Mode',help="Please enter the mode: Create/Test/Train", default='Test')
parser.add_argument('--ImageSet',help="Please enter the image set", default='1')
parser.add_argument('--AFS',help="Please enter the user afs directory", default='.')
parser.add_argument('--EOS',help="Please enter the user eos directory", default='.')
parser.add_argument('--LR',help="Please enter the value of learning rate", default='Default')
parser.add_argument('--Epoch',help="Please enter the epoch number", default='1')
parser.add_argument('--ModelName',help="Name of the model", default='2T_100_MC_1_model')
parser.add_argument('--ModelNewName',help="Name of the model", default='2T_100_MC_1_model')
parser.add_argument('--f',help="Image set location (for test)", default='')
########################################     Initialising Variables    #########################################
args = parser.parse_args()
ImageSet=args.ImageSet
Mode=args.Mode


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
EOSsubDIR=EOS_DIR+'/'+'EDER-TSU'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
flocation=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/GM4_GM5_TRAIN_SET_'+ImageSet+'.pkl'
model_name=EOSsubModelDIR+'/'+args.ModelName

if Mode=='Test' and args.f!='':
   vlocation=args.f
else:
   vlocation=EOS_DIR+'/EDER-TSU/Data/TRAIN_SET/GM4_GM5_VALIDATION_SET.pkl'

##############################################################################################################################
######################################### Starting the program ################################################################
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################  Initialising     EDER-TSU   model creation module #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)




from torch.nn import Linear
from torch.nn import Softmax
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GMMConv
from torch_geometric.nn import global_mean_pool

num_node_features = 3
num_classes = 2
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GMMConv(num_node_features , hidden_channels, dim=3, kernel_size=4)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        self.softmax = Softmax()

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        x = self.softmax(x)
        return x

model = GCN(hidden_channels=64)
print(model)

#Estimate number of images in the training file
#Calculate number of batches used for this job

print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+flocation+bcolors.ENDC)
train_file=open(flocation,'rb')
TrainImages=pickle.load(train_file)
train_file.close()

train_dataset = []
for image in TrainImages :
    train_dataset.append(image.GraphSeed)
train_dataset = train_dataset[0:200]



print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+vlocation+bcolors.ENDC)
test_file=open(vlocation,'rb')
TestImages=pickle.load(test_file)
test_file.close()
test_dataset = []
for image in TestImages :
    test_dataset.append(image.GraphSeed)



from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     loss = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.edge_attr, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         loss += criterion(out, data.y)
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset), loss/len(loader.dataset)  # Derive ratio of correct predictions.


record = []
for epoch in range(1, 32):
    train()
    train_acc, train_loss = test(train_loader)
    test_acc, test_loss = test(test_loader)
    record.append([epoch, train_acc, train_loss, test_acc, test_loss])
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

record_df = pandas.DataFrame(record,columns = ['epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss'])
record_df.to_csv("GNN_PERFORMANCE.csv")
torch.save(model, model_name)


exit()




##### This but has to be converted to a part that interprets DNA code  ###################################
if args.LR=='Default':
  LR=10**(-int(OutputDNA[0][3]))
  opt = adam(learning_rate=10**(-int(OutputDNA[0][3])))
else:
    LR=float(args.LR)
    opt = adam(learning_rate=float(args.LR))
# if Mode=='Train':
#            model_name=EOSsubModelDIR+'/'+args.ModelName
#            model=tf.keras.models.load_model(model_name)
#            K.set_value(model.optimizer.learning_rate, LR)
#            model.summary()
#            print(model.optimizer.get_config())
# if Mode!='Train' and Mode!='Test':
#            #try:
#              model = Sequential()
#              for HL in HiddenLayerDNA:
#                  Nodes=HL[0]*16
#                  KS=(HL[2]*2)+1
#                  PS=HL[3]
#                  DR=float(HL[6]-1)/10.0
#                  if HiddenLayerDNA.index(HL)==0:
#                     model.add(Conv3D(Nodes, activation=act_fun_list[HL[1]],kernel_size=(KS,KS,KS),kernel_initializer='he_uniform', input_shape=(TrainImages[0].H,TrainImages[0].W,TrainImages[0].L,1)))
#                  else:
#                     model.add(Conv3D(Nodes, activation=act_fun_list[HL[1]],kernel_size=(KS,KS,KS),kernel_initializer='he_uniform'))
#                  if PS>1:
#                     model.add(MaxPooling3D(pool_size=(PS, PS, PS)))
#                  model.add(BatchNormalization(center=HL[4]>1, scale=HL[5]>1))
#                  model.add(Dropout(DR))
#              model.add(Flatten())
#              for FC in FullyConnectedDNA:
#                      Nodes=4**FC[0]
#                      DR=float(FC[2]-1)/10.0
#                      model.add(Dense(Nodes, activation=act_fun_list[FC[1]], kernel_initializer='he_uniform'))
#                      model.add(Dropout(DR))
#              model.add(Dense(2, activation=act_fun_list[OutputDNA[0][0]]))
 # Compile the model
            #  model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
            #  model.summary()
            #  print(model.optimizer.get_config())
           #  exit()
           #except:
           #   print(UF.TimeStamp(), bcolors.FAIL+"Invalid model, aborting the training..."+bcolors.ENDC)
           #   ValidModel=False
            #  exit()
# if Mode=='Test':
#            model_name=EOSsubModelDIR+'/'+args.ModelName
#            model=tf.keras.models.load_model(model_name)
#            K.set_value(model.optimizer.learning_rate, LR)
#            model.summary()
#            print(model.optimizer.get_config())
#            for ib in range(0,NValBatches):
#               StartSeed=(ib*TrainBatchSize)+1
#               EndSeed=StartSeed+TrainBatchSize-1
#               BatchImages=UF.LoadRenderImages(ValImages,StartSeed,EndSeed)
#               a=model.test_on_batch(BatchImages[0], BatchImages[1], reset_metrics=False)
#               val_loss=a[0]
#               val_acc=a[1]
#               progress=int(round((float(ib)/float(NValBatches))*100,0))
#               print("Validation in progress ",progress,' %',"Validation loss is:",val_loss,"Validation accuracy is:",val_acc , end="\r", flush=True)
#            print('Test is finished')
#            print("Final Validation loss is:",val_loss)
#            print("Final Validation accuracy is:",val_acc)
#            exit()
records=[]
print(UF.TimeStamp(),'Starting the training process... ')
for ib in range(0,NTrainBatches):
    StartSeed=(ib*TrainBatchSize)+1
    EndSeed=StartSeed+TrainBatchSize-1
    BatchImages=UF.LoadRenderImages(TrainImages,StartSeed,EndSeed)
    model.train_on_batch(BatchImages[0],BatchImages[1])
    progress=int(round((float(ib)/float(NTrainBatches))*100,0))
    print("Training in progress ",progress,' %', end="\r", flush=True)
print(UF.TimeStamp(),'Finished with the training... ')
print(UF.TimeStamp(),'Evaluating this epoch ')
model.reset_metrics()
for ib in range(0,NTrainBatches):
    StartSeed=(ib*TrainBatchSize)+1
    EndSeed=StartSeed+TrainBatchSize-1
    BatchImages=UF.LoadRenderImages(TrainImages,StartSeed,EndSeed)
    t=model.test_on_batch(BatchImages[0], BatchImages[1], reset_metrics=False)
    train_loss=t[0]
    train_acc=t[1]
model.reset_metrics()
for ib in range(0,NValBatches):
    StartSeed=(ib*TrainBatchSize)+1
    EndSeed=StartSeed+TrainBatchSize-1
    BatchImages=UF.LoadRenderImages(ValImages,StartSeed,EndSeed)
    a=model.test_on_batch(BatchImages[0], BatchImages[1], reset_metrics=False)
    val_loss=a[0]
    val_acc=a[1]
if ValidModel:
    model_name=EOSsubModelDIR+'/'+args.ModelNewName
    model.save(model_name)
    records.append([int(args.Epoch),ImageSet,len(TrainImages),train_loss,train_acc,val_loss,val_acc])
    UF.LogOperations(EOSsubModelDIR+'/'+'GM5_GM5_model_train_log_'+ImageSet+'.csv','StartLog', records)

