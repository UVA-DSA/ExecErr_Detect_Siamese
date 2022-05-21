from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import scipy.io
import numpy as np
import torch.nn.functional as F
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle as pkl
import pandas as pd
from models import TimeseriesNet_lstm
from util import TimeseriesData, load_data


net_type='lstm'

data_dir_S=''  # todo fill in the directory
data_dir_N=''
all_g_S= ["G1","G2","G3","G4","G6","G8","G9"]
all_g_N=["G1","G2","G3","G4","G6"]
all_x,all_y,error_mode,ids = load_data(all_g_S,data_dir_S)
all_x_N,all_y_N,error_mode_N,ids_N = load_data(all_g_N,data_dir_N)
all_x = np.hstack((all_x,all_x_N))
all_y = np.hstack((all_y,all_y_N))
error_mode = np.vstack((error_mode,error_mode))
ids = np.hstack((ids,ids_N))

unique_ids=np.unique(ids)
fold_data={}
for i,idx in enumerate(unique_ids):
    test_loc=np.where(ids==idx)[0]
    train_loc=np.where(ids!=idx)[0]
    fold_data[i]=[train_loc,test_loc]
all_x_N,all_y_N,error_mode_N,ids_N =[None,None,None,None]

F1scores={}
F1scores={}
predicted_result={}
precision_result={}
recall_result={}
expected_result={}
error_types={} 
net_type='lstm'


 
F1_mean=np.empty((0,1),dtype=float)
F1_std=np.empty((0,1),dtype=float)

win_len=30
stride=20
unique_ids=np.unique(ids)
fold_data={}
for i,idx in enumerate(unique_ids):
    test_loc=np.where(ids==idx)[0]
    train_loc=np.where(ids!=idx)[0]
    fold_data[i]=[train_loc,test_loc]
F1scores={}
precision_result={}
recall_result={}
predicted_result={}
error_types={}
expected_result={}

#F1scores, precision_results, recall_results, predicted_result,expected_result,error_types
for fold in range(len(unique_ids)):
   

    
    model = TimeseriesNet_lstm()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model.to(device)
   
    train_ids,test_ids = fold_data[fold]
    xtrain=all_x[train_ids]
    ytrain=all_y[train_ids]
    xtest=all_x[test_ids]
    ytest=all_y[test_ids]
    subject_train_ids = ids[train_ids]
    err_type_train = error_mode[train_ids,:]
    error_type_test=error_mode[test_ids,:]
    traindata = TimeseriesData(xtrain,ytrain,error_mode[train_ids,:] , win_len=win_len, stride=stride)
    
    w = [sum(np.array(ytrain)!='err')/sum(np.array(ytrain)=='err')]
    class_weight=torch.FloatTensor(w).to(device)
    criterion = nn.BCEWithLogitsLoss(class_weight)
    cur_para = pkl.load() # load the set of parameters Todo need to fill in the directory
    
    config = {
"lr": cur_para['lr'],
"batch_size": cur_para['batch_size']}
    trainloader = DataLoader(dataset=traindata, batch_size=int(config["batch_size"])\
                              ,shuffle=True,num_workers=0)
    numOfEpochs=cur_para['epoch']
    optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"])
    for n in range(numOfEpochs):
        model.train()
    
        for i, data in enumerate(trainloader,0):
            local_batch_L,local_y,er = data
            local_batch_L,local_y = local_batch_L.to(device),\
            local_y.to(device)
            if local_batch_L.shape[0]!=config["batch_size"]:continue
           # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs =torch.squeeze(model(local_batch_L))
            loss = criterion(outputs.view(-1),local_y.view(-1))
            loss.backward()
            optimizer.step()
    
    save_folder_model='Siamese/baseline/{}_model'.format(net_type)
    if not os.path.exists(save_folder_model):
        os.makedirs(save_folder_model)
    model_path = 'Siamese/baseline/{}_model/{}_model_{}.pth'.format(net_type,net_type,fold)
    torch.save(model.state_dict(), model_path)
    
    print('finish train')

    testdata = TimeseriesData(xtest,ytest,error_type_test,win_len=win_len, stride=stride)
    testloader = DataLoader(dataset=testdata, batch_size=int(config["batch_size"])\
                                  ,shuffle=False,num_workers=8)
    
    result=[]
    y_expected=[]
    y_error_type=np.empty((0,5), int)       # evaluate this fold's perforamnce in terms of F1 score
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    total = 0
    TP=0
    FP=0
    FN=0
    model.eval()
    for i, data in enumerate(testloader,0):
        with torch.no_grad():
            
            local_batch_L, local_y,error_type = data
            local_batch_L, local_y = local_batch_L.to(device),\
               local_y.to(device) 
            if local_y.view(-1).cpu().numpy().size==0: continue
            outputs =torch.squeeze(model(local_batch_L))
            
            outputs_val=outputs>0
            correct+=(outputs_val.view(-1) == local_y.view(-1)).sum().cpu().numpy()
            total +=outputs.cpu().numpy().size
            result.extend(outputs_val.view(-1).cpu().numpy())
            y_expected.extend(local_y.view(-1).cpu().numpy())
            y_error_type=np.vstack((y_error_type,error_type))

    try:
        TN, FP, FN, TP =confusion_matrix(y_expected,result).ravel()
        aa=[TN, FP, FN, TP]
        TN, FP, FN, TP=[0.001 if a==0 else a for a in aa]

        precision=(TP)/(TP+FP)
        recall=(TP)/(TP+FN)

        F1=2*precision*recall/(precision+recall)
    except ValueError:
        F1=0

 
    F1scores[fold]=F1
    predicted_result[fold]=result
    precision_result[fold]=precision
    recall_result[fold]=recall
    expected_result[fold]=y_expected
    error_types[fold]=y_error_type

       
data = [F1scores, precision_result, recall_result, predicted_result,expected_result,error_types]
save_folder='Siamese/baseline/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

file_name='Siamese/baseline/result_{}.p'.format( net_type)
pkl.dump(data ,open(file_name,"wb"))

               