from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import scipy.io
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import precision_recall_fscore_support
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import warnings
import pickle as pkl
import pandas as pd
import time

def load_data(gesture,data_dir=None):
    '''
    This function load the data and return the training and testset

    Parameters
    ----------
    gesture : char
    data_dir : char
        DESCRIPTION. The default is './data':.

    Returns
    -------
    trainset and testset.

    '''
    # GST*
    if type(data_dir)==list:
        init_x=np.empty((0,1),dtype=object)
        init_y=np.empty((0,1),dtype=object)
        error_mode=np.empty((0,5),dtype=object)
        valid=np.empty((0,1),dtype=object)
        for dir in data_dir:
            mat = scipy.io.loadmat(dir)
            shape = mat['G1'].shape[1]
            g_data=np.empty((0,shape),dtype=float)
            d = mat[gesture]
            g_data=np.vstack((g_data,d))
            init_x=np.append(init_x,g_data[:,0])
            init_y=np.append(init_y,g_data[:,2])
            error_mode=np.vstack((error_mode,g_data[:,3:-1]))
            valid=np.append(valid,g_data[:,-1])
        
        return (init_x,init_y,error_mode,valid)

    #GSTS
    if type(gesture)!=list:
        mat = scipy.io.loadmat(data_dir)
        cur = mat[gesture]
        init_x = cur[:,0]
        init_y = cur[:,2]
        error_mode = cur[:,3:-1]
        valid = cur[:,-1]
        return (init_x,init_y,error_mode,valid)
    #G*TS
    else:
        mat = scipy.io.loadmat(data_dir)
        shape = mat['G1'].shape[1]
        g_data=np.empty((0,shape),dtype=float)
        for G in gesture:
            d = mat[G]
            g_data=np.vstack((g_data,d))
        init_x = g_data[:,0]
        init_y =g_data[:,2]
        error_mode = g_data[:,3:-1]
        valid = np.array(g_data[:,-1])
        return (init_x,init_y,error_mode,valid)

class TimeseriesData(Dataset):
    def __init__(self, init_x,init_y,error_mode,win_len=30, stride=1):
        # for single input
        ''' 
        
        Parameters
        ----------
        init_x : array of objects
            each instance of init_x is a multiD array.
        init_y : array of class
        
        error_mode : an Nx5 matrix indicating error type

        win_len : TYPE, optional
            DESCRIPTION. The default is 1.
        stride : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        '''

        # use a sliding window to create more data per trial
        self.L=[]

        self.y=[]
        self.err=np.empty((0,5),int)
        for idx,data in enumerate(init_x):
            time_len = data.shape[0]
            start = (time_len-win_len)%stride
            y_val=init_y[idx]
            L_data = data
            cur_data_L=[L_data[i:i+win_len,:].T for i in \
                      np.arange(start,time_len-win_len+stride,stride) ]
            for i,seq in enumerate(cur_data_L):
                count_zero=sum(np.array(seq[0,:])==0)/win_len
                if count_zero<0.4:
                    self.L.append(seq)
                    self.y.append(y_val)
                    self.err=np.vstack((self.err,error_mode[idx,:]))
                    
        self.y = [val=='err' for val in self.y]
        self.y = np.array(self.y, dtype=np.float32)
        self.L = np.array(self.L, dtype=np.float32)
        self.err = np.array(self.err, dtype=np.float32)
                
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,index):
        return self.L[index],self.y[index],self.err[index,:]


class TimeseriesPairs_train_v2(Dataset):
    def __init__(self, win_len,stride,x_train,y_train):
        ## for the dual
        ## pair the err with nor disregard of left and right hand

        '''
        Parameters
        ----------
        gesture : char
            The gesture to run the network on.
        datadir : char
            The .mat file.

        Returns
        -------
        None.

        '''
        self.L=[]
        self.L2=[]
        self.y=[]
        init_x = x_train
        init_y = y_train
        
        y_values = [val=='err' for val in init_y]
        
        pairs=[]
        pairs_y=[]  # 0 nor/nor 1 error/nor
        for i in np.arange(0,init_x.shape[0]):
            for j in np.arange(i+1,init_x.shape[0]):
                if y_values[i]==0 and y_values[j]==0 :
                    pairs_y.append(0)
                    pairs.append((i,j))
                elif (y_values[i]==0 and y_values[j]==1) or (y_values[i]==1 and y_values[j]==0):
                    pairs_y.append(1)
                    pairs.append((i,j))
                else: continue
        
        for i, indexs in enumerate(pairs):
            ar1,ar2=indexs
            L_data1 = init_x[ar1]
            time_len1 = L_data1.shape[0]
            start1 = (time_len1-win_len)%stride
            y_val=pairs_y[i]
            cur_data_L=[L_data1[i:i+win_len,:].T for i in \
                      np.arange(start1,time_len1-win_len+stride,stride) ]
            first=[]
            for idx,seq in enumerate(cur_data_L):
                count_zero=sum(np.array(seq[0,:])==0)/win_len
                if count_zero<0.4:
                    first.append(seq)
                    
            L_data2 = init_x[ar2]
            time_len2 = L_data2.shape[0]
            start2 = (time_len2-win_len)%stride
            cur_data_L2=[L_data2[i:i+win_len,:].T for i in \
                      np.arange(start2,time_len2-win_len+stride,stride) ]
            second=[]
            for idx,seq in enumerate(cur_data_L2):
                count_zero=sum(np.array(seq[0,:])==0)/win_len
                if count_zero<0.4:
                    second.append(seq)
            
            for a in np.arange(0,len(first)):
                for b in np.arange(0,len(second)):
                    self.L.append(first[a])
                    self.L2.append(second[b])
                    self.y.append(y_val)
            
        
        self.y = np.array(self.y, dtype=np.float32)
        self.L = np.array(self.L, dtype=np.float32)
        self.L2 = np.array(self.L2, dtype=np.float32)


    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,index):
        return self.L[index],self.L2[index],self.y[index]



class TimeseriesPairs_test_v2(Dataset):
    def __init__(self,win_len,stride,x_train,y_train,x_test, y_test):
        '''
        

        Parameters
        ----------
         win_len : int
            The default is 1.
        stride : int
            The default is 1.
        x_train : 1d array of object shape=[x,]
            the time series data.
        y_train : 1d array error or normal
            
        x_test : 1d array of object shape=[x,]

        y_test : 1d array error or normal
        
        error_mode: 1x5 one hot encoding arrray for the error modes of the test set

        Returns
        -------
        None.

        '''

        self.L=[]
        self.L2=[]
        self.y=[]
        self.idx=np.empty((0,2), int) # test trial, and window idx

        
        y_train = [val=='err' for val in y_train]
        y_test  =[val=='err' for val in y_test]
        
        pairs=[]
        pairs_y=[]  # 0 nor/nor 1 error/nor
        for i in np.arange(0,x_train.shape[0]):
            for j in np.arange(0,x_test.shape[0]):
                #print(f'y_train{y_train[i]},y_test{y_test[j]}\n')
                if y_train[i]==0 and y_test[j]==0 :
                    pairs_y.append(0)
                    pairs.append((i,j))
                    # pair with all the normal trials
                elif (y_train[i]==0 and y_test[j]==1): #or (y_train[i]==1 and y_test[j]==0):
                    pairs_y.append(1)
                    pairs.append((i,j))
                    
                else: continue
        
        for i, indexs in enumerate(pairs):
            ar1,ar2=indexs
            L_data1 = x_train[ar1]
            time_len1 = L_data1.shape[0]
            start1 = (time_len1-win_len)%stride
            y_val=pairs_y[i]
            cur_data_L=[L_data1[i:i+win_len,:].T for i in \
                      np.arange(start1,time_len1-win_len+stride,stride) ]
            first=[] # train array
            for idx,seq in enumerate(cur_data_L):
                count_zero=sum(np.array(seq[0,:])==0)/win_len
                if count_zero<0.4:
                    first.append(seq)
                    
            L_data2 = x_test[ar2]
            time_len2 = L_data2.shape[0]
            start2 = (time_len2-win_len)%stride
            cur_data_L2=[L_data2[i:i+win_len,:].T for i in \
                      np.arange(start2,time_len2-win_len+stride,stride) ]
            second=[]
            for idx,seq in enumerate(cur_data_L2):
                count_zero=sum(np.array(seq[0,:])==0)/win_len
                if count_zero<0.4:
                    second.append(seq)
            
            for b in np.arange(0,len(second)):
                for a in np.arange(0,len(first)):
                    self.L.append(first[a])
                    self.L2.append(second[b])
                    self.y.append(y_val)
                    self.idx=np.vstack((self.idx,[ar2,b])) # saving the test trial id, and the window
                    # append the error type for this training example
            
            
        self.y = np.array(self.y, dtype=np.float32)
        self.L = np.array(self.L, dtype=np.float32)
        self.L2 = np.array(self.L2, dtype=np.float32)
        # self.idx denote the classification result of each test sliding window
        self.idx = np.array(self.idx, dtype=np.float32)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,index):
        return self.L[index],self.L2[index],self.y[index],self.idx[index,:]


def para_lstm_dual(config,net,xtrain,ytrain,subject_train_ids,error_mode_train, gesture,win_len, stride, checkpoint_dir=None):
   
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    net.to(device) 
    
    unique_ids=np.unique(subject_train_ids)
    fold_data={}
    for i,idx in enumerate(unique_ids):
        test_loc=np.where(subject_train_ids==idx)[0]
        train_loc=np.where(subject_train_ids!=idx)[0]
        fold_data[i]=[train_loc,test_loc]
 
    
   
    optimizer = torch.optim.Adam(net.parameters(),lr=config["lr"])
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    

    F1s=[]
    for fold in range(len(unique_ids)):
      
        train_ids,test_ids = fold_data[fold]
        x_train=xtrain[train_ids]
        y_train=ytrain[train_ids]
        x_test=xtrain[test_ids]
        y_test=ytrain[test_ids]
        
  
        traindata = TimeseriesPairs_train_v2( win_len,stride,x_train,y_train)
        
        # set the class weight
        trainloader2 = DataLoader(dataset=traindata, batch_size=traindata.__len__()\
                                      ,shuffle=True,num_workers=8)
        x,x2,y=iter(trainloader2).next()
        w = [sum(y==0)/sum(y==1)]
        class_weight=torch.FloatTensor(w).to(device)
        criterion = nn.BCEWithLogitsLoss(class_weight)
        # loaders
        
        
        trainloader = DataLoader(dataset=traindata, batch_size=int(config["batch_size"])\
                                      ,shuffle=True,num_workers=8)
        valdata = TimeseriesPairs_test_v2( win_len,stride,x_train,y_train,x_test, y_test)
        valloader = DataLoader(dataset=valdata, batch_size=int(config["batch_size"])\
                                      ,shuffle=False,num_workers=8)
        N_EPOCHS=config["epoch"]
        
        for epoch in range(N_EPOCHS):
            running_loss = 0.0 
            epoch_steps = 0
            net.train()
            for i, data in enumerate(trainloader,0):
                local_batch_L,local_batch_L2,local_y = data
                local_batch_L,local_batch_L2,local_y = local_batch_L.to(device),local_batch_L2.to(device),\
                    local_y.to(device)
                if local_batch_L.shape[0]!=config["batch_size"]:continue
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs =torch.squeeze(net(local_batch_L,local_batch_L2))
    
                loss = criterion(outputs.view(-1),local_y.view(-1))
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                epoch_steps += 1
                if i % 50 ==1:
                    # print("[%d, %5d] loss: %.3f" % (epoch+1,i+1,running_loss/epoch_steps))
                    running_loss = 0.0
            
            #validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        all_predict=[]
        result=[]
        y_expected=[]
        net.eval()
        for i, data in enumerate(valloader,0):
            with torch.no_grad():
                
                local_batch_L, local_batch_L2,local_y = data
                local_batch_L,local_batch_L2,local_y = local_batch_L.to(device),\
                    local_batch_L2.to(device),local_y.to(device)
                # if local_y.dim() == 0: continue
                if local_y.view(-1).cpu().numpy().size==0: continue
                outputs =torch.squeeze(net(local_batch_L,local_batch_L2))
                
                outputs_val=outputs>0
                all_predict.append(outputs)
                correct+=(outputs_val.view(-1) == local_y.view(-1)).sum().item()
                result.extend(outputs_val.view(-1).cpu().numpy())
                y_expected.extend(local_y.view(-1).cpu().numpy())
    
                total +=outputs.cpu().numpy().size

                loss = criterion(outputs.view(-1), local_y.view(-1))
                val_loss += loss.cpu().numpy()
                val_steps+=1
                
        try:
            TN, FP, FN, TP =confusion_matrix(y_expected, result).ravel()
            aa=[TN, FP, FN, TP]
            TN, FP, FN, TP=[0.001 if a==0 else a for a in aa]

        
            precision=TP/(TP+FP)
            recall=TP/(TP+FN)

            F1=2*precision*recall/(precision+recall)
        except ValueError:
            F1=0
        F1s.append(F1)

        
    with tune.checkpoint_dir(epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir,"checkpoint.pth")
        torch.save((net.state_dict(), optimizer.state_dict()),path)
 

    tune.report(loss=(val_loss/val_steps),accuracy=np.mean(F1s))
    print('Finish Training')   
    # print('Finish Training')   


def para_lstm_single(config,net,xtrain,ytrain,subject_train_ids,error_mode_train, gesture,win_len, stride, checkpoint_dir=None):
   
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    net.to(device)
    
    unique_ids=np.unique(subject_train_ids)
    fold_data={}
    for i,idx in enumerate(unique_ids):
        test_loc=np.where(subject_train_ids==idx)[0]
        train_loc=np.where(subject_train_ids!=idx)[0]
        fold_data[i]=[train_loc,test_loc]
 
    
   
    optimizer = torch.optim.Adam(net.parameters(),lr=config["lr"])
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    


    

    N_EPOCHS=config["epoch"]
    run=True
    state=0

    F1s=[]
    for fold in range(len(unique_ids)):
      
        train_ids,test_ids = fold_data[fold]
        x_train=xtrain[train_ids]
        y_train=ytrain[train_ids]
        x_test=xtrain[test_ids]
        y_test=ytrain[test_ids]
        
        
        traindata = TimeseriesData(x_train,y_train,error_mode_train[train_ids,:] , win_len=win_len, stride=stride)
        trainloader = DataLoader(dataset=traindata, batch_size=int(config["batch_size"])\
                                  ,shuffle=True,num_workers=0)
            
        testdata = TimeseriesData(x_test,y_test,error_mode_train[test_ids,:],win_len=win_len, stride=stride)
        testloader = DataLoader(dataset=testdata, batch_size=int(config["batch_size"])\
                                      ,shuffle=False,num_workers=8)
        w = [sum(np.array(y_train)!='err')/sum(np.array(y_train)=='err')]
        class_weight=torch.FloatTensor(w).to(device)
        criterion = nn.BCEWithLogitsLoss(class_weight)
        for epoch in range(N_EPOCHS):
            running_loss = 0.0
            epoch_steps = 0
            net.train()
            for i, data in enumerate(trainloader,0):
                local_batch_L,local_y,er = data
                local_batch_L,local_y = local_batch_L.to(device),local_y.to(device)
                if local_batch_L.shape[0]!=config["batch_size"]:continue
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs =torch.squeeze(net(local_batch_L))
                #print(f"input size{local_batch_L.shape}")
                #print(f"y size{local_y.shape}")
                if local_y.view(-1).cpu().numpy().size==0: continue
                loss = criterion(outputs.view(-1),local_y.view(-1))
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                epoch_steps += 1
                if i % 500 ==1:
                    # print("[%d, %5d] loss: %.3f" % (epoch+1,i+1,running_loss/epoch_steps))
                    running_loss = 0.0
            
        #validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        net.eval()
        result=[]
        y_expected=[]
        print('finish_train_{}'.format(fold))

        for i, data in enumerate(testloader,0):
            with torch.no_grad():
                
                local_batch_L,local_y,er = data
                local_batch_L,local_y = local_batch_L.to(device),local_y.to(device)
                # if local_y.dim() == 0: continue
                if local_y.view(-1).cpu().numpy().size==0: continue
                outputs =torch.squeeze(net(local_batch_L))
                if outputs.view(-1).cpu().numpy().size!=local_y.view(-1).cpu().numpy().size: 
                    continue
                outputs_val=outputs>0
                correct+=(outputs_val==local_y.view(-1)).sum().item()
                total +=outputs.cpu().numpy().size
                # if outputs.dim() == 0: continue
                result.extend(outputs_val.view(-1).cpu().numpy())
                y_expected.extend(local_y.view(-1).cpu().numpy())
                
                loss = criterion(outputs.view(-1), local_y.view(-1))
                val_loss += loss.cpu().numpy()
                val_steps+=1
     
        try:
            TN, FP, FN, TP =confusion_matrix(y_expected, result).ravel()
            aa=[TN, FP, FN, TP]
            TN, FP, FN, TP=[0.001 if a==0 else a for a in aa]
    
            
            precision=TP/(TP+FP)
            recall=TP/(TP+FN)
    
            F1=2*precision*recall/(precision+recall)
        except ValueError:
            F1=0
        F1s.append(F1)

    with tune.checkpoint_dir(epoch) as checkpoint_dir:
        print(checkpoint_dir)
        path = os.path.join(checkpoint_dir,"checkpoint.pth")
        torch.save((net.state_dict(), optimizer.state_dict()),path)
    
    tune.report(loss=(val_loss/val_steps),accuracy=np.mean(F1s))
    print('Finish Training')   

def find_para_single(G,net,xtrain,ytrain,subject_train_ids,error_mode_train,Task,fold,win_len,stride,net_type):
    num_samples=8
    gesture=G
    random_state=19
    # win_len=30,stride=3
    

    config = {
    "lr": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([10,16,32]),
    "epoch":tune.choice([2,4])} #8,16
    
    scheduler = ASHAScheduler(metric="loss",mode="min",max_t=200,grace_period=10,\
                              reduction_factor=3)
    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
    

    result = tune.run(
        partial(para_lstm, net=net,xtrain=xtrain,ytrain=ytrain,subject_train_ids=subject_train_ids,error_mode_train=error_mode_train,\
                gesture=gesture,win_len=win_len,stride=stride),
                      resources_per_trial={"cpu": 10, "gpu": 1},\
                          config=config, num_samples=num_samples, scheduler=scheduler,progress_reporter=reporter)
            
    best_trial = result.get_best_trial("accuracy", "max", "last")
    
    print(best_trial.checkpoint.value)
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    paras=best_trial.config
    file_name='Siamese/gesture_specific_task_nonspecific/{}_para_{}/{}_para_{}_nested_{}_{}.p'.format(G, net_type,G,net_type,Task,fold)
    save_folder='Siamese/gesture_specific_task_nonspecific/{}_para_{}/'.format(G,net_type)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    pkl.dump(paras, open(file_name,"wb"))
    return paras

def find_para_dual(G,net,xtrain,ytrain,subject_train_ids,error_mode_train,Task,fold,win_len,stride,net_type):  
    num_samples=5
    gesture=G
 
 
    config = {'lr':tune.loguniform(1e-5, 1e-3),
    "epoch":tune.choice([2]), #,4
    "batch_size": tune.choice([16,32,64,128])} #8,
    
    scheduler = ASHAScheduler(metric="loss",mode="min",max_t=100,grace_period=10,\
                              reduction_factor=3)
    # reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
    reporter = CLIReporter(metric_columns=[ "accuracy", "training_iteration"])

    result = tune.run(
        partial(para_lstm_dual, net=net,xtrain=xtrain,ytrain=ytrain,subject_train_ids=subject_train_ids,error_mode_train=error_mode_train,\
                gesture=gesture,win_len=win_len,stride=stride),
                      resources_per_trial={"cpu": 10, "gpu": 1},\
                          config=config, num_samples=num_samples, scheduler=scheduler,progress_reporter=reporter)
    best_trial = result.get_best_trial( "accuracy", "max", "last")
    print(best_trial.checkpoint.value)
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    paras=best_trial.config
    file_name='Siamese/gesture_specific_task_nonspecific/{}_para_{}/{}_para_{}_nested_{}_{}.p'.format(G, net_type,G,net_type,Task,fold)
    save_folder='Siamese/gesture_specific_task_nonspecific/{}_para_{}/'.format(G,net_type)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    pkl.dump(paras, open(file_name,"wb"))
    return paras





