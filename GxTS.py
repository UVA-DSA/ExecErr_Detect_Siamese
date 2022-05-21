from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle as pkl
import pandas as pd
import time
from util import find_para_dual,find_para_single, TimeseriesData, TimeseriesPairs_test_v2,TimeseriesPairs_train_v2,load_data
from models import TimeseriesNet_dural_cnn,TimeseriesNet_dural_lstm,TimeseriesNet_cnn,TimeseriesNet_lstm




type="dual"
Tasks = ["Suturing","NeedlePassing"]

if type =="dual":
    m =[TimeseriesNet_dural_cnn(),TimeseriesNet_dural_lstm()]
    net_types=['dual_cnn','dual_lstm'] #
    for ith,net_type in enumerate(net_types):
        for Task in Tasks:
            
            if Task=="Suturing":
                all_g= ["G1","G2","G3","G4","G6","G8","G9"]
                data_dir=['',""] # todo fill in the data dir for suturing
            else: 
                all_g=["G1","G2","G3","G4","G6"]
                data_dir='' # todo fill in the data dir for needle passing
            F1_mean=np.empty((0,1),dtype=float)
            F1_std=np.empty((0,1),dtype=float)
            
            F1_mean=np.empty((0,1),dtype=float)
            F1_std=np.empty((0,1),dtype=float)
            F1_v_mean=np.empty((0,1),dtype=float)
            F1_v_std=np.empty((0,1),dtype=float)
            time_run=np.empty((0,1),dtype=float)

            gesture=all_g
            win_len=30
            stride=20
            
            all_x,all_y,error_mode,ids = load_data(gesture,data_dir)
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
            
            for fold in range(len(unique_ids)):
                # if fold !=1 : continue
                
                # model defintion here
                model=TimeseriesNet_dural_cnn()
                device = "cpu"
                if torch.cuda.is_available():
                    device = "cuda:0"
                model.to(device)
                
                train_ids,test_ids = fold_data[fold]
                xtrain=all_x[train_ids]
                ytrain=all_y[train_ids]
                xtest=all_x[test_ids]
                ytest=all_y[test_ids]
                error_type_test=error_mode[test_ids,:]
                error_mode_train=error_mode[train_ids,:]
                subject_train_ids=ids[train_ids]
                
                cur_para=find_para_dual(G,model,xtrain,ytrain,subject_train_ids,error_mode_train,Task,fold,win_len,stride,net_type)
                config = {
                "l1": 8,
                "l2": 32,
                "lr":cur_para['lr'],
                "batch_size":cur_para['batch_size'],
                'epoch':cur_para['epoch']}
                optimizer= torch.optim.Adam(model.parameters(),lr=config["lr"])
                traindata = TimeseriesPairs_train_v2( win_len,stride,xtrain,ytrain)
                trainloader = DataLoader(dataset=traindata, batch_size=int(config["batch_size"])\
                                        ,shuffle=True,num_workers=0)
            
                w = [sum(np.array(ytrain)!='err')/sum(np.array(ytrain)=='err')]
                class_weight=torch.FloatTensor(w).to(device)
                criterion = nn.BCEWithLogitsLoss(class_weight)
                
                testdata = TimeseriesPairs_test_v2( win_len,stride,xtrain,ytrain,xtest,ytest)
                testloader = DataLoader(dataset=testdata, batch_size=int(config["batch_size"])\
                                        ,shuffle=False,num_workers=0)
                for n in range(config['epoch']):
                    model.train()
                    
                    for i, data in enumerate(trainloader,0):
                        local_batch_L,local_batch_L2,local_y = data
                        local_batch_L,local_batch_L2,local_y = local_batch_L.to(device),local_batch_L2.to(device),\
                        local_y.to(device)
                        if local_batch_L.shape[0]!=config["batch_size"]:continue
                    # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward + backward + optimize
                        outputs =torch.squeeze(model(local_batch_L,local_batch_L2))
                        #print(f"input size{local_batch_L.shape}")
                        #print(f"y size{local_y.shape}")
                        loss = criterion(outputs.view(-1),local_y.view(-1))
                        loss.backward()
                        optimizer.step()
                    print('epoch={}'.format(n))    
                print('finish train at fold={}'.format(fold))
                save_folder_model='Siamese/GxTS/para_{}/'.format(net_type)
                if not os.path.exists(save_folder_model):
                    os.makedirs(save_folder_model,exist_ok=True)
                model_path = 'Siamese/GxTS/para_{}/{}_model_all_{}.pth'.format(net_type,net_type,fold)
                torch.save(model.state_dict(), model_path)
                # evaluate this fold's perforamnce in terms of F1 score
                val_loss = 0.0
                val_steps = 0
                total = 0
                correct = 0
                total = 0
                TP=0
                FP=0
                FN=0
                result=[]
                y_expected=[]
                y_idx=np.empty((0,2), int)  # for evaluating per testing window
                model.eval()
                for i, data in enumerate(testloader,0):
                    with torch.no_grad():
                        
                        local_batch_L, local_batch_L2,local_y,idx = data
                        local_batch_L,local_batch_L2,local_y = local_batch_L.to(device),\
                            local_batch_L2.to(device),local_y.to(device)

                        if local_y.view(-1).cpu().numpy().size==0: continue
                        outputs =torch.squeeze(model(local_batch_L,local_batch_L2))
                        
                        outputs_val=outputs>0
                        correct+=(outputs_val.view(-1) == local_y.view(-1)).sum().cpu().numpy()
                        total +=outputs.cpu().numpy().size

                        
                        result.extend(outputs_val.view(-1).cpu().numpy())
                        y_expected.extend(local_y.view(-1).cpu().numpy())
                        y_idx=np.vstack((y_idx,idx))
            
                # calculating the voting result
                result = np.array(result)
                y_expected = np.array(y_expected)
                voting=[] # store the result for each testing window
                y_truth=[]
                errortypes=np.empty((0,5), int)
                for idx_r in np.unique(y_idx,axis=0):
                    # get the locations of the corresponding test windows
                    trial,window=idx_r
                    trial=int(trial)
                    mask = np.logical_and(y_idx[:,0]==trial,y_idx[:,1]==window)
                    y_result_w=result[mask]
                    y_expected_w=y_expected[mask][0]
                    vote = sum(y_result_w)/len(y_result_w)
                    if vote>0.5:
                        voting.append(1)
                    else:
                        voting.append(0)
                    y_truth.append(y_expected_w)
                    errortypes=np.vstack((errortypes,error_type_test[trial,:]))
                    
                try:
                    TN_v, FP_v, FN_v, TP_v =confusion_matrix(y_truth,voting).ravel()
                    aa=[TN_v, FP_v, FN_v, TP_v]
                    TN_v, FP_v, FN_v, TP_v=[0.001 if a==0 else a for a in aa]
                    precision_v=TP_v/(TP_v+FP_v)
                    recall_v=TP_v/(TP_v+FN_v)
                    F1_v=2*precision_v*recall_v/(precision_v+recall_v)  
                except ValueError:
                    F1_v=0
                
                try:
                    TN,FP,FN,TP=confusion_matrix(y_expected,result).ravel()
                    aa=[TN, FP, FN, TP]
                    TN, FP, FN, TP=[0.001 if a==0 else a for a in aa]
                    precision=TP/(TP+FP)
                    recall=TP/(TP+FN)
                    accuracy=correct/total
                    F1=2*precision*recall/(precision+recall)

                except ValueError:
                    F1=0
                F1scores[fold]=[F1,F1_v]
                precision_result[fold]=[precision,precision_v]
                recall_result[fold]=[recall,recall_v]
                predicted_result[fold]=voting
                expected_result[fold]=y_truth
                error_types[fold]=errortypes
                # print('fold={}'.format(fold))
                
                        
            F_val=np.empty((0,1))
            F_v_val=np.empty((0,1))           
            for i in F1scores.keys(): 
                F_val=np.append(F_val,F1scores[i][0])
                F_v_val=np.append(F_v_val,F1scores[i][1])
            
            
            F_mean=np.mean(F_val)
            F_std=np.std(F_val)
            F_v_mean=np.mean(F_v_val)
            F_v_std=np.std(F_v_val)
            # append to the list for making the table
            F1_mean=np.append(F1_mean,F_mean)
            F1_std=np.append(F1_std,F_std)
            F1_v_mean=np.append(F1_v_mean,F_v_mean)
            F1_v_std=np.append(F1_v_std,F_v_std)
            # dump the dictionary into binary file
            AllD=[F1scores, precision_result, recall_result, predicted_result,expected_result,error_types]
            save_folder='Siamese/GxTS/{}'.format(net_type)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            file_name='Siamese/GxTS/{}/result_lstm_{}.p'.format(net_type, Task)
            pkl.dump(AllD, open(file_name,"wb"))
            toc = time.perf_counter()
        

        tb={'F1_mean': F1_mean, 'F1_std':F1_std, 'F1_v_mean':F1_v_mean,'F1_v_std':F1_v_std,'time':time_run}
        df=pd.DataFrame(tb,index=all_g)

        df.to_csv('Siamese/GxTS/{}_input_F1_mean.csv'.format(net_type))

else:
    net_types=['cnn','lstm']
    m =[TimeseriesNet_cnn(),TimeseriesNet_lstm()]
    for ith,net_type in enumerate(net_types):
        for Task in Tasks:
            
            if Task=="Suturing":
                all_g= ["G1","G2","G3","G4","G6","G8","G9"]
                data_dir='' # todo fill in the data dir for suturing
            else: 
                all_g=["G1","G2","G3","G4","G6"]
                data_dir='' # todo fill in the data dirfor needle passing
        
            F_means=[]
            F_stds=[]
            F1_mean=np.empty((0,1),dtype=float)
            F1_std=np.empty((0,1),dtype=float)    
            gesture=all_g
            win_len=30
            stride=20
            test_stride=20
            # numOfEpochs=10
            all_x,all_y,error_mode,ids = load_data(all_g,data_dir)
            ids=np.array([ i[0] for i in ids])
            unique_ids=np.unique(ids)
            fold_data={}
            for i,idx in enumerate(unique_ids):
                test_loc=np.where(ids==idx)[0]
                train_loc=np.where(ids!=idx)[0]
                fold_data[i]=[train_loc,test_loc]
                
            F1scores={}
            precision_results={}
            recall_results={}
            predicted_result={}
            error_types={}
            expected_result={}
            
            
            model = m[ith]
            for fold in range(len(unique_ids)):
                
                # if fold !=3 : continue
    
                device = "cpu"
                if torch.cuda.is_available():
                    device = "cuda:0"
                model.to(device)
                
                train_ids,test_ids = fold_data[fold]
                xtrain=all_x[train_ids]
                ytrain=all_y[train_ids]
                xtest=all_x[test_ids]
                ytest=all_y[test_ids]
                error_type_test=error_mode[test_ids,:]
                err_type_train=error_mode[train_ids,:]
                subject_train_ids=ids[train_ids]
                Task='all'
                cur_para =  pkl.load(open('para_{}/para_{}_nested_all_{}.p'.format(net_type,net_type,fold),'rb'))
                find_para_single(G,model,xtrain,ytrain,subject_train_ids,err_type_train,Task,fold,win_len,stride,net_type)
                config = {
            "lr":cur_para['lr'],
            "batch_size": cur_para['batch_size'],
            "epoch":cur_para['epoch']}
                optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"])
                traindata = TimeseriesData(xtrain,ytrain,error_mode[train_ids,:] , win_len=win_len, stride=stride)
                trainloader = DataLoader(dataset=traindata, batch_size=int(config["batch_size"])\
                                        ,shuffle=True,num_workers=0)
                w = [sum(np.array(ytrain)!='err')/sum(np.array(ytrain)=='err')]
                class_weight=torch.FloatTensor(w).to(device)
                criterion = nn.BCEWithLogitsLoss(class_weight)
                for n in range(config["epoch"]):
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
                        #print(f"input size{local_batch_L.shape}")
                        #print(f"y size{local_y.shape}")
                        loss = criterion(outputs.view(-1),local_y.view(-1))
                        loss.backward()
                        optimizer.step()
                save_folder_model='Siamese/GxTS/para_{}/'.format(net_type)
                if not os.path.exists(save_folder_model):
                    os.makedirs(save_folder_model,exist_ok=True)
                model_path = 'Siamese/GxTS/para_{}/{}_model_all_{}.pth'.format(net_type, net_type,fold)
                torch.save(model.state_dict(), model_path)
                        
                result=[]
                y_expected=[]
                y_error_type=np.empty((0,5), int)  # for evaluating per testing window
            
                testdata = TimeseriesData(xtest,ytest,error_type_test,win_len=win_len, stride=stride)
                testloader = DataLoader(dataset=testdata, batch_size=int(config["batch_size"])\
                                            ,shuffle=False,num_workers=8)
                        
                # evaluate this fold's perforamnce in terms of F1 score
                val_loss = 0.0
                val_steps = 0
                total = 0
                correct = 0
                total = 0
                TP=0
                FP=0
                FN=0
    
                model.eval()
                testloader2 = DataLoader(dataset=testdata, batch_size=int(config["batch_size"])\
                                            ,shuffle=False,num_workers=8)
                
                for i, data in enumerate(testloader,0):
                    with torch.no_grad():
                        
                        local_batch_L, local_y,error_type = data
                        local_batch_L, local_y = local_batch_L.to(device),\
                            local_y.to(device) 
                        # if local_y.dim() == 0: continue
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
                precision_results[fold]=precision
                recall_results[fold]=recall
                expected_result[fold]=y_expected
                error_types[fold]=y_error_type
            
            
            F_val=np.empty((0,1))
            R_val=np.empty((0,1))
            P_val=np.empty((0,1))
                
            for i in F1scores.keys(): 
                F_val=np.append(F_val,F1scores[i])
                R_val=np.append(F_val,recall_results[i])
                P_val=np.append(F_val,precision_results[i])
                
            F_mean=np.mean(F_val)
            F_means.append(F_mean)
            F_std=np.std(F_val)  
            F_stds.append(F_std)# append to the list for making the table
        
            # dump the dictionary into binary file
            AllD=[F1scores, precision_results, recall_results, predicted_result,expected_result,error_types]
            save_folder='Siamese/GxTS/'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        
            file_name='Siamese/GxTS/result_{}.p'.format(net_type)
            pkl.dump(AllD, open(file_name,"wb"))
            print('finish_{}'.format(G))
        tb={'F1_mean': F_means, 'F1_std':F_stds}
        df=pd.DataFrame(tb,index=all_g)
        df.to_csv('Siamese/GxTS/{}_input_F1_mean.csv'.format(net_type))
        print('finish{}'.format(Task))










