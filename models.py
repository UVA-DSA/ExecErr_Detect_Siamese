import torch.nn as nn
import torch
import torch.nn.functional as F

class TimeseriesNet(nn.Module):
    def __init__(self):
        super(TimeseriesNet,self).__init__()
        self.stage_1_conv = nn.Conv1d(26,64,kernel_size=5,stride=1)
        self.stage_1_pool = nn.MaxPool1d(2,2)
        self.stage_1_drop = nn.Dropout(p=0.2)
        self.stage_1_norm = nn.BatchNorm1d(64)
        self.stage_2_conv = nn.Conv1d(64,128,kernel_size=5,stride=1)
        self.stage_2_pool = nn.MaxPool1d(2,2)
        self.stage_2_drop = nn.Dropout(p=0.2)
        self.stage_2_norm = nn.BatchNorm1d(128)

        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(512,256)
        self.linear2 = nn.Linear(256,32)
        self.linear3 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16,1)

        
        self.initialize_weights()
        
        #barch normalization
        #self.stage_2_conv = nn.Conv1d()
    def forward(self,l):
        l = F.relu(self.stage_1_conv(l))
        l = self.stage_1_pool(l)
        l = self.stage_1_drop(l)
        l = self.stage_1_norm(l)
        l = F.relu(self.stage_2_conv(l))
        l = self.stage_2_pool(l)
        l = self.stage_2_drop(l)
        l = self.stage_2_norm(l)
        l = self.flat(l)
        l = F.relu(self.linear1(l))
        l = F.relu(self.linear2(l))
        l = F.relu(self.linear3(l))
        l = self.linear4(l)
        
        return l
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)

class TimeseriesNet_lstm(nn.Module):
    def __init__(self):
        super(TimeseriesNet_lstm,self).__init__()

       #LSTM(INPUTFEATURES, HIDDEN_SIZE, NUM_LAYERS, BATCH_FIRST=TRUE)
        self.festures=26
        self.seq_len =60
        self.layer_dim =1
        self.lstm1 = nn.LSTM(26,512, dropout=0, num_layers=self.layer_dim,batch_first=True)


        self.lstm2 = nn.LSTM(512, 128, dropout=0, num_layers=self.layer_dim,batch_first=True)
        self.lstm3 = nn.LSTM(128, 64, dropout=0 , num_layers=self.layer_dim,batch_first=True)
        # self.norm = nn.BatchNorm1d(30)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(p=0.55)
        self.linear1 = nn.Linear(1920,960)
        self.linear2 = nn.Linear(960,480)
        self.linear3 = nn.Linear(480,16)
        self.linear4 = nn.Linear(16,1)
        
        self.initialize_weights()

    def forward(self,l):
        l=l.transpose(1,2).contiguous()
        
        h0_l = torch.randn(self.layer_dim,l.size(0),512,device=torch.device("cuda:0")).requires_grad_()
        c0_l = torch.randn(self.layer_dim,l.size(0),512,device=torch.device("cuda:0")).requires_grad_()
        lstm,(hn,cn) = self.lstm1(l,(h0_l.detach(), c0_l.detach()))
        lstm = F.relu(lstm)
        
        h1_l = torch.randn(self.layer_dim,lstm.size(0),128,device=torch.device("cuda:0")).requires_grad_()
        c1_l = torch.randn(self.layer_dim,lstm.size(0),128,device=torch.device("cuda:0")).requires_grad_()
        lstm,(hn,cn) = self.lstm2(lstm,(h1_l.detach(), c1_l.detach()))
        lstm = F.relu(lstm)
        
        h2_l = torch.randn(self.layer_dim,lstm.size(0),64,device=torch.device("cuda:0")).requires_grad_()
        c2_l = torch.randn(self.layer_dim,lstm.size(0),64,device=torch.device("cuda:0")).requires_grad_()
        lstm,(hn,cn) = self.lstm3(lstm,(h2_l.detach(), c2_l.detach()))
        lstm = F.relu(lstm)
        # lstm = self.norm(lstm)
        lstm = self.flat(lstm)
        
        
        lstm = F.relu(self.linear1(lstm))
        lstm= self.drop(lstm)
        lstm = F.relu(self.linear2(lstm))
        lstm= self.drop(lstm)
        lstm = F.relu(self.linear3(lstm))
        lstm = self.linear4(lstm)


        
        return lstm
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)


class TimeseriesNet_dural_lstm(nn.Module):
    def __init__(self):
        super(TimeseriesNet_dural_lstm,self).__init__()

       #LSTM(INPUTFEATURES, HIDDEN_SIZE, NUM_LAYERS, BATCH_FIRST=TRUE)
        self.festures=26
        self.seq_len =60
        self.layer_dim =1
        self.lstm1 = nn.LSTM(26,512, dropout=0, num_layers=self.layer_dim,batch_first=True)


        self.lstm2 = nn.LSTM(512, 128, dropout=0, num_layers=self.layer_dim,batch_first=True)
        self.lstm3 = nn.LSTM(128, 64, dropout=0, num_layers=self.layer_dim,batch_first=True)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(p=0.55)
        self.linear1 = nn.Linear(1920,960)
        self.linear2 = nn.Linear(960,480)

        self.linear6 = nn.Linear(480,16)
        self.linear7 = nn.Linear(16,1)
        
        self.initialize_weights()

    def forward(self,l,l1):
        l=l.transpose(1,2).contiguous()
        l1=l1.transpose(1,2).contiguous()
        
        h0_l = torch.randn(self.layer_dim,l.size(0),512,device=torch.device("cuda:0")).requires_grad_()
        c0_l = torch.randn(self.layer_dim,l.size(0),512,device=torch.device("cuda:0")).requires_grad_()
        lstm,(hn,cn) = self.lstm1(l,(h0_l.detach(), c0_l.detach()))
        lstm = F.relu(lstm)
        
        h1_l = torch.randn(self.layer_dim,lstm.size(0),128,device=torch.device("cuda:0")).requires_grad_()
        c1_l = torch.randn(self.layer_dim,lstm.size(0),128,device=torch.device("cuda:0")).requires_grad_()
        lstm,(hn,cn) = self.lstm2(lstm,(h1_l.detach(), c1_l.detach()))
        lstm = F.relu(lstm)
        
        h2_l = torch.randn(self.layer_dim,lstm.size(0),64,device=torch.device("cuda:0")).requires_grad_()
        c2_l = torch.randn(self.layer_dim,lstm.size(0),64,device=torch.device("cuda:0")).requires_grad_()
        lstm,(hn,cn) = self.lstm3(lstm,(h2_l.detach(), c2_l.detach()))
        lstm = F.relu(lstm)
        lstm = self.flat(lstm)
        
        
        h0_l1 = torch.randn(self.layer_dim,l.size(0),512,device=torch.device("cuda:0")).requires_grad_()
        c0_l1 = torch.randn(self.layer_dim,l.size(0),512,device=torch.device("cuda:0")).requires_grad_()
        lstm1,(hn1,cn1) = self.lstm1(l1,(h0_l1.detach(), c0_l1.detach()))
        lstm1 = F.relu(lstm1)
        
        h1_l1 = torch.randn(self.layer_dim,lstm.size(0),128,device=torch.device("cuda:0")).requires_grad_()
        c1_l1 = torch.randn(self.layer_dim,lstm.size(0),128,device=torch.device("cuda:0")).requires_grad_()
        lstm1,(hn1,cn1) = self.lstm2(lstm1,(h1_l1.detach(), c1_l1.detach()))
        lstm1 = F.relu(lstm1)
        
        h2_l1 = torch.randn(self.layer_dim,lstm.size(0),64,device=torch.device("cuda:0")).requires_grad_()
        c2_l1 = torch.randn(self.layer_dim,lstm.size(0),64,device=torch.device("cuda:0")).requires_grad_()
        lstm1,(hn1,cn1) = self.lstm3(lstm1,(h2_l1.detach(), c2_l1.detach()))
        lstm1 = F.relu(lstm1)
     
        lstm1 = self.flat(lstm1)
        
        final= torch.abs(torch.sub(lstm,lstm1))
        
        final = F.relu(self.linear1(final))
    
        final = F.relu(self.linear2(final))
   
        final = F.relu(self.linear6(final))
        final = self.linear7(final)
        
        return final
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)




class TimeseriesNet_dural_cnn(nn.Module):
    def __init__(self,l1,l2):
        super(TimeseriesNet_dural_cnn,self).__init__()
        self.stage_1_conv = nn.Conv1d(26,64,kernel_size=5,stride=1)
        self.stage_1_pool = nn.MaxPool1d(2,2)
        self.stage_1_drop = nn.Dropout(p=0.2)
        self.stage_1_norm = nn.BatchNorm1d(64)
        self.stage_2_conv = nn.Conv1d(64,128,kernel_size=5,stride=1)
        self.stage_2_pool = nn.MaxPool1d(2,2)
        self.stage_2_drop = nn.Dropout(p=0.2)
        self.stage_2_norm = nn.BatchNorm1d(128)

        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(512,l1)
        self.linear2 = nn.Linear(l1,l2)
        self.linear3 = nn.Linear(l2,32)
        self.linear4 = nn.Linear(32,1)

        
        self.initialize_weights()
        
        #barch normalization
        #self.stage_2_conv = nn.Conv1d()
    def forward(self,l,l2):
        l = F.relu(self.stage_1_conv(l))
        l = self.stage_1_pool(l)
        l = self.stage_1_drop(l)
        l = self.stage_1_norm(l)
        l = F.relu(self.stage_2_conv(l))
        l = self.stage_2_pool(l)
        l = self.stage_2_drop(l)
        l = self.stage_2_norm(l)
        l = self.flat(l)

        
        l2 = F.relu(self.stage_1_conv(l2))
        l2= self.stage_1_pool(l2)
        l2 = self.stage_1_drop(l2)
        l2 = self.stage_1_norm(l2)
        l2 = F.relu(self.stage_2_conv(l2))
        l2 = self.stage_2_pool(l2)
        l2 = self.stage_2_drop(l2)
        l2 = self.stage_2_norm(l2)
        l2 = self.flat(l2)
        
        final= torch.abs(torch.sub(l2,l))
        final = F.relu(self.linear1(final))
        final = F.relu(self.linear2(final))
        final = F.relu(self.linear3(final))
        

        final = self.linear4(final)
        
        return final
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)
