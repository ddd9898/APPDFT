import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import random


symbol_dic_FASTA = {'X':0,'Y':1, 'S':2, 'M':3, 'R':4, 'E':5, 'I':6, 'N':7, 'V':8, 'G':9, 'L':10,
                    'D':11, 'T':12, 'W':13, 'H':14, 'K':15, 'A':16, 'F':17, 'Q':18, 'C':19, 'P':20,'#':21,'*':22}

    

class pmhcData(Dataset):
    def __init__(self, data_path = './data/EL_full_testingData.csv'):
        super(pmhcData,self).__init__()

        #Load data file
        self.data = pd.read_csv(data_path).values.tolist()
                
        #Load pseudo sequences
        pseudo_dir = './data/pseudoSequence(ELIM).csv' 
        temp = pd.read_csv(pseudo_dir).values.tolist()
        self.pseudoMHC_Dic = dict()
        for item in temp:
            MHCname,pseudoMHC = item[0],item[1]
            self.pseudoMHC_Dic[MHCname] = pseudoMHC
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        peptide = self.data[i][0]
        mhcName = self.data[i][1]
        mhcSeq = self.pseudoMHC_Dic[mhcName]
        gt = self.data[i][2]

        #Get input
        ConcatSeq = '#' + mhcSeq + peptide.ljust(14, "*")
        ConcatSeq = np.asarray([symbol_dic_FASTA[char] for char in ConcatSeq])
        
        #Get output
        gt = float(gt)
        
        #To tensor
        ConcatSeq = torch.LongTensor(ConcatSeq)
        gt = torch.FloatTensor([gt])

        # return data
        return ConcatSeq,gt


class BAorELIMData(Dataset):
    def __init__(self,pretrain = True,valfold = 0,val = False):
        super(BAorELIMData,self).__init__()
        

        #Load data file
        self.pretrain = pretrain
        self.data = list()
        fivefold_val_flags = pd.read_csv("./data/fivefold_val_flags(ELIM).csv").values.tolist()
        

        if pretrain: #Load BA data only
            fullData = pd.read_csv('./data/BA_full_trainingData.csv').values.tolist()
            self.data = fullData

        else: #Load EL and IM data
            #Choose the specific fold
            fullData = pd.read_csv('./data/ELIM_full_trainingData.csv').values.tolist()
            val_flags = [item[valfold] for item in fivefold_val_flags]
            for n in range(len(val_flags)):
                val_flag = val_flags[n]
          
                if val and (val_flag==1): #val
                    self.data.append(fullData[n])
                elif (not val) and (val_flag==0): #train
                    self.data.append(fullData[n])
            
            # if (not val):
            #     self.data = dataDownSample(self.data)
                
        
            # #Data amplification
            # if (not val):
            #     self.data = dataAmplification(self.data)
                
        #Load pseudo sequences
        pseudo_dir = './data/pseudoSequence(ELIM).csv' 
        temp = pd.read_csv(pseudo_dir).values.tolist()
        self.pseudoMHC_Dic = dict()
        for item in temp:
            MHCname,pseudoMHC = item[0],item[1]
            self.pseudoMHC_Dic[MHCname] = pseudoMHC
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        peptide = self.data[i][0]
        mhcName = self.data[i][1]
        mhcSeq = self.pseudoMHC_Dic[mhcName]
        if self.pretrain:
            g_BA = self.data[i][2]
        else:
            g_EL = self.data[i][2]
            g_IM = self.data[i][3]

        #Get input
        ConcatSeq = '#' + mhcSeq + peptide.ljust(14, "*")
        ConcatSeq = np.asarray([symbol_dic_FASTA[char] for char in ConcatSeq])
        ConcatSeq = torch.LongTensor(ConcatSeq) #To tensor
        
        #Get output
        if self.pretrain:
            g_BA = float(g_BA)
            g_BA = torch.FloatTensor([g_BA]) #To tensor
        else:
            g_EL = float(g_EL)
            g_IM = float(g_IM)
            
            #To tensor
            g_EL = torch.FloatTensor([g_EL]) 
            g_IM = torch.FloatTensor([g_IM])


        # return data
        if self.pretrain:
            return ConcatSeq,g_BA
        else:
            return ConcatSeq,g_EL,g_IM

class BAorELorELIMData(Dataset):
    def __init__(self,pretrain = True,valfold = 0,val = False,EL_flag = True):
        super(BAorELorELIMData,self).__init__()
        

        #Load data file
        self.pretrain = pretrain
        self.EL_flag = EL_flag
        self.data = list()
        

        if pretrain: #Load BA data only
            fullData = pd.read_csv('./data/BA_full_trainingData.csv').values.tolist()
            self.data = fullData

        else: #Load EL or ELIM data
            if EL_flag:
                fivefold_val_flags = pd.read_csv("./data/fivefold_val_flags(ELSA).csv").values.tolist()
                #Choose the specific fold
                fullData = pd.read_csv('./data/ELSA_full_trainingData.csv').values.tolist()
                val_flags = [item[valfold] for item in fivefold_val_flags]
                for n in range(len(val_flags)):
                    val_flag = val_flags[n]
                    if val and (val_flag==1): #val
                        self.data.append(fullData[n])
                    elif (not val) and (val_flag==0): #train
                        self.data.append(fullData[n])
            else:
                fivefold_val_flags = pd.read_csv("./data/fivefold_val_flags(ELIM).csv").values.tolist()
                #Choose the specific fold
                fullData = pd.read_csv('./data/ELIM_full_trainingData.csv').values.tolist()
                val_flags = [item[valfold] for item in fivefold_val_flags]
                for n in range(len(val_flags)):
                    val_flag = val_flags[n]
                    if val and (val_flag==1): #val
                        self.data.append(fullData[n])
                    elif (not val) and (val_flag==0): #train
                        self.data.append(fullData[n])
                
        #Load pseudo sequences
        pseudo_dir = './data/pseudoSequence(ELIM).csv' 
        temp = pd.read_csv(pseudo_dir).values.tolist()
        self.pseudoMHC_Dic = dict()
        for item in temp:
            MHCname,pseudoMHC = item[0],item[1]
            self.pseudoMHC_Dic[MHCname] = pseudoMHC
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        peptide = self.data[i][0]
        mhcName = self.data[i][1]
        mhcSeq = self.pseudoMHC_Dic[mhcName]
        if self.pretrain or self.EL_flag:
            g = self.data[i][2]
        else:
            g = self.data[i][2]
            g_IM = self.data[i][3]
            if not np.isnan(g_IM):
                g = g_IM

        #Get input
        ConcatSeq = '#' + mhcSeq + peptide.ljust(14, "*")
        ConcatSeq = np.asarray([symbol_dic_FASTA[char] for char in ConcatSeq])
        ConcatSeq = torch.LongTensor(ConcatSeq) #To tensor
        
        #Get output
        g = float(g)
        g = torch.FloatTensor([g]) #To tensor


        # return data
        return ConcatSeq,g

if __name__ == '__main__':
    
    # testDataset = pmhcData()
    trainDataset = BAorELIMData()
    from torch.utils.data import DataLoader
    train_loader = DataLoader(trainDataset, batch_size=10, shuffle=True)
    for item in train_loader:
        a = 1
    b = 1


