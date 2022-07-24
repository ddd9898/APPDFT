from models.baseline import Model_atten_score
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


symbol_dic_FASTA = {'X':0,'Y':1, 'S':2, 'M':3, 'R':4, 'E':5, 'I':6, 'N':7, 'V':8, 'G':9, 'L':10,
                    'D':11, 'T':12, 'W':13, 'H':14, 'K':15, 'A':16, 'F':17, 'Q':18, 'C':19, 'P':20,'#':21,'*':22}

AA_pos_dict = {'A':1, 'R':2, 'N':3, 'D':4, 'C':5, 'Q':6, 'E':7, 'G':8, 'H':9, 'I':10,
                            'L':11, 'K':12, 'M':13, 'F':14, 'P':15, 'S':16, 'T':17, 'W':18, 'Y':19, 'V':20}

def get_args():
    parser = argparse.ArgumentParser(description='Immune epitope Motif from APPDFT',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--MHC', dest='MHC', type=str, default='',
                        help='The MHC name',metavar='E')
    parser.add_argument('--MHCseq', dest='MHCseq', type=str, default='',
                        help='The MHC sequence',metavar='E')


    return parser.parse_args()

class randomPepData(Dataset):
    def __init__(self, data_path = './data/atten/randomPeptides_9mer.csv',mhcSeq = ''):
        super(randomPepData,self).__init__()

        #Load data file
        self.data = pd.read_csv(data_path).values.tolist()
        self.mhcSeq = mhcSeq
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        peptide = self.data[i][0]
        mhcSeq = self.mhcSeq

        #Get input
        ConcatSeq = '#' + mhcSeq + peptide.ljust(14, "*")
        ConcatSeq = np.asarray([symbol_dic_FASTA[char] for char in ConcatSeq])
        
        #To tensor
        ConcatSeq = torch.LongTensor(ConcatSeq)

        # return data
        return ConcatSeq

if __name__ == '__main__':
    
    #python APPDFT_motif.py --MHC HLA-A*11:01 --MHCseq YYAMYQENVAQTDVDTLYIIYRDYTWAAQAYRWY
    
    #Get argument parse
    args = get_args()
    MHC =  args.MHC
    MHCSeq = args.MHCseq
    
    #Init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Load Data
    testDataset = randomPepData(mhcSeq = MHCSeq)
    test_loader = DataLoader(testDataset, batch_size=2048, shuffle=False)


    model = Model_atten_score(modelfilename = 'Model-link_trainingData_fold0_index0_EL.model').to(device)

    
    attn_output_weights = torch.Tensor()
    scores_output = torch.Tensor()
    #Test 
    for data in test_loader:
        #Get input
        ConcatSeq = data.to(device)
    
        model.eval()
        with torch.no_grad():
            attn_output_weight,y_EL = model(ConcatSeq)
        
        attn_output_weights = torch.cat((attn_output_weights, attn_output_weight), 0)
        scores_output = torch.cat((scores_output, y_EL), 0)
        

    attn_output_weights = attn_output_weights.numpy()  #49*49
    scores_output = scores_output.numpy().flatten()


    atten_data = list()
    for heatmap_name in ['postive','negative']:
        title =  MHC + '(' + heatmap_name + ')'

        #Find top 1000
        TOP_NUM = 1000
        if heatmap_name == 'postive':
            top_indexs = np.argsort(1-scores_output)[:TOP_NUM]
        else:
            top_indexs = np.argsort(scores_output)[:TOP_NUM]
        top_scores = [scores_output[idx] for idx in top_indexs]
        top_attens = [attn_output_weights[idx] for idx in top_indexs]


        #Save attention scores
        atten_scores = list()
        for n in range(TOP_NUM):
            idx = top_indexs[n]
            peptide = testDataset.data[idx][0]
            HLA = MHC
            
            attn_weight = np.sum(top_attens[n],axis=0)[35:]
            
            attn_weight = attn_weight/np.sum(attn_weight)
            
            writeLine = [peptide,HLA]
            writeLine.extend(attn_weight)
            atten_scores.append(writeLine)

        heatMapData = np.zeros((20,9))
        for item in atten_scores:
            peptide = item[0]
            if 'X' in peptide:
                continue
            atten_scores = item[2:11]
            for col_index in range(len(peptide)):
                symbol = peptide[col_index]
                row_index = AA_pos_dict[symbol] - 1
                
                heatMapData[row_index,col_index] += atten_scores[col_index]
        atten_data.append(heatMapData)

        #Draw
        plt.figure(figsize=(5, 8), dpi=150)
        ax = sns.heatmap(heatMapData,cmap='hot', yticklabels=AA_pos_dict.keys()) #Greens_r,coolwarm
        ax.set_title(title)
        ax.set_ylabel('Amino-acid type of peptides')
        ax.set_xticklabels([str(item) for item in range(1,10)])
        plt.xticks(fontsize=8)
        plt.yticks(rotation=0,fontsize=8)

        # plt.pause(1)
        plt.savefig('./' + title.replace('*','').replace(':','') + '.jpg',bbox_inches = 'tight')  # 保存图片
        # plt.savefig('./' + title + '.pdf',bbox_inches = 'tight',format='pdf',transparent=True)  # 保存图片
        plt.clf()
        plt.close()


    heatMapData = atten_data[0] - atten_data[1]
    Max_value = np.max(np.abs(heatMapData))

    #Draw
    plt.figure(figsize=(5, 8), dpi=150)
    ax = sns.heatmap(heatMapData,cmap='icefire', yticklabels=AA_pos_dict.keys(),vmax=Max_value,vmin=-Max_value) #Greens_r,coolwarm
    ax.set_title(MHC)
    ax.set_ylabel('Amino-acid type of peptides')
    ax.set_xticklabels([str(item) for item in range(1,10)])
    plt.xticks(fontsize=8)
    plt.yticks(rotation=0,fontsize=8)

    # plt.pause(1)
    plt.savefig('./' + MHC.replace('*','').replace(':','') + '.jpg',bbox_inches = 'tight')  # 保存图片
    # plt.savefig('./' + MHC + '.pdf',bbox_inches = 'tight',format='pdf',transparent=True)  # 保存图片
    plt.clf()
    plt.close()

        








