from models.baseline import Model
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
import sys

symbol_dic_FASTA = {'X':0,'Y':1, 'S':2, 'M':3, 'R':4, 'E':5, 'I':6, 'N':7, 'V':8, 'G':9, 'L':10,
                    'D':11, 'T':12, 'W':13, 'H':14, 'K':15, 'A':16, 'F':17, 'Q':18, 'C':19, 'P':20,'#':21,'*':22}


def get_args():
    parser = argparse.ArgumentParser(description='The application of APPDFT',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', dest='input', type=str, default='',
                        help='The input file',metavar='E')
    parser.add_argument('--output', dest='output', type=str, default='',
                        help='The output file',metavar='E')


    return parser.parse_args()
 

class new_dataset(Dataset):
    def __init__(self, data_path = './test.csv'):
        super(new_dataset,self).__init__()

        #Load data file
        self.data = pd.read_csv(data_path).values.tolist()
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        peptide = self.data[i][0]
        mhcName = self.data[i][1]
        mhcSeq = self.data[i][2]

        #Get input
        ConcatSeq = '#' + mhcSeq + peptide.ljust(14, "*")
        ConcatSeq = np.asarray([symbol_dic_FASTA[char] for char in ConcatSeq])
        
        #To tensor
        ConcatSeq = torch.LongTensor(ConcatSeq)

        # return data
        return ConcatSeq

if __name__ == '__main__':
    
    #python APPDFT.py --input test.csv --output output.csv
    
    #Get argument parse
    args = get_args()
    input_file = './' + args.input
    output_file = './' + args.output
    
    if not os.path.exists(input_file):
        print("Not find {}".format(input_file))
        sys.exit()

    #Init 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testDataset = new_dataset(data_path = args.input)
    test_loader = DataLoader(testDataset, batch_size=2048, shuffle=False)

    model_dir = './output/models/'
    model_basename = 'Model-link_trainingData_fold*_index0_EL.model'
    
    


    models = []
    for n in range(5):
        model = Model(num_encoder_layers = 2).to(device)
        model_name = model_basename.replace('*', str(n))
        model_path = model_dir + model_name
        # weights = torch.load(model_path)
        weights = torch.load(model_path,map_location=torch.device('cpu'))
        model.load_state_dict(weights)

        models.append(model)

    #Test 
    total_preds_EL = torch.Tensor()
    for data in tqdm(test_loader):
        #Get input
        ConcatSeq = data.to(device)

        #Calculate output
        output_ave_EL = 0
        for model in models:
            model.eval()
            with torch.no_grad():
                y_EL,_ = model(ConcatSeq,link = True)
                y_EL = y_EL.cpu()
                output_ave_EL = output_ave_EL + y_EL
        output_ave_EL = output_ave_EL / len(models)
        total_preds_EL = torch.cat((total_preds_EL, output_ave_EL), 0)

    P_EL = total_preds_EL.numpy().flatten()

    #Save to local
    column=['peptide','HLA','score']
    results = list()
    for n in range(len(P_EL)):
        results.append([testDataset.data[n][0],testDataset.data[n][1],P_EL[n]])
        
    output = pd.DataFrame(columns=column,data=results)
    output.to_csv(output_file,index = None)
