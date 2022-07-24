import numpy as np
import pandas as pd
from tqdm import  tqdm





HLA_path = './data/NetMHCpan_test/HLA_list.txt'
HLA_list = pd.read_csv(HLA_path,header=None).values.tolist()
dataset = list()
for HLA in tqdm(HLA_list):
    HLA = HLA[0]
    
    #Load data
    data_dir = './data/NetMHCpan_test/'
    data_path = data_dir + HLA.replace(':','') + '.txt'
    pmhcData =pd.read_csv(data_path,sep = ' ',keep_default_na=False).values.tolist()
    
    #Fix HLA name
    if 'HLA-' in HLA:
        HLA = HLA.replace('HLA-A','HLA-A*')
        HLA = HLA.replace('HLA-B','HLA-B*')
        HLA = HLA.replace('HLA-C','HLA-C*') 
        
    for item in pmhcData:
        peptide = item[0]
        EL = item[1]
        
        dataset.append([peptide,HLA,EL])
        
#Save pmhcData to local
column=['peptide','MHC','EL']
output_dir = './data/EL_full_testingData.csv'
output = pd.DataFrame(columns=column,data=dataset)
output.to_csv(output_dir,index = None)

   