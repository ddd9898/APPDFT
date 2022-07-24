import numpy as np
import pandas as pd

###########Load NetMHCpan4.1 BA training data#############
data_dir = './data/NetMHCpan_train/'
NetMHCpanData = list()
folds_names = ['c000_ba','c001_ba','c002_ba','c003_ba','c004_ba']
for fold in folds_names:
    data_path = data_dir + fold
    temp =pd.read_csv(data_path,sep = ' ',header=None,keep_default_na=False,engine='python').values.tolist()
    NetMHCpanData.extend(temp)
    
print("Data size of binding affinity(BA) data NetMHCpan4.1 is {}".format(len(NetMHCpanData)))

#Create pmhcData_BA
pmhcData_BA = list()
mhc_ba_list = list()
for n,item in enumerate(NetMHCpanData):
    peptide = item[0]
    y = item[1]
    mhcName = item[2]
    if 'HLA-' in mhcName:
        mhcName = mhcName.replace('HLA-A','HLA-A*')
        mhcName = mhcName.replace('HLA-B','HLA-B*')
        mhcName = mhcName.replace('HLA-C','HLA-C*') 

    if len(peptide)>=8 and len(peptide)<=14:
        if mhcName not in mhc_ba_list:
            mhc_ba_list.append(mhcName)
        writeLine = [peptide,mhcName,y]
        pmhcData_BA.append(writeLine)
print('{} MHCs in BA data'.format(len(mhc_ba_list)))
print("After filtering samples, Data size of binding affinity(BA) data NetMHCpan4.1 is {}".format(len(pmhcData_BA)))


#Save to local
column=['peptide','MHC','Binding affinity']
# column=['peptide','HLA','CDR3','immunogenicity']
output_dir = './data/BA_full_trainingData.csv'
output = pd.DataFrame(columns=column,data=pmhcData_BA)
output.to_csv(output_dir,index = None)