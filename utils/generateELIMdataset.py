import numpy as np
import pandas as pd



###########Load DeepHLApan IM training data#############
DeepHLApanData = pd.read_excel("./data/DeepHLApan/Table_4.XLSX").values.tolist()
print("Data size of DeepHLApan IM training data is {}".format(len(DeepHLApanData)))

#Create pmhcData_IM
pmhcData_IM = list()
mhc_im_list = list()
for n in range(2,len(DeepHLApanData)):
    item = DeepHLApanData[n]
    peptide = item[1]
    mhcName = item[0]
    IM = item[2]
    if IM == 1:
        EL = 1
    else:
        EL = 'NA'
    if 'HLA-' in mhcName:
        mhcName = mhcName.replace('HLA-A','HLA-A*')
        mhcName = mhcName.replace('HLA-B','HLA-B*')
        mhcName = mhcName.replace('HLA-C','HLA-C*')

    if len(peptide)>=8 and len(peptide)<=14:
        if mhcName not in mhc_im_list:
            mhc_im_list.append(mhcName)
        writeLine = [peptide,mhcName,EL,IM]
        pmhcData_IM.append(writeLine)
print('{} MHCs in IM data'.format(len(mhc_im_list)))
print("After filtering samples, Data size of DeepHLApan IM training data is {}".format(len(pmhcData_IM)))

#Save pmhcData to local
column=['peptide','MHC','immunognicity']
output_dir = './data/IM_full_trainingData.csv'
output = pd.DataFrame(columns=column,data=[[item[0],item[1],item[3]]for item in pmhcData_IM])
output.to_csv(output_dir,index = None)

###########Load NetMHCpan4.1 EL training data#############
data_dir = './data/NetMHCpan_train/'
NetMHCpanData = list()
folds_names = ['c000_el','c001_el','c002_el','c003_el','c004_el']
for fold in folds_names:
    data_path = data_dir + fold
    temp =pd.read_csv(data_path,sep = ' ',header=None,keep_default_na=False,engine='python').values.tolist()
    NetMHCpanData.extend(temp)
print("Data size of eluted ligand(EL) data NetMHCpan4.1 is {}".format(len(NetMHCpanData)))

#Load allelelist
data_path = './data/NetMHCpan_train/allelelist'
temp =pd.read_csv(data_path,sep = ' |\t',header=None,keep_default_na=False,engine='python').values.tolist()
cellline_dic = dict()
allele_dic = dict()
IM_index_dic = dict()
for item in temp:
    allele_list = item[1].split(',')
    cellline_dic[item[0]] = allele_list
    if(len(allele_list)==1):
        allele = allele_list[0]
        if 'HLA-' in allele:
            allele = allele.replace('HLA-A','HLA-A*')
            allele = allele.replace('HLA-B','HLA-B*')
            allele = allele.replace('HLA-C','HLA-C*')
        allele_dic[allele] = list()
        IM_index_dic[allele] = list()
        
    
#Get EL SA data
pmhcData_EL = list()
mhc_ELSA_list = list()
num_pos_EL_SA = 0
for item in NetMHCpanData:
    peptide = item[0]
    EL = item[1]
    if EL == 1:
        IM = 'NA'
    else:
        IM = 0
    mhcName = cellline_dic[item[2]]

    if len(mhcName) == 1 and len(peptide)>=8 and len(peptide)<=14:
        mhcName = mhcName[0]
        if 'HLA-' in mhcName:
            mhcName = mhcName.replace('HLA-A','HLA-A*')
            mhcName = mhcName.replace('HLA-B','HLA-B*')
            mhcName = mhcName.replace('HLA-C','HLA-C*') 
        if mhcName not in mhc_ELSA_list:
            mhc_ELSA_list.append(mhcName)
        writeLine = [peptide,mhcName,EL,IM]
        pmhcData_EL.append(writeLine)

        if EL == 1:
            num_pos_EL_SA += 1

print('{} MHCs in ELSA data'.format(len(mhc_ELSA_list)))
num_EL_SA = len(pmhcData_EL)
print("pos/neg={}/{}".format(num_pos_EL_SA,num_EL_SA-num_pos_EL_SA))
print("After filtering samples, Data size of eluted ligand(EL) data NetMHCpan4.1 is {}".format(len(pmhcData_EL)))


#Save pmhcData to local
column=['peptide','MHC','eluted ligand']
output_dir = './data/ELSA_full_trainingData.csv'
output = pd.DataFrame(columns=column,data=[[item[0],item[1],item[2]]for item in pmhcData_EL])
output.to_csv(output_dir,index = None)

###########Mix IM and EL data#############
#Fusion
pmhcData = list()
miss_HLA_list = list()
for n,item in enumerate(pmhcData_IM):
    peptide = item[0]
    mhcName = item[1]
    IM = item[3]

    if mhcName in allele_dic.keys():
        allele_dic[mhcName].append(peptide)
        IM_index_dic[mhcName].append(n)
    elif mhcName not in miss_HLA_list:
        miss_HLA_list.append(mhcName)
        # print("Missing {} in allele list!".format(mhcName)) #No comment for debugging

    pmhcData.append(item)

mix_count = 0
conflict_count = 0
EL1IM0_count = 0
for n,item in enumerate(pmhcData_EL):
    peptide = item[0]
    mhcName = item[1]
    EL = item[2]

    if peptide in allele_dic[mhcName]:
        mix_count += 1
        temp = allele_dic[mhcName].index(peptide)
        index = IM_index_dic[mhcName][temp]
        if pmhcData[index][2] != 'NA' and EL != pmhcData[index][2]:
            conflict_count += 1
            
        if EL == 1 and pmhcData[index][3] == 0:
            EL1IM0_count += 1
        
        pmhcData[index][2] = EL
    else:
        pmhcData.append(item)

print('pure_IM={},pure_EL={},mix={}'.format(len(pmhcData_IM)-mix_count,len(pmhcData_EL)-mix_count,mix_count))
print('conflict_count={},EL1IM0_count={}'.format(conflict_count,EL1IM0_count))

#Save pmhcData to local
column=['peptide','MHC','EL','IM']
output_dir = './data/ELIM_full_trainingData.csv'
output = pd.DataFrame(columns=column,data=pmhcData)
output.to_csv(output_dir,index = None)




