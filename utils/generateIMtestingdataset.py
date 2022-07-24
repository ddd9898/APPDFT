import numpy as np
import pandas as pd
# from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score,precision_recall_curve,auc

def transFormat(HLA):
    # HLA = 'HLA-C*0702'
    HLA = [item for item in HLA]
    star_index = HLA.index('*')
    HLA.insert(star_index+3,':')
    HLA = ''.join(HLA)
    
    return HLA

#Sars-Cov-2 data
data = pd.read_csv("./data/DeepImmuno/sars_cov_2_result.csv").values.tolist()
new_data_con = list()
new_data_un = list()
gt1_list = list()
gt2_list = list()
pre_list = list()
for n in range(len(data)):
    peptide = data[n][1]
    HLA = transFormat(data[n][4])
    IM_con = data[n][7]
    IM_un = data[n][8]
    
    # IM_score = data[n][9] #DeepImmuno
    # IM_score = data[n][10] #IEDB
    IM_score = data[n][11] #DeepHLApan
    
    gt1_list.append(IM_con)
    gt2_list.append(IM_un)
    pre_list.append(1 if IM_score>=0.5 else 0)
    
    new_data_con.append([peptide,HLA,IM_con])
    new_data_un.append([peptide,HLA,IM_un])

# recall_con = recall_score(gt1_list,pre_list)
# recall_un = recall_score(gt2_list,pre_list)
# precision_con = precision_score(gt1_list,pre_list)
# precision_un = precision_score(gt2_list,pre_list)
# F1_score_con = f1_score(gt1_list,pre_list)
# F1_score_un = f1_score(gt2_list,pre_list)
# print(recall_con,precision_con,F1_score_con)
# print(recall_un,precision_un,F1_score_un)

#Save to local
column=['peptide','HLA','immunogenicity']
# column=['peptide','HLA','CDR3','immunogenicity']
output_dir = './data/SarsCov2-con_full_testingData.csv'
output = pd.DataFrame(columns=column,data=new_data_con)
output.to_csv(output_dir,index = None)

#Save to local
column=['peptide','HLA','immunogenicity']
# column=['peptide','HLA','CDR3','immunogenicity']
output_dir = './data/SarsCov2-un_full_testingData.csv'
output = pd.DataFrame(columns=column,data=new_data_un)
output.to_csv(output_dir,index = None)


#TESLA data
data = pd.read_csv("./data/DeepImmuno/ori_test_cells.csv").values.tolist()
new_data = list()
gt_list = list()
pre_list = list()
for n in range(len(data)):
    peptide = data[n][0]
    HLA = transFormat(data[n][1])
    IM = data[n][2]
    
    # IM_score = data[n][9] #DeepImmuno
    IM_score = data[n][10] #IEDB
    # IM_score = data[n][11] #DeepHLApan
    
    gt_list.append(IM)
    # pre_list.append(1 if IM_score>=0.5 else 0)
    pre_list.append(1 if IM_score>=0 else 0)
    
    new_data.append([peptide,HLA,IM])

# recall = recall_score(gt_list,pre_list)
# precision = precision_score(gt_list,pre_list)
# F1_score = f1_score(gt_list,pre_list)
# print(recall,precision,F1_score)


#Save to local
column=['peptide','HLA','immunogenicity']
# column=['peptide','HLA','CDR3','immunogenicity']
output_dir = './data/TESLA_full_testingData.csv'
output = pd.DataFrame(columns=column,data=new_data)
output.to_csv(output_dir,index = None)




#DeepHLApan
data = pd.read_excel("./data/DeepHLApan/Table_8.XLSX").values.tolist()
new_data = list()
gt_list = list()
pre_list = list()
for n in range(2,len(data)):
    peptide = data[n][0]
    HLA = data[n][3]
    IM = data[n][4]
    if IM == 'random':
        IM = 0
    else:
        IM = 1
    
    IM_score = data[n][7]
    
    gt_list.append(IM)
    pre_list.append(1 if IM_score>=0.5 else 0)
    
    new_data.append([peptide,HLA,IM])

# recall = recall_score(gt_list,pre_list)
# precision = precision_score(gt_list,pre_list)
# precision_list, recall_list, _ = precision_recall_curve(gt_list, pre_list)
# AUC_PR = auc(recall_list, precision_list)
# print(recall,precision,AUC_PR)

#Save to local
column=['peptide','HLA','immunogenicity']
# column=['peptide','HLA','CDR3','immunogenicity']
output_dir = './data/IM_full_testingData.csv'
output = pd.DataFrame(columns=column,data=new_data)
output.to_csv(output_dir,index = None)



