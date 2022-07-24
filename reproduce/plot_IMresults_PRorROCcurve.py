import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,auc
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

#Code 1: draw PR curve of all compared methods
precision_list = list()
recall_list = list()
PRAUC_list = list()

#Load Model-link
results_path = './output/results/results_IMdata_Model-link_index0(5folds).csv'
results =pd.read_csv(results_path,sep = ',',keep_default_na=False).values.tolist()

gt_all = list()
pre_all = list()
for n,item in enumerate(results):
    peptide = item[0]
    hla = item[1]
    gt = item[2]
    pre = item[3] #y_IM
    
    gt_all.append(gt)
    pre_all.append(pre)


precision, recall, thresholds = precision_recall_curve(gt_all, pre_all)
AUC_PR = auc(recall, precision)
print(AUC_PR)
precision_list.append(precision)
recall_list.append(recall)
PRAUC_list.append(AUC_PR)

#Load NetMHCpan4.1
data = pd.read_csv('./data/IM_full_testingData.csv').values.tolist()
results = pd.read_csv('./data/test_by_otherMethods/IM_test_forNetMHCpan.csv').values.tolist()

replace_MHC = {
'HLA-A*3':'HLA-A*3001',
'HLA-A*1':'HLA-A*0101',
'HLA-A*11':'HLA-A*1101',
'HLA-B*44:01':'HLA-B*44:02',
'HLA-B*08:011':'HLA-B*08:11',
}

pre_dict = dict()
gt_dict = dict()
for item in data:
    peptide = item[0]
    HLA = item[1]
    if HLA in replace_MHC.keys():
        HLA = replace_MHC[HLA]
    IM = item[2]
    
    key = peptide + ',' + HLA
    
    gt_dict[key] = IM

for item in results:
    peptide = item[1]
    HLA = item[0]
    # IM = item[6] #Bind
    IM = item[2] #EL score
    # IM = 1- item[3]
    
    key = peptide + ',' + HLA
    
    pre_dict[key] = IM


pre_list = list()
gt_list = list()
for key in pre_dict.keys():
    pre_list.append(pre_dict[key])
    gt_list.append(gt_dict[key])


precision, recall, thresholds = precision_recall_curve(gt_list, pre_list)
AUC_PR = auc(recall, precision)
print(AUC_PR)
precision_list.append(precision)
recall_list.append(recall)
PRAUC_list.append(AUC_PR)


##Load transPHLA
data = pd.read_csv('./data/IM_full_testingData.csv').values.tolist()
results = pd.read_csv('./data/test_by_otherMethods/IM_test_forTransPHLA.csv').values.tolist()

pre_dict = dict()
gt_dict = dict()
for item in data:
    peptide = item[0]
    HLA = item[1]
    IM = item[2]
    
    key = peptide + ',' + HLA
    
    gt_dict[key] = IM

for item in results:
    peptide = item[2]
    HLA = item[0]
    # IM = item[3] #0/1
    IM = item[4] #score
    
    key = peptide + ',' + HLA
    
    pre_dict[key] = IM


pre_list = list()
gt_list = list()
for key in pre_dict.keys():
    pre_list.append(pre_dict[key])
    gt_list.append(gt_dict[key])

precision, recall, thresholds = precision_recall_curve(gt_list, pre_list)
AUC_PR = auc(recall, precision)
print(AUC_PR)
precision_list.append(precision)
recall_list.append(recall)
PRAUC_list.append(AUC_PR)


##Load DeepHLApan
results_path = './data/test_by_otherMethods/IM_test_forDeepHLApan.csv'
results =pd.read_csv(results_path,sep = ',',keep_default_na=False).values.tolist()

gt_all = list()
pre_all = list()
for n,item in enumerate(results):
    peptide = item[2]
    hla = item[1]
    
    gt = item[0]
    if gt == 'random':
        gt = 0
    else:
        gt = 1

    pre = item[4]

    
    
    gt_all.append(gt)
    pre_all.append(pre)


precision, recall, thresholds = precision_recall_curve(gt_all, pre_all)
AUC_PR = auc(recall, precision)
print(AUC_PR)
precision_list.append(precision)
recall_list.append(recall)
PRAUC_list.append(AUC_PR)



PRAUC_list[0] = 'APPDFT:' + str(round(PRAUC_list[0],4))
PRAUC_list[1] = 'NetMHCpan4.1:' + str(round(PRAUC_list[1],4))
PRAUC_list[2] = 'transPHLA:' + str(round(PRAUC_list[2],4))
PRAUC_list[3] = 'DeepHLApan' + str(round(PRAUC_list[3],4))

plt.figure(1)
plt.title('Precision/Recall Curve on IM full test data')
plt.xlabel('Recall')
plt.ylabel('Precision')

color = ['r','g','dodgerblue','orange']
for n in range(4):
    precision = precision_list[n]
    recall = recall_list[n]
    plt.plot(precision, recall,c = color[n])
plt.legend(PRAUC_list)
plt.pause(1)
plt.savefig('./output/figures/PR curve benchmark on IM testing data.png',bbox_inches = 'tight',transparent=True)
plt.savefig('./output/figures/PR curve benchmark on IM testing data.pdf',bbox_inches = 'tight',format='pdf',transparent=True)


#Code 2 :
precision_list = list()
recall_list = list()
PRAUC_list = list()

#Load deepImmuno-CNN
data = pd.read_csv('./data/IM_full_testingData.csv').values.tolist()
results = pd.read_csv('./data/test_by_otherMethods/IM_test_forDeepImmuno.txt',sep='\t').values.tolist()


pre_dict_DeepImmuno = dict()
gt_dict = dict()
for item in data:
    peptide = item[0]
    HLA = item[1].replace(':','')
    IM = item[2]
    
    key = peptide + ',' + HLA
    
    gt_dict[key] = IM

for item in results:
    peptide = item[0]
    HLA = item[1]
    # IM = item[6] #Bind
    IM = item[2] #EL score
    
    key = peptide + ',' + HLA
    
    pre_dict_DeepImmuno[key] = IM


pre_list = list()
gt_list = list()
for key in pre_dict_DeepImmuno.keys():
    pre_list.append(pre_dict_DeepImmuno[key])
    gt_list.append(gt_dict[key])

precision, recall, thresholds = precision_recall_curve(gt_list, pre_list)
AUC_PR = auc(recall, precision)
print(AUC_PR)
precision_list.append(precision)
recall_list.append(recall)
PRAUC_list.append(AUC_PR)


#Load Model-link
results_path = './output/results/results_IMdata_Model-link_index0(5folds).csv' 
results =pd.read_csv(results_path,sep = ',',keep_default_na=False).values.tolist()
count = 0
pre_dict_Model_link =  dict()
for n,item in enumerate(results):
    peptide = item[0]
    HLA = item[1].replace(':','')
    gt = item[2]
    pre = item[3] #y_IM
    
    key = peptide + ',' + HLA
    if len(peptide) == 9 or len(peptide) == 10:
        pre_dict_Model_link[key] = pre
    
    # if key in pre_dict_Model_link.keys():
    #     pre_dict_Model_link[key] = pre
    #     count += 1

pre_list = list()
gt_list = list()
for key in pre_dict_Model_link.keys():
    pre_list.append(pre_dict_Model_link[key])
    gt_list.append(gt_dict[key])


precision, recall, thresholds = precision_recall_curve(gt_list, pre_list)
AUC_PR = auc(recall, precision)
print(AUC_PR)
precision_list.append(precision)
recall_list.append(recall)
PRAUC_list.append(AUC_PR)



#Load NetMHCpan4.1
results = pd.read_csv('./data/test_by_otherMethods/IM_test_forNetMHCpan.csv').values.tolist()


pre_dict_NetMHCpan41 = dict()
for item in results:
    peptide = item[1]
    HLA = item[0].replace(':','')
    # IM = item[6] #Bind
    IM = item[2] #EL score
    # IM = 1- item[3]
    
    key = peptide + ',' + HLA
    
    if len(peptide) == 9 or len(peptide) == 10:
        pre_dict_NetMHCpan41[key] = IM


pre_list = list()
gt_list = list()
for key in pre_dict_NetMHCpan41.keys():
    pre_list.append(pre_dict_NetMHCpan41[key])
    gt_list.append(gt_dict[key])


precision, recall, thresholds = precision_recall_curve(gt_list, pre_list)
AUC_PR = auc(recall, precision)
print(AUC_PR)
precision_list.append(precision)
recall_list.append(recall)
PRAUC_list.append(AUC_PR)

##Load transPHLA
results = pd.read_csv('./data/test_by_otherMethods/IM_test_forTransPHLA.csv').values.tolist()

pre_dict_transPHLA = dict()
for item in results:
    peptide = item[2]
    HLA = item[0].replace(':','')
    # IM = item[3] #0/1
    IM = item[4] #score
    
    key = peptide + ',' + HLA
    
    if len(peptide) == 9 or len(peptide) == 10:
        pre_dict_transPHLA[key] = IM


pre_list = list()
gt_list = list()
for key in pre_dict_transPHLA.keys():
    pre_list.append(pre_dict_transPHLA[key])
    gt_list.append(gt_dict[key])

precision, recall, thresholds = precision_recall_curve(gt_list, pre_list)
AUC_PR = auc(recall, precision)
print(AUC_PR)
precision_list.append(precision)
recall_list.append(recall)
PRAUC_list.append(AUC_PR)


##Load DeepHLApan
results_path = './data/test_by_otherMethods/IM_test_forDeepHLApan.csv'
results =pd.read_csv(results_path,sep = ',',keep_default_na=False).values.tolist()

pre_dict_DeepHLApan = dict()
for n,item in enumerate(results):
    peptide = item[2]
    HLA = item[1].replace(':','').replace('HLA-A','HLA-A*').replace('HLA-B','HLA-B*').replace('HLA-C','HLA-C*')
    gt = item[0]
    pre = item[4]
    
    key = peptide + ',' + HLA
    if len(peptide) == 9 or len(peptide) == 10:
        pre_dict_DeepHLApan[key]=pre

pre_list = list()
gt_list = list()
for key in pre_dict_DeepHLApan.keys():
    pre_list.append(pre_dict_DeepHLApan[key])
    gt_list.append(gt_dict[key])

precision, recall, thresholds = precision_recall_curve(gt_list,pre_list)
AUC_PR = auc(recall, precision)
print(AUC_PR)
precision_list.append(precision)
recall_list.append(recall)
PRAUC_list.append(AUC_PR)


print(len(pre_dict_DeepHLApan),len(pre_dict_DeepImmuno),len(pre_dict_Model_link),len(pre_dict_NetMHCpan41),len(pre_dict_transPHLA))

PRAUC_list[0] = 'DeepImmuno-CNN:' + str(round(PRAUC_list[0],4))
PRAUC_list[1] = 'APPDFT:' + str(round(PRAUC_list[1],4))
PRAUC_list[2] = 'NetMHCpan4.1:' + str(round(PRAUC_list[2],4))
PRAUC_list[3] = 'transPHLA:' + str(round(PRAUC_list[3],4))
PRAUC_list[4] = 'DeepHLApan' + str(round(PRAUC_list[4],4))

plt.figure(2)
plt.title('Precision/Recall Curve of IM 9-10mer test data')
plt.xlabel('Recall')
plt.ylabel('Precision')

color = ['lime','r','g','dodgerblue','orange']
for n in range(5):
    precision = precision_list[n]
    recall = recall_list[n]
    plt.plot(precision, recall,c=color[n])
plt.legend(PRAUC_list)
plt.pause(1)
plt.savefig('./output/figures/PR curve benchmark on IM testing data(9-10mer).png',bbox_inches = 'tight',transparent=True)
plt.savefig('./output/figures/PR curve benchmark on IM testing data(9-10mer).pdf',bbox_inches = 'tight',format='pdf',transparent=True)  # 保存图片




#Code 3: draw PR curve of self compared methods
data_list = ['results_IMdata_Model-link_index0(5folds).csv','results_IMdata_baseline-ELIM_index0(5folds).csv',
             'results_IMdata_baseline-EL_index0(5folds).csv']

precision_list = list()
recall_list = list()
PRAUC_list = list()
for data_name in data_list:
    results_path = './output/results/' + data_name
    results =pd.read_csv(results_path,sep = ',',keep_default_na=False).values.tolist()

    gt_all = list()
    pre_all = list()
    for n,item in enumerate(results):
        peptide = item[0]
        hla = item[1]
        gt = item[2]
        pre = item[3] #y_IM
        
        gt_all.append(gt)
        pre_all.append(pre)


    precision, recall, thresholds = precision_recall_curve(gt_all, pre_all)
    AUC_PR = auc(recall, precision)
    print(AUC_PR)
    
    precision_list.append(precision)
    recall_list.append(recall)
    PRAUC_list.append(AUC_PR)

PRAUC_list[0] = 'APPDFT:' + str(round(PRAUC_list[0],4))
PRAUC_list[1] = 'baseline-ELIM:' + str(round(PRAUC_list[1],4))
PRAUC_list[2] = 'baseline-EL:' + str(round(PRAUC_list[2],4))

plt.figure(3)
plt.title('Precision/Recall Curve on IM test data')
plt.xlabel('Recall')
plt.ylabel('Precision')

color = ['r','g','dodgerblue']
for n in range(3):
    precision = precision_list[n]
    recall = recall_list[n]
    plt.plot(precision, recall,c = color[n])
plt.legend(PRAUC_list)
plt.pause(1)

plt.savefig('./output/figures/PR curve benchmark on IM testing data(Self).png',bbox_inches = 'tight',transparent=True)
plt.savefig('./output/figures/PR curve benchmark on IM testing data(Self).pdf',bbox_inches = 'tight',format='pdf',transparent=True)  # 保存图片




#Code 4: draw PR curve of self compared methods
data_list = ['results_TESLAdata_Model-link_index0(5folds).csv','results_TESLAdata_baseline-ELIM_index0(5folds).csv',
             'results_TESLAdata_baseline-EL_index0(5folds).csv']

precision_list = list()
recall_list = list()
PRAUC_list = list()
for data_name in data_list:
    results_path = './output/results/' + data_name
    results =pd.read_csv(results_path,sep = ',',keep_default_na=False).values.tolist()

    gt_all = list()
    pre_all = list()
    for n,item in enumerate(results):
        peptide = item[0]
        hla = item[1]
        gt = item[2]
        pre = item[3] #y_IM
        
        gt_all.append(gt)
        pre_all.append(pre)


    precision, recall, thresholds = precision_recall_curve(gt_all, pre_all)
    AUC_PR = auc(recall, precision)
    print(AUC_PR)
    
    precision_list.append(precision)
    recall_list.append(recall)
    PRAUC_list.append(AUC_PR)

PRAUC_list[0] = 'APPDFT:' + str(round(PRAUC_list[0],4))
PRAUC_list[1] = 'baseline-ELIM:' + str(round(PRAUC_list[1],4))
PRAUC_list[2] = 'baseline-EL:' + str(round(PRAUC_list[2],4))

plt.figure(4)
plt.title('Precision/Recall Curve on TESLA test data')
plt.xlabel('Recall')
plt.ylabel('Precision')

color = ['r','g','dodgerblue']
for n in range(3):
    precision = precision_list[n]
    recall = recall_list[n]
    plt.plot(precision, recall,c = color[n])
plt.legend(PRAUC_list)
plt.pause(1)

plt.savefig('./output/figures/PR curve benchmark on TESLA testing data(Self).png',bbox_inches = 'tight',transparent=True)
plt.savefig('./output/figures/PR curve benchmark on TESLA testing data(Self).pdf',bbox_inches = 'tight',format='pdf',transparent=True)  # 保存图片




#Code 5: draw ROC curve of self compared methods
from sklearn.metrics import roc_curve,auc
data_list = ['results_SarsCov2-conData_Model-link_index0(5folds).csv','results_SarsCov2-conData_baseline-ELIM_index0(5folds).csv',
             'results_SarsCov2-conData_baseline-EL_index0(5folds).csv']

fpr_list = list()
tpr_list = list()
ROCAUC_list = list()
for data_name in data_list:
    results_path = './output/results/' + data_name
    results =pd.read_csv(results_path,sep = ',',keep_default_na=False).values.tolist()

    gt_all = list()
    pre_all = list()
    for n,item in enumerate(results):
        peptide = item[0]
        hla = item[1]
        gt = item[2]
        pre = item[3] #y_IM
        
        gt_all.append(gt)
        pre_all.append(pre)


    fpr, tpr, thresholds  =  roc_curve(gt_all, pre_all) 
    AUC_ROC = auc(fpr, tpr) 
    print(AUC_ROC)
    
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    ROCAUC_list.append(AUC_ROC)

ROCAUC_list[0] = 'APPDFT:' + str(round(ROCAUC_list[0],4))
ROCAUC_list[1] = 'baseline-ELIM:' + str(round(ROCAUC_list[1],4))
ROCAUC_list[2] = 'baseline-EL:' + str(round(ROCAUC_list[2],4))

plt.figure(5)
plt.title('ROC curve on SarsCov2 convalescent test data')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')

color = ['r','g','dodgerblue']
for n in range(3):
    precision = fpr_list[n]
    recall = tpr_list[n]
    plt.plot(precision, recall,c = color[n])
plt.legend(ROCAUC_list)
plt.pause(1)

plt.savefig('./output/figures/ROC curve benchmark on SarsCov2-conData(Self).png',bbox_inches = 'tight',transparent=True)
plt.savefig('./output/figures/ROC curve benchmark on SarsCov2-conData(Self).pdf',bbox_inches = 'tight',format='pdf',transparent=True)  # 保存图片

