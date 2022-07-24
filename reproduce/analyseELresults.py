import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def cal_PPV(gt_list,pre_list):
    th = 0.5
    gt1_list = list()
    pre1_list = list()
    for n in range(len(gt_list)):
        if (pre_list[n] > th):
            pre1_list.append(pre_list[n])
            gt1_list.append(gt_list[n])
    
    Z = zip(pre1_list,gt1_list)
    Z = sorted(Z,reverse=True)
    pre1_list,gt1_list = zip(*Z)
    
    len1 = np.int(len(pre1_list)*0.95)
    pre1_list = pre1_list[:len1]
    gt1_list = gt1_list[:len1]
    
    PPV = np.sum(gt1_list)/len1
    
    return PPV

#Init
for flag in [0,1,2]:
    #Load NetMHCpan4.1 result
    results_path_a = './data/NetMHCpan_test/supplementary_table_8.xlsx'
    results_a =pd.read_excel(results_path_a,keep_default_na=False).values.tolist()
    AUC_a = dict()
    AUC_b = dict()
    AUC_01_a = dict()
    AUC_01_b = dict()
    pre_list = dict()
    gt_list = dict()
    for i in range(2,len(results_a)):
        method = 'NetMHCpan-4.1' # 'NetMHCpan-4.1' 'NetMHCpan-4.0'    'MixMHCpred'  'MHCFlurry'  'MHCFlurry_EL'
        idx = results_a[0].index(method)

        hla = results_a[i][0]
        
        if 'HLA-' in hla:
            hla = hla.replace('HLA-A','HLA-A*')
            hla = hla.replace('HLA-B','HLA-B*')
            hla = hla.replace('HLA-C','HLA-C*')
        
        auc = results_a[i][idx]
        auc01 = results_a[i][idx+1]
        AUC_a[hla] = auc
        AUC_01_a[hla] = auc01
        
        AUC_b[hla] = 0
        AUC_01_b[hla] = 0
        
        pre_list[hla] = list()
        gt_list[hla] = list()


    #Load our results
    if flag==0:
        results_path_b = './output/results/results_ELdata_Model-link_index0(5folds).csv'
    elif flag == 1:
        results_path_b = './output/results/results_ELdata_baseline-ELIM_index0(5folds).csv'
    elif flag == 2:
        results_path_b = './output/results/results_ELdata_baseline-EL_index0(5folds).csv'
        
    results_b =pd.read_csv(results_path_b,sep = ',',keep_default_na=False).values.tolist()
    gt_all = list()
    pre_all = list()
    for item in results_b:
        peptide = item[0]
        hla = item[1]
        gt = item[2]
        pre = item[3] #y_EL
        pre_list[hla].append(pre)
        gt_list[hla].append(gt)

        gt_all.append(gt)
        pre_all.append(pre)


    #Evaluate
    results_samples = list()
    for sample in AUC_b.keys():
        GT_sample = gt_list[sample]
        pre_sample = pre_list[sample]
        AUC_ROC_01 = roc_auc_score(GT_sample,pre_sample,max_fpr=0.1)
        AUC_ROC = roc_auc_score(GT_sample,pre_sample)
        AUC_b[sample] = AUC_ROC
        PPV = cal_PPV(GT_sample,pre_sample)
        results_samples.append([sample,AUC_ROC,AUC_ROC_01,PPV])


    #Save evaluation results on each hla to local.
    column = ['hla','AUC_ROC','AUC01_ROC','PPV']

    if flag==0:
        output_dir = './output/results/evaluations_ELdata_Model-link_index0(5folds).csv'
    elif flag == 1:
        output_dir = './output/results/evaluations_ELdata_baseline-ELIM_index0(5folds).csv'
    elif flag == 2:
        output_dir = './output/results/evaluations_ELdata_baseline-EL_index0(5folds).csv'
        
    output = pd.DataFrame(columns=column,data=results_samples)
    output.to_csv(output_dir,index = None)


    #Compare per HLA
    count_better = 0
    diff_AUC = []
    for sample in AUC_b.keys():
        if AUC_a[sample]<AUC_b[sample]:
            count_better += 1
        diff_AUC.append(AUC_b[sample] -  AUC_a[sample])
    print('{}/{}'.format(count_better,len(AUC_b)))
    print(np.mean(diff_AUC))



