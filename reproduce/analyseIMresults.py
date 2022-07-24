import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score,precision_recall_curve,auc
import numpy as np

def evaluate(G,P):
    
    #Convert to np array
    G = np.array(G).flatten()
    P = np.array(P).flatten()

    ######evalute
    GT_labels = list()
    Pre = list()
    for n,item in enumerate(G):
        if (not np.isnan(item)):
            GT_labels.append(int(item))
            Pre.append(P[n])

    # Thresh = 0.425625 #1-np.log(500)/np.log(50000)
    AUC_ROC = roc_auc_score(GT_labels,Pre)
    precision_list, recall_list, _ = precision_recall_curve(GT_labels, Pre)
    AUC_PR = auc(recall_list, precision_list)

    Thresh = 0.5
    pre_labels = [1 if item>Thresh else 0 for item in Pre]

    accuracy = accuracy_score(GT_labels,pre_labels)
    recall = recall_score(GT_labels,pre_labels)
    precision = precision_score(GT_labels,pre_labels)
    F1_score = f1_score(GT_labels,pre_labels)
    
    evaluation = [precision,recall,accuracy,F1_score,AUC_ROC,AUC_PR] 
    

    return evaluation


#Load our results
results_paths = ['./output/results/results_IMdata_Model-link_index0(5folds).csv',
                 './output/results/results_TESLAdata_Model-link_index0(5folds).csv',
                 './output/results/results_SarsCov2-conData_Model-link_index0(5folds).csv',
                 './output/results/results_SarsCov2-unData_Model-link_index0(5folds).csv',
                 
                 './output/results/results_IMdata_baseline-ELIM_index0(5folds).csv',
                 './output/results/results_TESLAdata_baseline-ELIM_index0(5folds).csv',
                 './output/results/results_SarsCov2-conData_baseline-ELIM_index0(5folds).csv',
                 './output/results/results_SarsCov2-unData_baseline-ELIM_index0(5folds).csv',
                 
                 './output/results/results_IMdata_baseline-EL_index0(5folds).csv',
                 './output/results/results_TESLAdata_baseline-EL_index0(5folds).csv',
                 './output/results/results_SarsCov2-conData_baseline-EL_index0(5folds).csv',
                 './output/results/results_SarsCov2-unData_baseline-EL_index0(5folds).csv']
for results_path_b in results_paths:

    results_b =pd.read_csv(results_path_b,sep = ',',keep_default_na=False).values.tolist()

    gt_all = list()
    pre_all = list()
    for n,item in enumerate(results_b):
        peptide = item[0]
        hla = item[1]
        gt = item[2]
        pre = item[3] #y_EL
        
        gt_all.append(gt)
        pre_all.append(pre)

    precision_all,recall_all,acc_all,F1_score_all,AUC_all,PR_AUC_all = evaluate(gt_all,pre_all)
    
    print(results_path_b)
    print('AUC={:4f},Acc={:4f},Recall={:4f},Precision={:4f},F1 Score={:4f},PR AUC={:4f}\n'.format(
    AUC_all,acc_all,recall_all,precision_all,F1_score_all,PR_AUC_all
    ))
