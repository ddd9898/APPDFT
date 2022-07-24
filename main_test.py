from models.baseline import Model,baseline
import torch
from utils.dataloader import pmhcData
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm




###############Test APPDFT and baseline-ELIM ############
#Init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LINK = True
data_flag = 4
BASELINE_EL = True


#Load Data
if data_flag == 0:
    testDataset = pmhcData(data_path = './data/EL_full_testingData.csv')
elif data_flag == 1:
    testDataset = pmhcData(data_path='./data/IM_full_testingData.csv')
elif data_flag == 2:
    testDataset = pmhcData(data_path='./data/TESLA_full_testingData.csv')
elif data_flag == 3:
    testDataset = pmhcData(data_path='./data/SarsCov2-con_full_testingData.csv')
elif data_flag == 4:
    testDataset = pmhcData(data_path='./data/SarsCov2-un_full_testingData.csv')
elif data_flag == 5:
    testDataset = pmhcData(data_path='./data/IM_full_trainingData.csv')

test_loader = DataLoader(testDataset, batch_size=2048, shuffle=False)

#Load models
model_dir = './output/models/'
if not BASELINE_EL:
    if LINK:
        model_basename = 'Model-link_trainingData_fold*_index^_EL.model'
    else:  
        model_basename = 'baseline-ELIM_trainingData_fold*_index^_EL.model'
else:
    model_basename = 'baseline-EL_trainingData_fold*_index^_EL.model'
    
models = []
for n in range(5):
    m = 0
    if not BASELINE_EL:
        model = Model(num_encoder_layers = 2).to(device)
    else:
        model = baseline(num_encoder_layers = 2).to(device)
    
    model_name = model_basename.replace('*', str(n)).replace('^', str(m))
    model_path = model_dir + model_name
    # weights = torch.load(model_path)
    weights = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(weights)

    models.append(model)

#Test 
total_preds_EL = torch.Tensor()
total_labels = torch.Tensor()
for data in tqdm(test_loader):
    #Get input
    ConcatSeq = data[0].to(device)

    #Calculate output
    gt = data[1]
    output_ave_EL = 0
    for model in models:
        model.eval()
        with torch.no_grad():
            if not BASELINE_EL:
                y_EL,_ = model(ConcatSeq,link = LINK)
            else:
                y_EL = model(ConcatSeq)
            y_EL = y_EL.cpu()
            output_ave_EL = output_ave_EL + y_EL
    output_ave_EL = output_ave_EL / len(models)
    total_preds_EL = torch.cat((total_preds_EL, output_ave_EL), 0)
    total_labels = torch.cat((total_labels, gt), 0)
G = total_labels.numpy().flatten()
P_EL = total_preds_EL.numpy().flatten()

#Save to local
column=['peptide','HLA','label','score']
results = list()
for n in range(len(G)):
    if(G[n]!= testDataset.data[n][2]):
        print("Something is wrong!")
    results.append([testDataset.data[n][0],testDataset.data[n][1],G[n],P_EL[n]])

if not BASELINE_EL:
    if LINK :
        if data_flag == 0:
            output_dir = './output/results/results_ELdata_Model-link_index0(5folds).csv'
        elif data_flag == 1:
            output_dir = './output/results/results_IMdata_Model-link_index0(5folds).csv'
        elif data_flag == 2:
            output_dir = './output/results/results_TESLAdata_Model-link_index0(5folds).csv'
        elif data_flag == 3:
            output_dir = './output/results/results_SarsCov2-conData_Model-link_index0(5folds).csv'
        elif data_flag == 4:
            output_dir = './output/results/results_SarsCov2-unData_Model-link_index0(5folds).csv'
        elif data_flag == 5:
            output_dir = './output/results/results_IMtrainData_Model-link_index0(5folds).csv'

    else:
        if data_flag == 0:
            output_dir = './output/results/results_ELdata_baseline-ELIM_index0(5folds).csv'
        elif data_flag == 1:
            output_dir = './output/results/results_IMdata_baseline-ELIM_index0(5folds).csv'
        elif data_flag == 2:
            output_dir = './output/results/results_TESLAdata_baseline-ELIM_index0(5folds).csv'
        elif data_flag == 3:
            output_dir = './output/results/results_SarsCov2-conData_baseline-ELIM_index0(5folds).csv'
        elif data_flag == 4:
            output_dir = './output/results/results_SarsCov2-unData_baseline-ELIM_index0(5folds).csv'
        elif data_flag == 5:
            output_dir = './output/results/results_IMtrainData_baseline-ELIM_index0(5folds).csv'
else:
    if data_flag == 0:
        output_dir = './output/results/results_ELdata_baseline-EL_index0(5folds).csv'
    elif data_flag == 1:
        output_dir = './output/results/results_IMdata_baseline-EL_index0(5folds).csv'
    elif data_flag == 2:
        output_dir = './output/results/results_TESLAdata_baseline-EL_index0(5folds).csv'
    elif data_flag == 3:
        output_dir = './output/results/results_SarsCov2-conData_baseline-EL_index0(5folds).csv'
    elif data_flag == 4:
        output_dir = './output/results/results_SarsCov2-unData_baseline-EL_index0(5folds).csv'
    elif data_flag == 5:
        output_dir = './output/results/results_IMtrainData_baseline-EL_index0(5folds).csv'
output = pd.DataFrame(columns=column,data=results)
output.to_csv(output_dir,index = None)

