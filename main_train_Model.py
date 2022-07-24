import numpy as np
import torch
import os
import random
from models.baseline import Model
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score,precision_recall_curve,auc
from scipy.stats import pearsonr
import logging
from utils.dataloader import BAorELIMData
from utils.FocalLoss import MultiTaskLoss_pretrain
from torch.utils.tensorboard import SummaryWriter

def pretrain(model, device, train_loader, optimizer, epoch):
    '''
    training function at each epoch
    '''
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    logging.info('Pretraining on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        #Get input
        ConcatSeq = data[0].to(device)

        #Calculate output
        optimizer.zero_grad()
        y_EL,y_IM = model(ConcatSeq,link=False)
        y_EL = y_EL.squeeze()
        y_IM = y_IM.squeeze()

        ###Calculate loss
        g_BA = data[1].to(device).squeeze()

        weights = torch.FloatTensor([1,1]).to(device)
        w_ave_loss,task_loss = MultiTaskLoss_pretrain(y_EL,y_IM,g_BA,g_BA,weights,pretrain=True)
    
        train_loss = train_loss + w_ave_loss.item()

        #Optimize the model
        w_ave_loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL_PRETRAIN == 0:
            logging.info('Pretrain epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                            batch_idx * BATCH_SIZE,
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            w_ave_loss.item()))
    train_loss = train_loss / len(train_loader)
    return train_loss

def train(model, device, train_loader, optimizer, epoch):
    '''
    training function at each epoch
    '''
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    logging.info('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        #Get input
        ConcatSeq = data[0].to(device)

        #Calculate output
        optimizer.zero_grad()
        y_EL,y_IM = model(ConcatSeq,link=LINK)
        y_EL = y_EL.squeeze()
        y_IM = y_IM.squeeze()

        ###Calculate loss
        g_EL = data[1].to(device).squeeze()
        g_IM = data[2].to(device).squeeze()
        
        weights = torch.FloatTensor([1,1]).to(device)
        # weights = torch.FloatTensor([0.2,1.8]).to(device)#4
        # weights = torch.FloatTensor([0.4,1.6]).to(device)#5
        # weights = torch.FloatTensor([0.6,1.4]).to(device)#6
        # weights = torch.FloatTensor([0.8,1.2]).to(device) #7
        # weights = torch.FloatTensor([1.2,0.8]).to(device) #8
        # weights = torch.FloatTensor([1.4,0.6]).to(device) #9
        # weights = torch.FloatTensor([1.6,0.4]).to(device) #10
        # weights = torch.FloatTensor([1.8,0.2]).to(device) #11

        w_ave_loss,task_loss = MultiTaskLoss_pretrain(y_EL,y_IM,g_EL,g_IM,weights,pretrain=False)
    
        train_loss = train_loss + w_ave_loss.item()

        #Optimize the model
        w_ave_loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            logging.info('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                            batch_idx * BATCH_SIZE,
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            w_ave_loss.item()))
    train_loss = train_loss / len(train_loader)
    return train_loss

def predicting(model, device, loader):
    model.eval()
    preds_IM = torch.Tensor()
    labels_IM = torch.Tensor()
    preds_EL = torch.Tensor()
    labels_EL = torch.Tensor()

    logging.info('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            #Get input
            ConcatSeq = data[0].to(device)

            #Calculate output
            g_EL = data[1]
            g_IM = data[2]
            y_EL,y_IM = model(ConcatSeq,link = LINK)

         
            preds_IM = torch.cat((preds_IM, y_IM.cpu()), 0)
            labels_IM = torch.cat((labels_IM, g_IM), 0)
            preds_EL = torch.cat((preds_EL, y_EL.cpu()), 0)
            labels_EL = torch.cat((labels_EL, g_EL), 0)

    return labels_EL,preds_EL,labels_IM,preds_IM


def evalute(G_EL,P_EL,G_IM,P_IM):
    
    #Convert to np array
    G_EL = G_EL.numpy().flatten()
    P_EL = P_EL.numpy().flatten()
    G_IM = G_IM.numpy().flatten()
    P_IM = P_IM.numpy().flatten()

    ######evalute EL
    GT_labels = list()
    Pre = list()
    for n,item in enumerate(G_EL):
        if (not np.isnan(item)):
            GT_labels.append(int(item))
            Pre.append(P_EL[n])

    # Thresh = 0.425625 #1-np.log(500)/np.log(50000)
    AUC_ROC = roc_auc_score(GT_labels,Pre)
    PCC = np.real(pearsonr(GT_labels,Pre))[0]
    precision_list, recall_list, _ = precision_recall_curve(GT_labels, Pre)
    AUC_PR = auc(recall_list, precision_list)

    Thresh = 0.5 
    pre_labels = [1 if item>Thresh else 0 for item in Pre]

    accuracy = accuracy_score(GT_labels,pre_labels)
    recall = recall_score(GT_labels,pre_labels)
    precision = precision_score(GT_labels,pre_labels)
    F1_score = f1_score(GT_labels,pre_labels)
    
    evaluation_EL = [precision,recall,accuracy,F1_score,AUC_ROC,PCC,AUC_PR]
    
    ######evalute IM
    GT_labels = list()
    Pre = list()
    for n,item in enumerate(G_IM):
        if (not np.isnan(item)):
        # if (not np.isnan(item) and G_EL[n] !=0 ):
            GT_labels.append(int(item))
            Pre.append(P_IM[n])
    AUC_ROC = roc_auc_score(GT_labels,Pre)
    PCC = np.real(pearsonr(GT_labels,Pre))[0]
    precision_list, recall_list, _ = precision_recall_curve(GT_labels, Pre)
    AUC_PR = auc(recall_list, precision_list)
    
    Thresh = 0.5 
    pre_labels = [1 if item>Thresh else 0 for item in Pre]

    accuracy = accuracy_score(GT_labels,pre_labels)
    recall = recall_score(GT_labels,pre_labels)
    precision = precision_score(GT_labels,pre_labels)
    F1_score = f1_score(GT_labels,pre_labels)
    
    evaluation_IM = [precision,recall,accuracy,F1_score,AUC_ROC,PCC,AUC_PR]
    

    return evaluation_EL,evaluation_IM

def get_args():
    parser = argparse.ArgumentParser(description='Train the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fold', dest='fold', type=int, default=0,
                        help='Number of fold',metavar='E')
    parser.add_argument('--load', dest='load', type=str, default=False,
                        help='pretrain or load')
    parser.add_argument('--index', dest='index', type=int, default=0,
                        help='Number of fold',metavar='E')


    return parser.parse_args()


def LoadDataset(fold = 0):
    '''
    Load training dataset and  validation dataset.
    Output:
        trainDataset, valDataset
    '''
    #Load Train and Val Data
    pretrainDataset = BAorELIMData(pretrain = True)
    trainDataset = BAorELIMData(pretrain = False,valfold = fold, val = False)
    valDataset = BAorELIMData(pretrain = False,valfold = fold, val = True)

    return pretrainDataset,trainDataset, valDataset


def seed_torch(seed = 1000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministics = True


if __name__ == '__main__':
    #Fix random seed and reproduce expermental results
    seed_torch()

    #Train setting
    BATCH_SIZE = 3200 #2560
    LR = 0.0001 # 0.001
    LR_PRETRAIN = 0.001 # 0.001
    LOG_INTERVAL = 300 #300
    LOG_INTERVAL_PRETRAIN = 50
    NUM_EPOCHS = 200 #500
    NUM_EPOCHS_PRETRAIN = 50
    LINK = False  #True or False

    #Get argument parse
    args = get_args()

    #Set log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #Output name
    if LINK:
        model_name = 'Model-link'
    else:
        model_name = 'baseline-ELIM'
    # add_name = '_trainingData'+'_fold' + str(args.fold) + str(bool(args.load)) + '_' + str(args.index)
    add_name = '_trainingData'+'_fold' + str(args.fold) + '_index' + str(args.index)
    model_file_name =  './output/models/' + model_name + add_name
    result_file_name = './output/results/result_' + model_name + add_name + '.csv'
    
    
    logfile = './output/log/log_' + model_name + add_name + '.txt'
    fh = logging.FileHandler(logfile,mode='a')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    #Tensorboard
    logfile = './output/log/log_' + model_name + add_name
    writer = SummaryWriter(logfile)
    

    #Step 1:Prepare dataloader
    pretrainDataset,trainDataset, valDataset = LoadDataset(fold = args.fold)
    pretrain_loader = DataLoader(pretrainDataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False)


    #Step 2: Set  model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(num_encoder_layers = 2).to(device) #2
    
    
    if(args.load):
        logging.info('Load pretrained model!')
        
        modelfilename =  'Model_BAtrainingData_pretrain.model'
        # modelfilename =  'Model_BAtrainingData_pretrain(attenL3).model'
        pretrained_path = './output/models/' + modelfilename
        # pre_model = torch.load(pretrained_path,map_location=torch.device('cpu'))
        pre_model = torch.load(pretrained_path)
        model2dict = model.state_dict()
        state_dict = {k:v for k,v in pre_model.items() if k in model2dict.keys()}
        model2dict.update(state_dict)
        model.load_state_dict(model2dict)
    else: #pretrain 
        logging.info(f'''Starting pretraining:
        Epochs:          {NUM_EPOCHS_PRETRAIN}
        Batch size:      {BATCH_SIZE}
        Learning rate:   {LR_PRETRAIN}
        Pretraining size:{len(pretrainDataset)}
        Device:          {device.type}
        ''')
        
        optimizer_pretrain = torch.optim.AdamW(model.parameters(), lr=LR_PRETRAIN, weight_decay=0.01)  #0.001
        scheduler_pretrain = torch.optim.lr_scheduler.MultiStepLR(optimizer_pretrain, milestones=[10,20],
                                                    gamma=0.1)
        for epoch in range(NUM_EPOCHS_PRETRAIN):
            #Train
            pretrain_loss = pretrain(model, device, pretrain_loader, optimizer_pretrain, epoch)
            scheduler_pretrain.step()
            
            writer.add_scalar('loss_pretrain', pretrain_loss, epoch)
            
        #End pretrain
        logging.info('Pretrain finished! Save pretrained model to local.')
        
        #Save model
        torch.save(model.state_dict(),'./output/models/Model_BAtrainingData_pretrain.model')
        # torch.save(model.state_dict(),'./output/models/Model_BAtrainingData_pretrain(attenL3).model')
   

    #Step 3: Train the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01) #0.01
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR) #,weight_decay=0.0001
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,50], #10,20
                                                gamma=0.1)
                                                    
    logging.info(f'''Starting training:
    Epochs:          {NUM_EPOCHS}
    Batch size:      {BATCH_SIZE}
    Learning rate:   {LR}
    Training size:   {len(trainDataset)}
    Validation size: {len(valDataset)}
    LINK:            {LINK}
    Device:          {device.type}
    ''')

    best_EL_AUCROC = -1
    best_IM_AUCROC = -1

    early_stop_count = 0
    for epoch in range(NUM_EPOCHS):
        #Train
        train_loss = train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        
        #Validate
        logging.info('predicting for valid data')
        G_EL,P_EL,G_IM,P_IM = predicting(model, device, valid_loader)
        # evaluation_EL,evaluation_IM = evalute(G_EL,P_EL,G_IM,P_IM)
        evaluation_EL,evaluation_IM = evalute(G_EL,P_EL,G_IM,P_EL)

        #Logging
        logging.info('Epoch {} with EL-AUCROC:IM-AUCROC={:.4f}:{:.4f}'.format(epoch,evaluation_EL[4],evaluation_IM[4]))
        
        flag_no_pregress = 0
        if(best_IM_AUCROC < evaluation_IM[4]):
            best_IM_AUCROC = evaluation_IM[4]
            BestEpoch_IM = epoch
            early_stop_count = 0
            #Save model
            torch.save(model.state_dict(), model_file_name + '_IM' +'.model')
        else:
            flag_no_pregress = flag_no_pregress + 1

        if(best_EL_AUCROC < evaluation_EL[4]):
            best_EL_AUCROC = evaluation_EL[4]
            BestEpoch_EL = epoch
            early_stop_count = 0
            #Save model
            torch.save(model.state_dict(), model_file_name + '_EL' +'.model')
        else:
            flag_no_pregress = flag_no_pregress + 1
            
        if flag_no_pregress == 2:
            early_stop_count = early_stop_count + 1
            

        logging.info('BestEpoch={},{}; BestResult={:.4f},{:.4f} with EL-AUCROC,IM-AUCROC in turn.'.format(
            BestEpoch_EL,BestEpoch_IM,best_EL_AUCROC,best_IM_AUCROC
        ))

        #Tensorboard
        precision,recall,accuracy,F1_score,AUC_ROC,PCC,AUC_PR = evaluation_EL
        writer.add_scalar('accuracy_val_EL', accuracy, epoch)
        writer.add_scalar('AUC_ROC_val_EL', AUC_ROC, epoch)
        writer.add_scalar('PCC_val_EL', PCC, epoch)
        writer.add_scalar('recall_val_EL', recall, epoch)
        writer.add_scalar('precision_val_EL', precision, epoch)
        writer.add_scalar('F1_score_val_EL', F1_score, epoch)
        writer.add_scalar('AUC_PR_val_EL', AUC_PR, epoch)
        
        precision,recall,accuracy,F1_score,AUC_ROC,PCC,AUC_PR = evaluation_IM
        writer.add_scalar('accuracy_val_IM', accuracy, epoch)
        writer.add_scalar('AUC_ROC_val_IM', AUC_ROC, epoch)
        writer.add_scalar('PCC_val_IM', PCC, epoch)
        writer.add_scalar('recall_val_IM', recall, epoch)
        writer.add_scalar('precision_val_IM', precision, epoch)
        writer.add_scalar('F1_score_val_IM', F1_score, epoch)
        writer.add_scalar('AUC_PR_val_IM', AUC_PR, epoch)
        
        writer.add_scalar('loss_train', train_loss, epoch)

        if early_stop_count >= 20:
            logging.info('Early Stop.')
            break


            


            

        

        







