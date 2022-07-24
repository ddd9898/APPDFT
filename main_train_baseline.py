import numpy as np
import torch
import os
import random
from models.baseline import baseline
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score,precision_recall_curve,auc
from scipy.stats import pearsonr
import logging
from utils.dataloader import BAorELorELIMData
from utils.FocalLoss import SingleTaskLoss_pretrain
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
        y = model(ConcatSeq)
        y = y.squeeze()

        ###Calculate loss
        g_BA = data[1].to(device).squeeze()
        loss = SingleTaskLoss_pretrain(y,g_BA,pretrain=True)
    
        train_loss = train_loss + loss.item()

        #Optimize the model
        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL_PRETRAIN == 0:
            logging.info('Pretrain epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                            batch_idx * BATCH_SIZE,
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss.item()))
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
        y = model(ConcatSeq)
        y = y.squeeze()

        ###Calculate loss
        g = data[1].to(device).squeeze()
        loss = SingleTaskLoss_pretrain(y,g,pretrain=False)

    
        train_loss = train_loss + loss.item()

        #Optimize the model
        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            logging.info('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                            batch_idx * BATCH_SIZE,
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss.item()))
    train_loss = train_loss / len(train_loader)
    return train_loss

def predicting(model, device, loader):
    model.eval()
    preds = torch.Tensor()
    labels = torch.Tensor()

    logging.info('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            #Get input
            ConcatSeq = data[0].to(device)

            #Calculate output
            g = data[1]
            y = model(ConcatSeq)

         
            preds = torch.cat((preds, y.cpu()), 0)
            labels = torch.cat((labels, g), 0)

    return labels,preds


def evalute(G,P):
    
    #Convert to np array
    G = G.numpy().flatten()
    P = P.numpy().flatten()


    ######evalute EL
    GT_labels = list()
    Pre = list()
    for n,item in enumerate(G):
        if (not np.isnan(item)):
            GT_labels.append(int(item))
            Pre.append(P[n])

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
    
    evaluation = [precision,recall,accuracy,F1_score,AUC_ROC,PCC,AUC_PR] 
    

    return evaluation

def get_args():
    parser = argparse.ArgumentParser(description='Train the TFMHC',
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
    pretrainDataset = BAorELorELIMData(pretrain = True)
    trainDataset = BAorELorELIMData(pretrain = False,valfold = fold, val = False,EL_flag=True)
    valDataset = BAorELorELIMData(pretrain = False,valfold = fold, val = True,EL_flag=True)

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

    #Get argument parse
    args = get_args()

    #Set log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #Output name
    model_name = 'baseline-EL'
        
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
    model = baseline(num_encoder_layers = 2).to(device) #1
    
    
    if(args.load):
        logging.info('Load pretrained model!')
        
        modelfilename = 'baseline_BAtrainingData_pretrain.model'
        pretrained_path = './output/models/' + modelfilename
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
        torch.save(model.state_dict(),'./output/models/baseline_BAtrainingData_pretrain.model')
   

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
    Device:          {device.type}
    ''')

    best_AUCROC = -1

    early_stop_count = 0
    for epoch in range(NUM_EPOCHS):
        #Train
        train_loss = train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        
        #Validate
        logging.info('predicting for valid data')
        G,P = predicting(model, device, valid_loader)
        # loss,precision,recall,accuracy,F1_score,AUC_ROC,PCC,AUC_PR = evalute(G,P)
        evaluation = evalute(G,P)

        #Logging
        logging.info('Epoch {} with AUCROC:AUCPR={:.4f}:{:.4f}'.format(epoch,evaluation[4],evaluation[6]))
        
        if(best_AUCROC < evaluation[4]):
            best_AUCROC = evaluation[4]
            BestEpoch = epoch
            early_stop_count = 0

            #Save model
            torch.save(model.state_dict(), model_file_name + '_EL' +'.model')
    
        else:
            early_stop_count = early_stop_count + 1
            
            

        logging.info('BestEpoch={}; BestResult={:.4f} with AUCROC.'.format(
            BestEpoch,best_AUCROC
        ))

        #Tensorboard
        precision,recall,accuracy,F1_score,AUC_ROC,PCC,AUC_PR = evaluation
        writer.add_scalar('accuracy_val', accuracy, epoch)
        writer.add_scalar('AUC_ROC_val', AUC_ROC, epoch)
        writer.add_scalar('PCC_val', PCC, epoch)
        writer.add_scalar('recall_val', recall, epoch)
        writer.add_scalar('precision_val', precision, epoch)
        writer.add_scalar('F1_score_val', F1_score, epoch)
        writer.add_scalar('AUC_PR_val', AUC_PR, epoch)
        
        writer.add_scalar('loss_train', train_loss, epoch)

        if early_stop_count >= 20:
            logging.info('Early Stop.')
            break


            


            

        

        







