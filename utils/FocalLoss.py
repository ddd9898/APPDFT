from torch import nn
import torch
from torch.nn import functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, sampling='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.sampling = sampling

    def forward(self, y_pred, y_true,weights):
        alpha = self.alpha
        alpha_ = (1 - self.alpha)
        if self.logits:
            y_pred = torch.sigmoid(y_pred)

        pt_positive = torch.where(y_true == 1, y_pred, torch.ones_like(y_pred))
        pt_negative = torch.where(y_true == 0, y_pred, torch.zeros_like(y_pred))
        pt_positive = torch.clamp(pt_positive, 1e-3, .999)
        pt_negative = torch.clamp(pt_negative, 1e-3, .999)
        pos_ = (1 - pt_positive) ** self.gamma
        neg_ = pt_negative ** self.gamma

        pos_loss = -alpha * pos_ * torch.log(pt_positive)
        neg_loss = -alpha_ * neg_ * torch.log(1 - pt_negative)
        loss = pos_loss + neg_loss
        loss = torch.mul(weights, loss)

        if self.sampling == "mean":
            return loss.mean()
        elif self.sampling == "sum":
            return loss.sum()
        elif self.sampling == None:
            return loss


def MultiTaskLoss_pretrain(y_EL,y_IM,g_EL,g_IM,weights,pretrain=False):
    '''
    Calculate the loss of multiple task.
    args:
        y_EL:The predicted possibility of  presentation.
        y_IM:The predicted immunogenicity.
        g_EL: The ground truth of presentation.
        g_IM: The ground truth of immunogenicity.
        weights: The weight of different loss
    '''
    
    #get the coefficient
    EL_coefficient = 1- torch.isnan(g_EL).float()
    IM_coefficient = 1- torch.isnan(g_IM).float()
    
    #Set nan to 0
    g_EL = torch.where(torch.isnan(g_EL),torch.full_like(g_EL,0),g_EL)
    g_IM = torch.where(torch.isnan(g_IM),torch.full_like(g_IM,0),g_IM)

    if pretrain: #regression
        loss_EL = torch.mean(torch.pow((y_EL - g_EL), 2)*EL_coefficient)
        loss_IM = torch.mean(torch.pow((y_IM - g_IM), 2)*IM_coefficient)
    else: #classify
        focal_loss_EL = FocalLoss()
        focal_loss_IM = FocalLoss()
        loss_EL = focal_loss_EL(y_EL,g_EL,EL_coefficient)
        loss_IM = focal_loss_IM(y_IM,g_IM,IM_coefficient)
    

    # weights = torch.FloatTensor(weights)
    task_loss = torch.stack([loss_EL,loss_IM])
    w_ave_loss = torch.sum(torch.mul(weights, task_loss))

    return w_ave_loss,task_loss

def SingleTaskLoss_pretrain(y,g,pretrain=False):
    '''
    Calculate the loss of single task.
    args:
        y:The predicted possibility of  presentation or immunogenicity.
        g: The ground truth of presentation or immunogenicity.
    '''
    if pretrain: #regression
        loss = torch.mean(torch.pow((y - g), 2))
    else: #classify
        coefficient = 1- torch.isnan(g).float()
        focal_loss = FocalLoss()
        loss = focal_loss(y,g,coefficient)

    return loss

if __name__ == '__main__':
    y_BA = torch.FloatTensor([0.1,0.5,0.6,0.7,1])
    y_EL = torch.FloatTensor([0.2,0.6,0.4,0.9,0.9])
    y_IM = torch.FloatTensor([0.2,0.6,0.4,0.9,0.9])
    g_BA = torch.FloatTensor([0.2,0.2,9000,9000,0.2])
    g_EL = torch.FloatTensor([-np.nan,1,0,1,0])
    g_IM = torch.FloatTensor([-0,-np.nan,0,1,1])


    # r = F.relu(-corrcoef(y_BA,y_EL))
    # weights = torch.nn.Parameter(torch.ones(2).float())
    weights = torch.FloatTensor([1,1,1])
    loss = MultiTaskLoss_pretrain(y_EL,y_IM,g_EL,g_IM,weights)

    print(loss)