import torch
import torch.nn as nn
from copy import deepcopy

class baseline(nn.Module):
    def __init__(self,
                dropout=0.2,
                num_heads=4,
                vocab_size=23,
                num_encoder_layers=2,
                d_embedding = 128, #128
                Max_len = 49, # 1 + 34 + 14 = 49
                ):
        super(baseline, self).__init__()
        

        self.embeddingLayer = nn.Embedding(vocab_size, d_embedding)
        self.positionalEncodings = nn.Parameter(torch.rand(Max_len, d_embedding), requires_grad=True)

        ##Encoder 
        encoder_layers = nn.TransformerEncoderLayer(d_embedding, num_heads,dim_feedforward=1024,dropout=dropout)
        encoder_norm = nn.LayerNorm(d_embedding)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,num_encoder_layers,encoder_norm)
        
        
        self.weights = torch.nn.Parameter(torch.ones(2).float())

        # prediction layers for EL
        self.fc1 = nn.Linear(d_embedding, 1024)
        self.bn1 =  nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 =  nn.BatchNorm1d(256)
        self.outputlayer = nn.Linear(256, 1)
        
        # activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        

    def make_len_mask(self, inp):
        # inp[:,0] = inp[:,0] + 1
        mask = (inp == 22)  #from the dataload.py, * = 22
        # mask[:,0] = False
        return mask


    def forward(self,ConcatSeq):
        # print(ConcatSeq)

        #Get padding mask 
        pad_mask = self.make_len_mask(ConcatSeq)
        
        #Get embedding
        # ConcatEmbedding = ConcatSeq

        ConcatEmbedding = self.embeddingLayer(ConcatSeq)  #batch * seq * feature
        ConcatEmbedding = ConcatEmbedding + self.positionalEncodings[:ConcatEmbedding.shape[1],:]

        #input feed-forward:
        ConcatEmbedding = ConcatEmbedding.permute(1,0,2) #seq * batch * feature
        Concatfeature = self.transformer_encoder(ConcatEmbedding,src_key_padding_mask=pad_mask)
        Concatfeature = Concatfeature.permute(1,0,2) #batch * seq * feature
        # print(Concatfeature.shape)
        # x = Concatfeature.contiguous().view(-1,self.num_features) #batch * (seq * feature)
        # representation = Concatfeature[:,0,:] #batch * seq * feature

        # representation = torch.mean(Concatfeature,dim = 1)
        coff = 1-pad_mask.float()
        representation = torch.sum(coff.unsqueeze(2) * Concatfeature,dim=1)/torch.sum(coff,dim=1).unsqueeze(1)

        #Predict
        x = self.fc1(representation)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)            
        y = self.sigmoid(self.outputlayer(x))
        
        
        #Logical 
        return y


class Model(nn.Module):
    def __init__(self,
                dropout=0.2,
                num_heads=4,
                vocab_size=23,
                num_encoder_layers=2,
                d_embedding = 128, #128
                Max_len = 49, # 1 + 34 + 14 = 49
                ):
        super(Model, self).__init__()
        

        self.embeddingLayer = nn.Embedding(vocab_size, d_embedding)
        self.positionalEncodings = nn.Parameter(torch.rand(Max_len, d_embedding), requires_grad=True)

        ##Encoder 
        encoder_layers = nn.TransformerEncoderLayer(d_embedding, num_heads,dim_feedforward=1024,dropout=dropout)
        encoder_norm = nn.LayerNorm(d_embedding)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,num_encoder_layers,encoder_norm)
        
        
        self.weights = torch.nn.Parameter(torch.ones(2).float())

        # prediction layers for EL
        self.fc1_EL = nn.Linear(d_embedding, 1024)
        self.bn1_EL =  nn.BatchNorm1d(1024)
        self.fc2_EL = nn.Linear(1024, 256)
        self.bn2_EL =  nn.BatchNorm1d(256)
        self.outputlayer_EL = nn.Linear(256, 1)
        
        # prediction layers for IM
        self.fc1_IM = nn.Linear(d_embedding, 1024)
        self.bn1_IM =  nn.BatchNorm1d(1024)
        self.fc2_IM = nn.Linear(1024, 256)
        self.bn2_IM =  nn.BatchNorm1d(256)
        self.outputlayer_IM = nn.Linear(256, 1)
        
        # activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        

    def make_len_mask(self, inp):
        # inp[:,0] = inp[:,0] + 1
        mask = (inp == 22)  #from the dataload.py, * = 22
        # mask[:,0] = False
        return mask


    def forward(self,ConcatSeq,link = True):
        # print(ConcatSeq)

        #Get padding mask 
        pad_mask = self.make_len_mask(ConcatSeq)
        
        #Get embedding
        # ConcatEmbedding = ConcatSeq

        ConcatEmbedding = self.embeddingLayer(ConcatSeq)  #batch * seq * feature
        ConcatEmbedding = ConcatEmbedding + self.positionalEncodings[:ConcatEmbedding.shape[1],:]

        #input feed-forward:
        ConcatEmbedding = ConcatEmbedding.permute(1,0,2) #seq * batch * feature
        Concatfeature = self.transformer_encoder(ConcatEmbedding,src_key_padding_mask=pad_mask)
        Concatfeature = Concatfeature.permute(1,0,2) #batch * seq * feature
        # print(Concatfeature.shape)
        # x = Concatfeature.contiguous().view(-1,self.num_features) #batch * (seq * feature)
        # representation = Concatfeature[:,0,:] #batch * seq * feature

        # representation = torch.mean(Concatfeature,dim = 1)
        coff = 1-pad_mask.float()
        representation = torch.sum(coff.unsqueeze(2) * Concatfeature,dim=1)/torch.sum(coff,dim=1).unsqueeze(1)

        #Predict EL
        x_EL = self.fc1_EL(representation)
        x_EL = self.bn1_EL(x_EL)
        x_EL = self.relu(x_EL)
        x_EL = self.fc2_EL(x_EL)
        x_EL = self.bn2_EL(x_EL)
        x_EL = self.relu(x_EL)            
        y_EL = self.sigmoid(self.outputlayer_EL(x_EL))
        
        #Predict IM
        x_IM = self.fc1_IM(representation)
        x_IM = self.bn1_IM(x_IM)
        x_IM = self.relu(x_IM)
        x_IM = self.fc2_IM(x_IM)
        x_IM = self.bn2_IM(x_IM)
        x_IM = self.relu(x_IM)
        y_IM = self.sigmoid(self.outputlayer_IM(x_IM))
        if link:
            y_IM = y_IM * y_EL
        
        #Logical 
        return y_EL,y_IM

class Model_atten_score(nn.Module):
    def __init__(self,
                 modelfilename = 'Model_BAtrainingData_pretrain.model'):
        super(Model_atten_score, self).__init__()
        
        #Load model
        self.model = Model(num_encoder_layers = 2)
        pretrained_path = './output/models/' + modelfilename
        pre_model = torch.load(pretrained_path,map_location=torch.device('cpu'))
        model2dict = self.model.state_dict()
        state_dict = {k:v for k,v in pre_model.items() if k in model2dict.keys()}
        model2dict.update(state_dict)
        self.model.load_state_dict(model2dict)

        #Get attentions
        self.embeddingLayer = self.model.embeddingLayer
        self.positionalEncodings = self.model.positionalEncodings
        self.transformer_encoder = self.model.transformer_encoder


    def make_len_mask(self, inp):
        # inp[:,0] = inp[:,0] + 1
        mask = (inp == 22)  #from the dataload.py, * = 22
        # mask[:,0] = False
        return mask


    def forward(self,ConcatSeq):
        '''
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        '''
        # print(ConcatSeq)

        #Get padding mask 
        pad_mask = self.make_len_mask(ConcatSeq)
        
        #Get embedding
        # ConcatEmbedding = ConcatSeq

        ConcatEmbedding = self.embeddingLayer(ConcatSeq)  #batch * seq * feature
        ConcatEmbedding = ConcatEmbedding + self.positionalEncodings[:ConcatEmbedding.shape[1],:]
        
        ConcatEmbedding = ConcatEmbedding.permute(1,0,2) #seq * batch * feature
        attn_output_weights = self.transformer_encoder.layers[0].self_attn(ConcatEmbedding, ConcatEmbedding, ConcatEmbedding,
                              key_padding_mask=pad_mask)[1]
        
        y_EL,_ = self.model(ConcatSeq,link = True)

        return attn_output_weights,y_EL

if __name__ == '__main__':
    # ConcatSeq = torch.LongTensor([[1,1,2,3,4,22,1],[2,22,2,3,4,0,5]])
    # model = baseline(num_encoder_layers = 1,Max_len = 7)
    
    # y_EL,y_IM = model(ConcatSeq)
    # print(y_EL,y_IM)
    
    model = Model_atten_score()
    ConcatSeq = [21,5,4,6,7]
    ConcatSeq.extend(list(range(22)))
    ConcatSeq.extend(list(range(22)))
    ConcatSeq = torch.LongTensor([ConcatSeq,ConcatSeq])
    model.eval()
    with torch.no_grad():
        attn_output_weights,y_EL = model(ConcatSeq)
    attn_output_weights = attn_output_weights.numpy()
    print(attn_output_weights)
    
    
    
    

