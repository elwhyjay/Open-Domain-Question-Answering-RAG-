import sys
import os
from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)
import datasets

class EmbeddingMixin:
    def __init__(self,model_arg):
        if model_arg is None:
            self.mean = False
            print("Using mean : False")
        else:
            self.mean = model_arg.mean
            print(f"Using mean : {self.mean}")
    
    def __init__weight(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def masked_mean(self, t , mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    
    def masked(self, emb_all, mask):
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:,0] #[batch,len,dim] -> #[batch,0,dim] -> #[batch,dim]

    def query_embedding(self, input_ids, attention_mask):
        raise NotImplementedError("query_embedding function is not implemented")
    def body_embedding(self, input_ids, attention_mask):
        raise NotImplementedError("body_embedding function is not implemented")

class NLL(EmbeddingMixin):
    def forward(self,query_ids,attention_mask_q,
                input_ids_a = None, attention_mask_a = None, # positive passage
                input_ids_b = None, attention_mask_b = None, # negative passage
                is_query = True):
        if input_ids_b is None and is_query:
            return self.query_embedding(query_ids,attention_mask_q)
        elif input_ids_b is None:
            return self.body_embedding(query_ids,attention_mask_q)

        q_embs = self.query_embedding(query_ids,attention_mask_q)
        a_embs = self.body_embedding(input_ids_a,attention_mask_a)
        b_embs = self.body_embedding(input_ids_b,attention_mask_b)

        #nll loss
        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1), (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -lsm[:,0]
        return (loss.mean(),)
    
class RobertaNLL_NL(NLL, RobertaForSequenceClassification):
    def __init__(self,config,model_arg = None):
        NLL.__init__(self,model_arg)
        RobertaForSequenceClassification.__init__(self,config)
        self.embedding_head = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self.__init__weight)

    def query_emb(self,input_ids,attention_mask):
        outputs = self.roberta(input_ids,attention_mask) #(B, L, H)
        last_hidden_states = outputs[0]
        emb_all = self.masked(outputs,attention_mask)
        query = self.norm(self.embedding_head(emb_all))
        return query
    
    def body_emb(self,input_ids,attention_mask):
        return self.query_emb(input_ids,attention_mask)
    

def update_new_embedding(args,model,input,tokenizer,is_query_inference= True):
    embdding, embedding_ids = [],[]

    #tokenizing
    input_token = tokenizer(input, padding="max_length", truncation=True,  return_tensors="pt")
    input_idx = torch.tensor([idx for idx,_ in enumerate(input)])

    #dataloader
    dataset = TensorDataset(
        input_token["input_ids"],
        input_token["attention_mask"],
        input_token["token_type_ids"],
        input_idx
    )
