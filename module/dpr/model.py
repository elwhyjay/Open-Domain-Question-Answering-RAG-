import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    BertModel,
    BertTokenizerFast,
    BertPreTrainedModel,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

import string
class ColbertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(ColbertModel, self).__init__(config)

        # BertModel 사용
        self.similarity_metric = "cosine"
        self.dim = 128
        self.bert = BertModel(config)
        self.init_weights()
        self.linear = nn.Linear(config.hidden_size, self.dim, bias=False)
        tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
        mask_symbol = [tokenizer.mask_token_id]
        mask_symbol.extend([tokenizer.encode(symbol,add_special_tokens=False)[0] for symbol in string.punctuation])
        self.register_buffer("mask_buffer", torch.tensor(mask_symbol))
        
    def forward(self, p_inputs, q_inputs):
        Q = self.query(**q_inputs)
        D = self.doc(**p_inputs)
        
        return self.get_score(Q, D)

    def query(self, input_ids, attention_mask, token_type_ids):
        Q = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        Q = self.linear(Q)
        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, token_type_ids):
        D = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        D = self.linear(D)

        puntuation_mask = self.punctuation_mask(input_ids).unsqueeze(2)
        D = D * puntuation_mask
        return torch.nn.functional.normalize(D, p=2, dim=2)
    def punctuation_mask(self, input_ids):
        mask = (input_ids.unsqueeze(-1) == self.mask_buffer).any(dim=-1)
        mask = (~mask).float()
        return mask
    def get_score(self, Q, D, eval=False):
        # hard negative N은 train에만 쓰임.
        if eval:
            if self.similarity_metric == "cosine":
                final_score = torch.tensor([])
                for D_batch in tqdm(D):
                    D_batch = torch.Tensor(np.array(D_batch)).squeeze() # (p_batch_size, p_sequence_length, hidden_size)
                    p_sequence_output = D_batch.transpose(
                        1, 2
                    )  # (p_batch_size, hidden_size, p_sequence_length)
                    q_sequence_output = Q.view(
                        Q.shape[0], 1, -1, self.dim
                    )  # (q_batch_size, 1, q_sequence_length, hidden_size)
                    dot_prod = torch.matmul(
                        q_sequence_output, p_sequence_output
                    )  # (q_batch_size, p_batch_size, q_sequence_length, p_sequence_length)
                    max_dot_prod_score = torch.max(dot_prod, dim=3)[
                        0
                    ]   # (q_batch_size, p_batch_size, q_sequnce_length)
                    score = torch.sum(max_dot_prod_score, dim=2)  # (q_batch_size, p_batch_size)
                    final_score = torch.cat([final_score, score], dim=1)
                print(final_score.size()) #(q_batch_size, num_whole_p)
                return final_score

        else:
            if self.similarity_metric == "cosine":
                
                p_sequence_output = D.transpose(1, 2)  # (batch_size, hidden_size, p_sequence_length)
                q_sequence_output = Q.view(
                    Q.shape[0], 1, -1, self.dim
                )  # (batch_size, 1, q_sequence_length, hidden_size)
                dot_prod = torch.matmul(
                    q_sequence_output, p_sequence_output
                )  # (batch_size, batch_size + hard_negative_size, q_sequence_length, p_seqence_length)
                max_dot_prod_score = torch.max(dot_prod, dim=3)[0]  # (batch_size, batch_size + hard_negative_size, q_sequnce_length)
                final_score = torch.sum(max_dot_prod_score, dim=2)  # (batch_size, batch_size + hard_negative_size)
                
                return final_score