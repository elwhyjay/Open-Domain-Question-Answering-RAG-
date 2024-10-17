import json
import random
import pandas as pd
from tqdm import tqdm, trange
import os
import time
from contextlib import contextmanager
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NoReturn, Optional, Tuple, Union


from datasets import Dataset,load_dataset, load_metric, load_from_disk

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    set_seed,
    AdamW, get_linear_schedule_with_warmup,
    T5Model,T5PreTrainedModel
)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class Encoder(nn.Module):
    def __init__(self, model_checkpoint,do_train = True,do_eval = True):
        super(Encoder, self).__init__()
        if do_train:
            self.encoder = AutoModel.from_pretrained(model_checkpoint)
        else:
            config = AutoConfig.from_pretrained(model_checkpoint)
            self.encoder = AutoModel.from_config(config)
    def forward(self, input):
        outputs = self.encoder(**input)
        return outputs["pooler_output"]
    
class DPR_encoder(nn.Module):
    def __init__(self, model_checkpoint, do_train, do_eval):
        super(DPR_encoder, self).__init__()
        self.p_encoder = Encoder(model_checkpoint, do_train, do_eval)
        self.q_encoder = Encoder(model_checkpoint, do_train, do_eval)

        with open(os.path.join("../data/wikipedia_documents.json"), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

    

    def get_score(self,P,Q):
        pass

    def prepare_in_batch_negative(self,num_neg):
        corpus = list(self.dataset["context"])
        corpus = np.array(corpus)
        p_with_neg = []

        for c in self.dataset["context"]:
            while True:
                neg_idx = np.random.randint(len(corpus),size = num_neg)
                if c not in corpus[neg_idx]:
                    p_neg = corpus[neg_idx]
                    p_with_neg.append(c)
                    p_with_neg.append(p_neg)
                    break
        q_seqs = self.tokenizer(
            self.dataset["question"],
            padding = "max_length",
            truncation = True,
            return_tensors = "pt"
        )

        p_seqs = self.tokenizer(
            p_with_neg,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt"
        )

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1,num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1,num_neg + 1, max_len)
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1,num_neg + 1, max_len)

        self.dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"],
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )

    def train(
        self,
        args = None):

        batch_size = args.per_device_train_batch_size
        train_dataloader = DataLoader(self.dataset, batch_size = batch_size, shuffle = True)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        

        





if __name__ == "__main__":
    pass