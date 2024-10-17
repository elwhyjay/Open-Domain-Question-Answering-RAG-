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


class DenseRetrieval:

    def __init__(self,
        args,
        dataset,
        num_neg,
        tokenizer,
        p_encoder,
        q_encoder
    ):

        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder.to(args.device)
        self.q_encoder = q_encoder.to(args.device)
        
        self.prepare_in_batch_negative(num_neg=num_neg)
        with open(os.path.join("../data/wikipedia_documents.json"), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

    def prepare_in_batch_negative(self,
        dataset=None,
        num_neg=2,
        tokenizer=None
    ):

        """
        Args
        ----
        dataset
            Huggingface의 Dataset을 받아오면,
            in-batch negative를 추가해서 Dataloader를 만들어주세요
        """
        corpus = list(self.dataset["context"])
        corpus = np.array(corpus)
        p_with_neg = []

        for c in self.dataset["context"]:
            while True:
                neg_idxs = np.random.randint(len(corpus),size = num_neg)
                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]
                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        q_seqs = self.tokenizer (
            self.dataset["question"],
            padding = "max_length",
            truncation = True,
            return_tensors="pt"
        )
        p_seqs = self.tokenizer(
            p_with_neg,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, num_neg + 1, max_len)
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, num_neg + 1, max_len)

        self.dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"],
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )

    def train(self,
        args=None
    ):

        if args is None:
          args = self.args


        batch_size = args.per_device_train_batch_size

        # Dataloader
        train_dataloader = DataLoader(self.dataset, batch_size=batch_size)

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:

            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()

                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "attention_mask": batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "token_type_ids": batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }

                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device)
                    }

                    del batch
                    torch.cuda.empty_cache()
                    # (batch_size * (num_neg + 1), emb_dim)
                    p_outputs = self.p_encoder(**p_inputs)
                    # (batch_size, emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    # (batch_size, num_neg + 1)
                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.q_encoder.zero_grad()
                    self.p_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()
                    del p_inputs, q_inputs

    def get_relevant_doc(self,
        query,
        k=1,
        args=None
    ):

        with torch.no_grad():
            self.p_encoder.eval()
            self.q_encoder.eval()

            q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors="pt").to("cuda")
            q_emb = self.q_encoder(**q_seqs_val).to("cpu")  # (num_query, emb_dim)

            p_embs = []
            for p in self.contexts:
                p = self.tokenizer(p, padding="max_length", truncation=True, return_tensors="pt").to("cuda")
                p_emb = self.p_encoder(**p).to("cpu").numpy()
                p_embs.append(p_emb)

        p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))


        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        return rank[:k]
    

if __name__ == "__main__":
    args = TrainingArguments(
    output_dir="dense_retireval",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01
)
    
    model_checkpoint = "klue/bert-base"
    ret = DenseRetrieval(
        tokenize_fn=None,
        data_path="../data",
        context_path="wikipedia_documents.json",
        num_neg=2
    )
    ret.train(args=args)
    