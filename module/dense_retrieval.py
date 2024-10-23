import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import BertTokenizerFast
from module.dpr.model import ColbertModel
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

def load_colbert_model(model_path: str, device: str) -> ColbertModel:
    """
    ColBERT 모델을 올바르게 로드하는 함수
    
    Args:
        model_path: 저장된 모델 경로
        device: 사용할 디바이스
        
    Returns:
        로드된 ColBERT 모델
    """
    # 모델 초기화
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained('klue/bert-base')
    model = ColbertModel(config)
    tokenizer = BertTokenizerFast.from_pretrained('klue/bert-base')
    model.resize_token_embeddings(tokenizer.vocab_size + 2)
    # state dict 로드
    state_dict = torch.load(model_path, map_location=device)
    
    # state dict 적용
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    
    return model.to(device)

class ColBERTRetrieval:
    def __init__(
        self,
        model_path: str = '/data/ephemeral/home/LYJ/level2-mrc-nlp-05/module/dpr/colbert/compare_colbert_pretrain_v3_finetune_6.pth',
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1
    ) -> NoReturn:

        self.data_path = data_path
        self.device = device
        self.batch_size = batch_size

        # Load contexts
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Lengths of unique contexts : {len(self.contexts)}")

        # Load model and tokenizer
        self.model = load_colbert_model(model_path, self.device)
        self.model.eval()
        self.tokenizer = BertTokenizerFast.from_pretrained('klue/bert-base')
        self.max_length = 512

        # Initialize embeddings
        self.p_embedding = None

    def _tokenize(self, texts: Union[str, List[str]], padding=True) -> dict:
        """텍스트를 토크나이즈하는 내부 메서드"""
        if isinstance(texts, str):
            texts = [texts]
            
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length' if padding else False,
            truncation=True,
            return_tensors='pt'
        )

    def get_dense_embedding(self) -> NoReturn:
        """
        Summary:
            Passage Embedding을 만들고 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """
        pickle_name = f"colbert_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            print("Load passage embedding")
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
        else:
            print("Build passage embedding")
            passages_emb = []
            
            for i in tqdm(range(0, len(self.contexts), self.batch_size)):
                batch_texts = self.contexts[i:i + self.batch_size]
                inputs = self._tokenize(batch_texts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    emb = self.model.doc(**inputs)
                    passages_emb.append(emb)
            
            self.p_embedding = passages_emb  # List of tensors
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding saved")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        assert self.p_embedding is not None, "get_dense_embedding() 메소드를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """단일 쿼리에 대해 유사도가 가장 높은 k개의 passage를 반환합니다."""
        with torch.no_grad():
            # Query encoding
            q_inputs = self._tokenize(query)
            q_inputs = {k: v.to(self.device) for k, v in q_inputs.items()}
            Q = self.model.query(**q_inputs)
            
            # Calculate similarity scores using ColBERT's scoring method
            scores = self.model.get_score(Q, self.p_embedding, eval=True)
            scores = scores.squeeze()
            
            # Get top-k documents
            top_k_scores, top_k_indices = torch.topk(scores, k=k)
            
            return top_k_scores.tolist(), top_k_indices.tolist()

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """다수의 쿼리에 대해 유사도가 가장 높은 k개의 passage를 반환합니다."""
        all_scores = []
        all_indices = []
        
        # Process queries in batches
        for i in tqdm(range(0, len(queries), self.batch_size)):
            batch_queries = queries[i:i + self.batch_size]
            
            with torch.no_grad():
                # Query encoding
                q_inputs = self._tokenize(batch_queries)
                q_inputs = {k: v.to(self.device) for k, v in q_inputs.items()}
                Q = self.model.query(**q_inputs)
                
                # Calculate similarity scores
                print(f"Q size: {Q.size()}====================")
                
                scores = self.model.get_score(Q, self.p_embedding, eval=True)
                
                # Get top-k documents for each query in batch
                top_k_scores, top_k_indices = torch.topk(scores, k=k)
                
                all_scores.extend(top_k_scores.tolist())
                all_indices.extend(top_k_indices.tolist())
        
        return all_scores, all_indices

    def preprocess_query(self, query: Union[str, List[str]]) -> dict:
        """query를 전처리하는 메서드"""
        inputs = self._tokenize(query)
        return {k: v.to(self.device) for k, v in inputs.items()}