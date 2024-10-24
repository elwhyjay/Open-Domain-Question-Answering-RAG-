import logging
import os
import sys
import time
from typing import Callable, Dict, List, NoReturn, Tuple
from datetime import datetime, timedelta, timezone

import json
import torch
import numpy as np
from module.arguments import DataTrainingArguments, ModelArguments, CustomTrainingArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from module.sparse_retrieval import SparseRetrieval,BM25Retrieval,BM25andTfidfRetrieval
from module.trainer_qa import QuestionAnsweringTrainer
from module.dense_retrieval import ColBERTRetrieval
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from module.utils_qa import check_no_error, postprocess_qa_predictions
from module.dpr.tokenizer import *
from module.dpr.model import ColbertModel
from module.es_retrieval import ESRetrieval
from omegaconf import DictConfig, OmegaConf
from contextlib import contextmanager

from module.mrc import run_mrc



logger = logging.getLogger(__name__)
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

def inference(cfg: DictConfig):
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
       
    model_args = ModelArguments(**cfg.get("model"))
    data_args = DataTrainingArguments(**cfg.get("data"))
    training_args = TrainingArguments(**cfg.get("train"))
    customtraining_args = CustomTrainingArguments(**cfg.get("custom"))

    result_path = f"{model_args.model_name_or_path.split('/')[-1]}_{data_args.dataset_name.split('/')[-1]}_{datetime.now(timezone(timedelta(hours=9))).strftime('%m-%d-%H')}"
    training_args.output_dir = os.path.join(training_args.output_dir, result_path)
    project_name = f"{model_args.model_name_or_path.split('/')[-1]}_train_dataset_bm25Plus_max512_stride256"
    model_path = os.path.join(model_args.saved_model_path,project_name)
    #training_args.do_train = True

    print(f"model file from {model_path}")
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    #logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    #print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )
    
    training_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        # datasets = run_sparse_retrieval(
        #     tokenizer.tokenize, datasets, training_args, data_args,
        # )
        if customtraining_args.retrieval_type == "dense":
            datasets = run_colbert_retrieval(datasets)
        else:
            datasets = run_sparse_retrieval(
                tokenizer.tokenize, datasets, training_args, data_args, customtraining_args.sparse_retrieval_type
            )
    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, training_tokenizer, model)

    
def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    sparse_retrieval_type: str,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    
    if sparse_retrieval_type == "elastic":
        retriever = ESRetrieval("wiki")
    else:
        if sparse_retrieval_type == "bm25":
            retriever = BM25Retrieval(
                tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
            )
        elif sparse_retrieval_type == "tfidf":
            retriever = SparseRetrieval(
                tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
            )
        retriever.get_sparse_embedding()

    # retriever = BM25andTfidfRetrieval(
    #     tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    # )
    # retriever.get_bm25_embedding()
    # retriever.get_tfidf_embedding()
    
    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:       
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_colbert_retrieval(datasets):
    test_dataset = datasets["validation"].flatten_indices().to_pandas()
    MODEL_NAME = 'klue/bert-base'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    special_tokens={'additional_special_tokens' :['[Q]','[D]']}
    ret_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ret_tokenizer.add_special_tokens(special_tokens)
    model = ColbertModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(ret_tokenizer.vocab_size + 2)


    model.to(device)


    model.load_state_dict(torch.load('/data/ephemeral/home/LYJ/level2-mrc-nlp-05/module/dpr/colbert/compare_colbert_pretrain_v3_finetune_6.pth'))

    print('opening wiki passage...')
    with open('/data/ephemeral/home/LYJ/data/wikipedia_documents.json', "r", encoding="utf-8") as f:
        wiki = json.load(f)
    context = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print('wiki loaded!!!')

    query= list(test_dataset['question'])
    mrc_ids =test_dataset['id']
    length = len(test_dataset)

    if os.path.exists('/data/ephemeral/home/LYJ/level2-mrc-nlp-05/module/dpr/colbert/inferecne_colbert_rank.pth'):
        rank = torch.load('/data/ephemeral/home/LYJ/level2-mrc-nlp-05/module/dpr/colbert/inferecne_colbert_rank.pth')
        print(rank.size())
    else:
        batched_p_embs = []
        with torch.no_grad():
            model.eval

            q_seqs_val = tokenize_colbert(query,ret_tokenizer,corpus='query').to('cuda')
            q_emb = model.query(**q_seqs_val).to('cpu')
            print(q_emb.size())

            print(q_emb.size())

            print('Start passage embedding......')
            p_embs=[]
            for step,p in enumerate(tqdm(context)):
                p = tokenize_colbert(p,ret_tokenizer,corpus='doc').to('cuda')
                p_emb = model.doc(**p).to('cpu').numpy()
                p_embs.append(p_emb)
                if (step+1)%200 ==0:
                    batched_p_embs.append(p_embs)
                    p_embs=[]
            batched_p_embs.append(p_embs)
        


        dot_prod_scores = model.get_score(q_emb,batched_p_embs,eval=True)
        print(dot_prod_scores.size())

        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        print(dot_prod_scores)
        print(rank)
        torch.save(rank,'/data/ephemeral/home/LYJ/level2-mrc-nlp-05/module/dpr/colbert/inferecne_colbert_rank.pth')
        print(rank.size())
    

    k = 50
    passages=[]

    for idx in range(length):
        passage=''
        for i in range(k):
            passage += context[rank[idx][i]]
            passage += ' '
        passages.append(passage)

    df = pd.DataFrame({'question':query,'id':mrc_ids,'context':passages})
    f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    complete_datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return complete_datasets


class HybridRetrieval:
    def __init__(
        self,
        sparse_retriever: SparseRetrieval,
        colbert_path: str,
        data_path = "../data/",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        alpha: float = 0.5  # sparse와 dense 점수의 가중치
    ) -> NoReturn:
        """
        Arguments:
            sparse_retriever: 초기 retrieval을 위한 SparseRetrieval 인스턴스
            colbert_path: 학습된 ColBERT 모델 경로
            data_path: 데이터가 저장된 경로
            device: 연산 장치
            batch_size: 배치 크기
            alpha: sparse와 dense 점수를 결합할 때 sparse 점수의 가중치 (1-alpha가 dense 점수 가중치)
        """
        self.sparse_retriever = sparse_retriever
        self.device = device
        self.batch_size = batch_size
        self.alpha = alpha
        special_tokens={'additional_special_tokens' :['[Q]','[D]']}
        self.ret_tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
        self.ret_tokenizer.add_special_tokens(special_tokens)
        model = ColbertModel.from_pretrained('klue/bert-base')
        model.resize_token_embeddings(self.ret_tokenizer.vocab_size + 2)


        model.to(device)
        # ColBERT 모델 로드
        self.colbert = model.load_state_dict(torch.load('/data/ephemeral/home/LYJ/level2-mrc-nlp-05/module/dpr/colbert/compare_colbert_pretrain_v3_finetune_6.pth'))

        self.colbert.eval()
        
        self.max_length = 512

    def _tokenize(self, texts) -> dict:
        """텍스트를 토크나이즈하는 내부 메서드"""
        if isinstance(texts, str):
            texts = [texts]
            
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

    def rerank_documents(self, query, docs, 
                        sparse_scores, topk: int = 5):
        """
        주어진 문서들을 ColBERT로 재순위화합니다.

        Args:
            query: 검색 쿼리
            docs: 재순위화할 문서 리스트
            sparse_scores: Sparse Retrieval에서 얻은 점수들
            topk: 반환할 상위 문서 수
        
        Returns:
            (최종 점수 리스트, 재정렬된 문서 인덱스 리스트)
        """
        with torch.no_grad():
            # Query encoding
            q_inputs = self._tokenize(query)
            q_inputs = {k: v.to(self.device) for k, v in q_inputs.items()}
            Q = self.colbert.query(**q_inputs)
            
            # Document encoding and scoring
            dense_scores = []
            for i in range(0, len(docs), self.batch_size):
                batch_docs = docs[i:i + self.batch_size]
                d_inputs = self._tokenize(batch_docs)
                d_inputs = {k: v.to(self.device) for k, v in d_inputs.items()}
                D = self.colbert.doc(**d_inputs)
                
                # Calculate scores for batch
                batch_scores = self.colbert.get_score(Q, [D], eval=True)
                dense_scores.extend(batch_scores.cpu().squeeze().tolist())
            
            # Normalize scores
            sparse_scores = np.array(sparse_scores)
            dense_scores = np.array(dense_scores)
            
            # Min-Max normalization
            sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())
            dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
            
            # Combine scores
            final_scores = self.alpha * sparse_scores + (1 - self.alpha) * dense_scores
            
            # Get top-k
            top_k_indices = np.argsort(final_scores)[::-1][:topk]
            top_k_scores = final_scores[top_k_indices]
            
            return top_k_scores.tolist(), top_k_indices.tolist()

    def retrieve(
        self, query_or_dataset, topk = 5,
        candidate_k = None  # sparse retrieval에서 가져올 후보 문서 수
    ) :
        """
        Sparse Retrieval로 후보를 추출한 후 ColBERT로 재순위화를 수행합니다.
        
        Args:
            query_or_dataset: 쿼리 문자열 또는 데이터셋
            topk: 최종적으로 반환할 문서 수
            candidate_k: Sparse Retrieval에서 가져올 후보 문서 수 (None이면 topk의 3배)
        """
        if candidate_k is None:
            candidate_k = topk * 2

        if isinstance(query_or_dataset, str):
            # Sparse retrieval로 후보 추출
            sparse_scores, candidate_docs = self.sparse_retriever.retrieve(
                query_or_dataset, topk=candidate_k
            )
            
            # Re-ranking
            final_scores, doc_indices = self.rerank_documents(
                query_or_dataset, candidate_docs, sparse_scores, topk=topk
            )
            
            print("[Search query]\n", query_or_dataset, "\n")
            for i in range(topk):
                print(f"Top-{i+1} passage with score {final_scores[i]:4f}")
                print(candidate_docs[doc_indices[i]])
            
            return (final_scores, [candidate_docs[idx] for idx in doc_indices])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("hybrid retrieval"):
                for idx, example in enumerate(tqdm(query_or_dataset, desc="Hybrid retrieval: ")):
                    query = example["question"]
                    
                    # Sparse retrieval로 후보 추출
                    sparse_scores, candidate_docs = self.sparse_retriever.retrieve(
                        query, topk=candidate_k
                    )
                    
                    # Re-ranking
                    final_scores, doc_indices = self.rerank_documents(
                        query, candidate_docs, sparse_scores, topk=topk
                    )
                    
                    # 결과 저장
                    tmp = {
                        "question": example["question"],
                        "id": example["id"],
                        "context": " ".join(
                            [candidate_docs[did] for did in doc_indices]
                        ),
                    }
                    
                    if "context" in example.keys() and "answers" in example.keys():
                        tmp["original_context"] = example["context"]
                        tmp["answers"] = example["answers"]
                    
                    total.append(tmp)

            return pd.DataFrame(total)