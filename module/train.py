from typing import NoReturn

import logging
import os
import random
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk, load_metric
from omegaconf import DictConfig
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

from module.arguments import DataTrainingArguments, ModelArguments
from module.mrc import run_mrc
from module.trainer_qa import QuestionAnsweringTrainer
from module.utils_qa import check_no_error, postprocess_qa_predictions

seed = 2024
set_seed(seed)


logger = logging.getLogger(__name__)


def train(cfg: DictConfig):
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    model_args = ModelArguments(**cfg.get("model"))
    data_args = DataTrainingArguments(**cfg.get("data"))
    training_args = TrainingArguments(**cfg.get("train"))

    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다
    # training_args.per_device_train_batch_size = 4
    # print(training_args.per_device_train_batch_size)
    project_name = f"{model_args.model_name_or_path.split('/')[-1]}_{data_args.dataset_name.split('/')[-1]}_curr"

    # model_args.model_name_or_path = '/data/ephemeral/home/LYJ/level2-mrc-nlp-05/models/train_dataset/roberta-large_korQuad_v2'
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    # logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)

    ## using korQuad v1
    using_korQuad = False
    if using_korQuad:
        project_name = project_name + "_korQuad_v2"
        datasets["train"] = datasets["train"].remove_columns(["document_id", "__index_level_0__"])
        datasets["validation"] = datasets["validation"].remove_columns(["document_id", "__index_level_0__"])
        korQuad_dataset = load_dataset("squad_kor_v1", features=datasets["train"].features)
        korQuad_datasets = concatenate_datasets([korQuad_dataset["train"], korQuad_dataset["validation"]])
        train_dataset = concatenate_datasets([datasets["train"], korQuad_datasets])
        validation_datasets = datasets["validation"]
        datasets = DatasetDict({"train": train_dataset, "validation": validation_datasets})

    training_args.output_dir = os.path.join(training_args.output_dir, project_name)
    wandb.init(project="mrc", name=project_name)
    wandb.config.update(
        {
            "model_name": model_args.model_name_or_path,
            "data_path": data_args.dataset_name,
            "max_seq_length": data_args.max_seq_length,
            "doc_stride": data_args.doc_stride,
        }
    )

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name is not None else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name_or_path,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)

    wandb.finish()
