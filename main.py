import os
import argparse
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from module.train import train
from module.inference import inference


@hydra.main(version_base="1.3",config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        print("train")
       
        train(cfg.train)
    elif cfg.mode == "inference":
        print("inference")
        
        inference(cfg.inference)

if __name__ == "__main__":
    main()