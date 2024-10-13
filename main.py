import os
import argparse

import hydra
from omegaconf import DictConfig, OmegaConf

from module import train
from module import inference



@hydra.main(version_base="1.3",config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "inference":
        inference(cfg)

if __name__ == "__main__":
    main()