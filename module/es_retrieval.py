import json
import os
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union
from utils.elastic_setting import *
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
import pickle

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

