import os
import shutil
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class ResearchWorker(object, ABC):
    def __init__(self, name: str, record_dir: str):
        self.name = name
        self.record_dir = os.path.join(record_dir, name)
        if not os.path.exists(self.record_dir):
            os.mkdir(self.record_dir)

    @abstractmethod
    def download_data(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def process_data(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def data2feature(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def data2label(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def fit_model(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def evaluate_report(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def log_report(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def save_task(self, **kwargs):
        # save feature, label, model
        raise NotImplementedError
    
    