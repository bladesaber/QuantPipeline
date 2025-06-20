import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Literal


class MathBaseModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def save(self, model_path: str):
        pass

    @abstractmethod
    def load(self, model_path: str):
        pass

