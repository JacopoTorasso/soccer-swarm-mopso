from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketPrediction:
    match_1x2: tuple[float, float, float]
    over_under_25: tuple[float, float]
    btts: tuple[float, float]

    def as_array(self) -> np.ndarray:
        return np.array(self.match_1x2 + self.over_under_25 + self.btts)


class PredictionAgent(ABC):
    @abstractmethod
    def train(self, fixtures: list[dict], stats: list[dict]) -> None: ...

    @abstractmethod
    def predict(self, fixture: dict) -> MarketPrediction | None: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
