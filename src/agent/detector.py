from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from sklearn.ensemble import IsolationForest

from .features import featurize
from ..env.system import Telemetry


@dataclass
class Detection:
    """
    Detector output:
    - is_anomaly: boolean trigger
    - scores: per-service anomaly scores (higher = more anomalous)
    - global_score: average anomaly score across services
    """
    is_anomaly: bool
    scores: Dict[str, float]
    global_score: float


class AnomalyDetector:
    """
    I use an unsupervised detector because in ops I usually don’t have perfect labels.
    IsolationForest is a good “classic” choice for a prototype:
    - works without labels
    - handles nonlinear boundaries
    - easy to explain

    The catch: it needs some baseline data. So I buffer early telemetry and train after
    collecting enough samples.
    """

    def __init__(self, window: int = 30, contamination: float = 0.06, seed: int = 0):
        self.window = window
        self.contamination = contamination
        self.seed = seed

        # I keep a rolling buffer per service for training.
        self._buf: Dict[str, List[np.ndarray]] = {}

        # One model per service.
        self._models: Dict[str, IsolationForest] = {}
        self._trained = False

    def _ensure_service(self, s: str) -> None:
        self._buf.setdefault(s, [])

    def observe(self, tel: Telemetry) -> Detection:
        """
        Each step:
        1) featurize the telemetry
        2) append into buffers
        3) train if we have enough baseline
        4) score each service
        5) threshold scores into is_anomaly
        """
        fv = featurize(tel)

        # --- Update buffers ---
        for s, x in fv.service_features.items():
            self._ensure_service(s)
            self._buf[s].append(x)

            # I cap buffer size so memory doesn’t grow forever.
            if len(self._buf[s]) > self.window * 4:
                self._buf[s] = self._buf[s][-self.window * 4 :]

        # --- Train models once we have enough samples ---
        # I assume early episode is "mostly normal", which is common in anomaly detection setups.
        if (not self._trained) and all(len(b) >= self.window * 2 for b in self._buf.values()):
            for s, b in self._buf.items():
                X = np.stack(b[: self.window * 2], axis=0)
                m = IsolationForest(
                    n_estimators=200,          # more trees = more stable, but slower
                    contamination=self.contamination,
                    random_state=self.seed,
                )
                m.fit(X)
                self._models[s] = m
            self._trained = True

        # --- Compute anomaly scores ---
        scores: Dict[str, float] = {}
        if self._trained:
            for s, x in fv.service_features.items():
                m = self._models[s]

                # IsolationForest: decision_function is higher for “normal” points.
                # I negate it so “higher score = more anomalous” (more intuitive).
                df = float(m.decision_function([x])[0])
                scores[s] = -df

        global_score = float(np.mean(list(scores.values()))) if scores else 0.0
        api_score = scores.get("api", 0.0)

        # --- Convert scores into a boolean signal ---
        # You tuned this threshold already (Option A). This is where "sensitivity vs precision" lives.
        is_anomaly = (api_score > 0.15) or (global_score > 0.12)

        return Detection(is_anomaly=is_anomaly, scores=scores, global_score=global_score)