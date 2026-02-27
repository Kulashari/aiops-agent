from __future__ import annotations
import numpy as np
from typing import Dict
from dataclasses import dataclass

from ..env.system import Telemetry


@dataclass
class FeatureVector:
    """
    I separate service-level features from global features.
    In this prototype, the anomaly detector uses service-level metrics,
    while the overall decision logic can still look at global/API symptoms.
    """
    service_features: Dict[str, np.ndarray]
    global_features: np.ndarray


def featurize(tel: Telemetry) -> FeatureVector:
    """
    Convert Telemetry -> numeric arrays.
    For each service I use: [p95 latency, error rate, cpu, mem]
    """
    service_features: Dict[str, np.ndarray] = {}
    for s in tel.latency_ms_p95.keys():
        service_features[s] = np.array(
            [
                tel.latency_ms_p95[s],
                tel.error_rate[s],
                tel.cpu_util[s],
                tel.mem_util[s],
            ],
            dtype=float,
        )

    # Global features: request rate + the api symptoms (most user-visible).
    global_features = np.array(
        [tel.req_rate, tel.latency_ms_p95["api"], tel.error_rate["api"]],
        dtype=float,
    )

    return FeatureVector(service_features=service_features, global_features=global_features)