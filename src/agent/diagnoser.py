from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from ..env.system import Telemetry


@dataclass
class RCAHypothesis:
    """
    This is my “root cause” guess.
    I keep it intentionally explainable:
    - primary_service: which service I think is the main contributor
    - confidence: rough confidence heuristic (0..1-ish)
    - rationale: a human-readable explanation for the decision
    """
    primary_service: str
    confidence: float
    rationale: str


class Diagnoser:
    """
    I’m not doing deep causal inference here; I’m doing something that looks like
    a reasonable AIOps baseline:
    - look at per-service anomaly scores
    - look at correlation of service metrics with API symptoms
    - produce a single “best guess” service and explain why
    """

    def __init__(self):
        self.history: List[Telemetry] = []

    def observe(self, tel: Telemetry) -> None:
        # I store recent telemetry for correlation-based reasoning.
        self.history.append(tel)
        if len(self.history) > 200:
            self.history = self.history[-200:]

    def diagnose(self, anomaly_scores: Dict[str, float]) -> RCAHypothesis:
        # If the detector isn’t trained yet, I still have to say something.
        if not anomaly_scores:
            return RCAHypothesis("api", 0.1, "Detector not trained yet; defaulting to api.")

        # Base guess: the most anomalous service.
        ranked = sorted(anomaly_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_svc, top_score = ranked[0]

        # Use a recent window for correlation (enough points to not be nonsense).
        w = self.history[-40:] if len(self.history) >= 10 else self.history
        if len(w) < 8:
            conf = min(0.6, 0.15 + top_score)
            return RCAHypothesis(top_svc, conf, f"Most anomalous service: {top_svc} (score={top_score:.2f}).")

        # I treat API latency/errors as the “symptoms”.
        api_lat = np.array([t.latency_ms_p95["api"] for t in w], dtype=float)
        api_err = np.array([t.error_rate["api"] for t in w], dtype=float)

        def corr(x: np.ndarray, y: np.ndarray) -> float:
            # I guard against division by zero / degenerate correlation.
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1])

        # I pick the service whose metrics correlate best with API symptoms.
        best = (top_svc, 0.0, 0.0)  # (svc, |corr_lat|, |corr_err|)
        for svc in ["db", "cache", "api"]:
            x_lat = np.array([t.latency_ms_p95[svc] for t in w], dtype=float)
            x_err = np.array([t.error_rate[svc] for t in w], dtype=float)

            c1 = abs(corr(x_lat, api_lat))
            c2 = abs(corr(x_err, api_err))

            if (c1 + c2) > (best[1] + best[2]):
                best = (svc, c1, c2)

        svc = best[0]
        score = anomaly_scores.get(svc, 0.0)

        # Confidence is a heuristic blend of anomaly score + correlation strength.
        conf = float(min(0.95, 0.2 + 0.9 * score + 0.35 * (best[1] + best[2])))

        rationale = (
            f"I think {svc} is the root cause because it’s most correlated with API symptoms "
            f"(corr_lat={best[1]:.2f}, corr_err={best[2]:.2f}) and it has anomaly_score={score:.2f}."
        )

        return RCAHypothesis(primary_service=svc, confidence=conf, rationale=rationale)