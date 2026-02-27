from __future__ import annotations
from dataclasses import dataclass

from .detector import AnomalyDetector, Detection
from .diagnoser import Diagnoser, RCAHypothesis
from .policy import Policy, Decision
from ..env.env import AIOpsEnv
from ..env.system import Telemetry


@dataclass
class AgentStep:
    """
    This is a “trace record” for one decision cycle, which is nice for debugging and demos.
    """
    t: int
    slo_violation: bool
    detection: Detection
    rca: RCAHypothesis
    decision: Decision


class AIOpsAgent:
    """
    This class is my “agent brain”.
    I split it into:
    - detector: flags anomalies
    - diagnoser: guesses RCA
    - policy: picks an action
    """

    def __init__(self, seed: int = 0):
        self.detector = AnomalyDetector(seed=seed)
        self.diagnoser = Diagnoser()
        self.policy = Policy(use_bandit=True)

    def step(self, env: AIOpsEnv, tel: Telemetry, t: int):
        """
        One closed-loop cycle:
        1) I observe telemetry (store history)
        2) I detect anomalies
        3) If something looks wrong, I diagnose and decide an action
        4) I execute the action and advance the environment
        5) I learn from the reward signal
        """
        self.diagnoser.observe(tel)
        det = self.detector.observe(tel)

        # I trigger incident handling if either the detector fires OR the SLO is already violated.
        if det.is_anomaly or tel.slo_violation:
            rca = self.diagnoser.diagnose(det.scores)
            decision = self.policy.decide(tel, rca, t)
        else:
            # Normal operation: I keep a noop decision so the loop is consistent.
            rca = RCAHypothesis("api", 0.05, "Everything looks normal, so I’m just monitoring.")
            decision = Decision({"type": "noop"}, "No incident signal; doing nothing.")

        # Execute action, then step the environment.
        tel_before = tel
        env.act(decision.action)
        step_res = env.step()
        tel_after = step_res.telemetry

        # Learn from feedback. (In real ops, feedback might be time-to-recover, or reduced error rate, etc.)
        self.policy.learn(tel_before, tel_after, rca, decision.action, step_res.reward)

        return AgentStep(
            t=t,
            slo_violation=tel_before.slo_violation,
            detection=det,
            rca=rca,
            decision=decision,
        ), step_res