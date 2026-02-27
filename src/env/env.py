from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Optional, Dict, Any

from .faults import FaultEvent, sample_fault
from .system import SimulatedSystem, Telemetry


@dataclass
class StepResult:
    """
    This is what the env returns per step:
    - telemetry: what the agent sees next
    - reward: a simple scalar so the policy can learn
    - done: episode ended
    - info: debugging info (fault, timestamps, etc.)
    """
    telemetry: Telemetry
    reward: float
    done: bool
    info: Dict[str, Any]


class AIOpsEnv:
    """
    This wrapper gives me a clean “episode” API:
    reset() -> initial telemetry
    act(action) -> apply an action
    step() -> advance time and get next telemetry + reward
    """

    def __init__(self, seed: int = 0, steps: int = 240):
        self.rng = random.Random(seed)
        self.steps = steps
        self.system = SimulatedSystem(self.rng)
        self.fault: Optional[FaultEvent] = None
        self.t = 0

        # I track incident timing so I can compute an MTTR-like metric.
        self._incident_active = False
        self._first_slo_violation_t: Optional[int] = None
        self._recovered_t: Optional[int] = None

    def reset(self, seed: Optional[int] = None) -> Telemetry:
        """
        Start a new episode.
        I sample exactly one fault per episode to keep the evaluation easy to interpret.
        """
        if seed is not None:
            self.rng = random.Random(seed)
            self.system = SimulatedSystem(self.rng)

        self.fault = sample_fault(self.rng, self.system.services, self.steps)
        self.system.reset(self.fault)

        self.t = 0
        self._incident_active = False
        self._first_slo_violation_t = None
        self._recovered_t = None

        return self.system.observe()

    def act(self, action: Dict[str, Any]) -> None:
        """
        I represent actions as dicts so it’s easy to extend:
        {'type':'restart','service':'db'}
        {'type':'scale','service':'api','delta':1}
        etc.
        """
        typ = action.get("type")
        if typ == "restart":
            self.system.restart(action["service"])
        elif typ == "scale":
            self.system.scale(action["service"], int(action.get("delta", 1)))
        elif typ == "clear_cache":
            self.system.clear_cache()
        elif typ == "limit_traffic":
            self.system.limit_traffic(float(action.get("factor", 0.7)))
        elif typ == "noop":
            pass
        else:
            raise ValueError(f"Unknown action type: {typ}")

    def step(self) -> StepResult:
        """
        Advance the simulation one step and compute a reward signal.

        My reward is intentionally simple:
        - I reward “healthy” steps
        - I penalize SLO violation steps
        - I lightly penalize “time spent unrecovered” after the first violation
        """
        self.t += 1
        tel = self.system.step()

        # Detect when an incident begins (first SLO violation)
        if tel.slo_violation and self._first_slo_violation_t is None:
            self._first_slo_violation_t = self.t
            self._incident_active = True

        # Detect recovery (first time we return to non-violation after being in violation)
        if self._incident_active and not tel.slo_violation:
            self._recovered_t = self._recovered_t or self.t
            self._incident_active = False

        # Reward shaping: encourage recovery + discourage violations.
        reward = 1.0
        if tel.slo_violation:
            reward -= 2.0
        if self._first_slo_violation_t is not None and self._recovered_t is None:
            reward -= 0.02

        done = self.t >= self.steps
        info = {
            "fault": self.fault,
            "t": self.t,
            "first_slo_violation_t": self._first_slo_violation_t,
            "recovered_t": self._recovered_t,
        }
        return StepResult(telemetry=tel, reward=reward, done=done, info=info)