from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import random


class FaultType(str, Enum):
    """
    I keep the fault taxonomy small and recognizable, because the goal is to show an AIOps loop
    (detect → diagnose → act), not to model every real-world failure mode.
    """
    LATENCY_SPIKE = "latency_spike"
    ERROR_BURST = "error_burst"
    CPU_SATURATION = "cpu_saturation"
    MEMORY_LEAK = "memory_leak"
    CACHE_POISON = "cache_poison"


@dataclass(frozen=True)
class FaultEvent:
    """
    This is a single injected fault in the simulation.
    - t_start: when it begins
    - duration: how long it lasts (unless the agent mitigates it)
    - service: which service it hits (api/db/cache)
    - fault_type: the kind of fault
    - severity: how strong the fault is (0..1)
    """
    t_start: int
    duration: int
    service: str
    fault_type: FaultType
    severity: float  # 0..1

    @property
    def t_end(self) -> int:
        # I treat end time as exclusive: [t_start, t_end)
        return self.t_start + self.duration


def sample_fault(rng: random.Random, services: list[str], t_max: int) -> FaultEvent:
    """
    I randomly generate one fault per episode.
    The exact distribution isn’t sacred—this is a toy env—but I do want faults to appear
    after the system has had time to generate baseline telemetry.
    """
    service = rng.choice(services)
    fault_type = rng.choice(list(FaultType))

    # Start somewhere after the first chunk of the episode, so the detector has a chance to learn "normal".
    t_start = rng.randint(max(10, t_max // 6), max(20, t_max // 2))

    # Duration is long enough that the agent actually has time to react.
    duration = rng.randint(t_max // 12, t_max // 4)

    # Severity controls how much the metrics get multiplied/shifted.
    severity = rng.uniform(0.4, 1.0)

    return FaultEvent(
        t_start=t_start,
        duration=duration,
        service=service,
        fault_type=fault_type,
        severity=severity,
    )