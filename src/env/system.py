from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import Dict, Tuple, Optional

from .faults import FaultEvent, FaultType


@dataclass
class ServiceState:
    """
    This is the “hidden state” of each service that influences its metrics.
    I’m not trying to perfectly simulate reality — I just want plausible relationships:
    - more replicas → less load per replica → lower latency/errors
    - CPU/mem saturation correlates with performance issues
    """
    replicas: int = 1
    cpu_util: float = 0.25  # 0..1
    mem_util: float = 0.25  # 0..1
    cache_health: float = 1.0  # 0..1 (not deeply used, but helps me tell a story)


@dataclass
class Telemetry:
    """
    This is what the agent gets to see at each time step.
    I keep it structured and “metrics-like” rather than raw logs, because it’s easier to
    demonstrate anomaly detection + RCA from numeric telemetry.
    """
    latency_ms_p95: Dict[str, float]
    error_rate: Dict[str, float]
    cpu_util: Dict[str, float]
    mem_util: Dict[str, float]
    req_rate: float
    slo_violation: bool


class SimulatedSystem:
    """
    I’m modeling a tiny microservice system: api depends on db + cache.
    The key thing I want:
    - local service issues can propagate to api (the “user-facing” service)
    - faults affect metrics in realistic directions
    - actions (restart/scale/limit traffic/clear cache) have measurable effects
    """

    def __init__(self, rng: random.Random, services=("api", "db", "cache")):
        self.rng = rng
        self.services = list(services)
        self.state: Dict[str, ServiceState] = {s: ServiceState() for s in self.services}
        self.active_fault: Optional[FaultEvent] = None
        self.t = 0
        self._traffic_limit = 1.0  # 0..1; agent can reduce traffic during incidents

    def reset(self, fault: FaultEvent) -> None:
        # Reset everything to a clean initial state and set the fault for this episode.
        self.state = {s: ServiceState() for s in self.services}
        self.active_fault = fault
        self.t = 0
        self._traffic_limit = 1.0

    def _workload(self) -> float:
        """
        I generate a smooth-ish request rate with a sine wave + noise.
        This creates “normal” variability so the anomaly detector isn’t trivial.
        """
        base = 120.0 + 60.0 * (0.5 + 0.5 * math.sin(self.t / 24.0))
        noise = self.rng.gauss(0, 8.0)
        return max(10.0, (base + noise) * self._traffic_limit)

    def _fault_multiplier(self, service: str) -> Tuple[float, float, float, float]:
        """
        This is where I inject faults into the metrics.
        I return multipliers for (latency, error, cpu, mem).

        If there’s no active fault, or it’s not currently “on”, I return all 1.0.
        """
        if not self.active_fault:
            return 1.0, 1.0, 1.0, 1.0

        f = self.active_fault
        if not (f.service == service and f.t_start <= self.t < f.t_end):
            return 1.0, 1.0, 1.0, 1.0

        sev = f.severity

        # These are deliberately simple “directional” effects.
        # The goal is: faults change telemetry in ways an agent can learn and react to.
        if f.fault_type == FaultType.LATENCY_SPIKE:
            return 1.0 + 3.5 * sev, 1.0 + 0.2 * sev, 1.0 + 0.1 * sev, 1.0

        if f.fault_type == FaultType.ERROR_BURST:
            return 1.0 + 0.6 * sev, 1.0 + 6.0 * sev, 1.0 + 0.2 * sev, 1.0

        if f.fault_type == FaultType.CPU_SATURATION:
            return 1.0 + 1.8 * sev, 1.0 + 1.0 * sev, 1.0 + 2.5 * sev, 1.0

        if f.fault_type == FaultType.MEMORY_LEAK:
            return 1.0 + 1.0 * sev, 1.0 + 0.7 * sev, 1.0 + 0.3 * sev, 1.0 + 2.5 * sev

        if f.fault_type == FaultType.CACHE_POISON:
            return 1.0 + 2.0 * sev, 1.0 + 1.5 * sev, 1.0 + 0.2 * sev, 1.0

        return 1.0, 1.0, 1.0, 1.0

    def observe(self) -> Telemetry:
        """
        I compute synthetic metrics for each service.
        - db/cache computed first
        - api depends on db/cache via penalty terms
        """
        req_rate = self._workload()

        latency_ms_p95: Dict[str, float] = {}
        error_rate: Dict[str, float] = {}
        cpu_util: Dict[str, float] = {}
        mem_util: Dict[str, float] = {}

        # These penalties represent the idea that db/cache issues slow down api.
        api_dep_penalty = 0.0
        dep_err_penalty = 0.0

        # --- Compute db + cache first ---
        for s in ["db", "cache"]:
            st = self.state[s]

            # Load per replica: more replicas reduces load.
            load_per_replica = req_rate / (st.replicas * 200.0)

            # Baseline curves (piecewise-ish): low load = mostly stable, high load = rising latency/errors.
            lat_base = 18.0 + 55.0 * max(0.0, load_per_replica - 0.35)
            err_base = 0.004 + 0.03 * max(0.0, load_per_replica - 0.55)

            # Apply fault multipliers when relevant.
            lmul, emul, cmul, mmul = self._fault_multiplier(s)

            # CPU/mem should correlate with load and certain faults.
            st.cpu_util = min(1.2, 0.15 + 0.9 * load_per_replica * cmul + self.rng.random() * 0.05)
            st.mem_util = min(1.2, 0.18 + 0.6 * load_per_replica * mmul + self.rng.random() * 0.05)

            # Add a bit of noise so the detector sees variability.
            latency_ms_p95[s] = max(5.0, (lat_base * lmul) + self.rng.gauss(0, 2.0))
            error_rate[s] = min(0.8, max(0.0, (err_base * emul) + abs(self.rng.gauss(0, 0.002))))

            # Propagate symptoms into api penalties (toy causal-ish signal).
            api_dep_penalty += max(0.0, latency_ms_p95[s] - 35.0) * 0.45
            dep_err_penalty += error_rate[s] * 0.6

            cpu_util[s] = min(1.0, max(0.0, st.cpu_util))
            mem_util[s] = min(1.0, max(0.0, st.mem_util))

        # --- Compute api (depends on db/cache) ---
        st = self.state["api"]
        load_per_replica = req_rate / (st.replicas * 260.0)

        # api base also includes dependency penalty
        lat_base = 22.0 + 80.0 * max(0.0, load_per_replica - 0.35) + api_dep_penalty
        err_base = 0.003 + 0.025 * max(0.0, load_per_replica - 0.6) + dep_err_penalty

        lmul, emul, cmul, mmul = self._fault_multiplier("api")

        st.cpu_util = min(1.2, 0.18 + 0.95 * load_per_replica * cmul + self.rng.random() * 0.05)
        st.mem_util = min(1.2, 0.22 + 0.55 * load_per_replica * mmul + self.rng.random() * 0.05)

        latency_ms_p95["api"] = max(8.0, (lat_base * lmul) + self.rng.gauss(0, 3.0))
        error_rate["api"] = min(0.9, max(0.0, (err_base * emul) + abs(self.rng.gauss(0, 0.002))))

        cpu_util["api"] = min(1.0, max(0.0, st.cpu_util))
        mem_util["api"] = min(1.0, max(0.0, st.mem_util))

        # This is my SLO: I treat “incident” as crossing a latency or error threshold.
        slo_violation = (latency_ms_p95["api"] > 220.0) or (error_rate["api"] > 0.06)

        return Telemetry(
            latency_ms_p95=latency_ms_p95,
            error_rate=error_rate,
            cpu_util=cpu_util,
            mem_util=mem_util,
            req_rate=req_rate,
            slo_violation=slo_violation,
        )

    # -------- Actions (the agent executes these) --------

    def restart(self, service: str) -> None:
        """
        Restart is a common SRE playbook action.
        In this toy world, I model restart by “resetting” CPU/mem pressure a bit
        and sometimes shortening the fault duration for specific fault types.
        """
        st = self.state[service]
        st.cpu_util = max(0.18, st.cpu_util * 0.6)
        st.mem_util = max(0.18, st.mem_util * 0.6)
        if service == "cache":
            st.cache_health = 1.0

        # Small simulation hack:
        # FaultEvent is frozen, but I still want “restart” to visibly help some faults.
        if self.active_fault and self.active_fault.service == service:
            if self.active_fault.fault_type in (FaultType.ERROR_BURST, FaultType.CACHE_POISON):
                remaining = max(0, self.active_fault.t_end - self.t)
                shrink = int(remaining * 0.6)
                object.__setattr__(self.active_fault, "duration", max(5, self.active_fault.duration - shrink))

    def scale(self, service: str, delta: int) -> None:
        """
        Scaling is another common action.
        I clamp replicas to [1, 10] so the agent can’t do something silly.
        """
        st = self.state[service]
        st.replicas = int(max(1, min(10, st.replicas + delta)))

    def clear_cache(self) -> None:
        """
        Cache clearing is a classic remedy for cache poisoning or stale data.
        Here, I model it as restoring cache health and strongly reducing cache_poison duration.
        """
        self.state["cache"].cache_health = 1.0
        if self.active_fault and self.active_fault.fault_type == FaultType.CACHE_POISON:
            object.__setattr__(self.active_fault, "duration", min(self.active_fault.duration, 8))

    def limit_traffic(self, factor: float) -> None:
        """
        Traffic limiting (rate limiting / load shedding) is a safe defensive action.
        It reduces workload and can stop cascades.
        """
        self._traffic_limit = float(max(0.4, min(1.0, factor)))

    def step(self) -> Telemetry:
        # Each step advances time and generates new telemetry.
        self.t += 1
        return self.observe()