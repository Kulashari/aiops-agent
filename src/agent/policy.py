from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import math

from .diagnoser import RCAHypothesis
from ..env.system import Telemetry


@dataclass
class Decision:
    """
    I keep decisions as:
    - an action dict (the executor/env can apply it)
    - a rationale string (so I can explain the agent’s behavior)
    """
    action: Dict
    rationale: str


class UCB1Bandit:
    """
    This is a lightweight “learning” component.
    Instead of full RL, I use a bandit:
    - For each incident signature, I learn which action tends to give better reward.
    - UCB1 balances exploration vs exploitation.

    It’s simple, explainable, and it demonstrates adaptive behavior.
    """

    def __init__(self):
        self.counts: Dict[str, Dict[str, int]] = {}
        self.values: Dict[str, Dict[str, float]] = {}

    def _key(self, action: Dict) -> str:
        # I normalize action dicts into stable keys.
        if action["type"] in ("restart", "scale"):
            return action["type"] + ":" + action.get("service", "?")
        if action["type"] == "limit_traffic":
            return "limit_traffic"
        return action["type"]

    def select(self, sig: str, candidates: List[Dict]) -> Dict:
        # Initialize per-signature state.
        self.counts.setdefault(sig, {})
        self.values.setdefault(sig, {})
        total = sum(self.counts[sig].values()) + 1

        # First I try any actions I haven’t tried yet (pure exploration).
        for a in candidates:
            k = self._key(a)
            if self.counts[sig].get(k, 0) == 0:
                return a

        # Otherwise I pick the action with the best UCB score.
        best_a = candidates[0]
        best_ucb = -1e9
        for a in candidates:
            k = self._key(a)
            n = self.counts[sig][k]
            v = self.values[sig][k]
            ucb = v + math.sqrt(2.0 * math.log(total) / n)
            if ucb > best_ucb:
                best_ucb = ucb
                best_a = a

        return best_a

    def update(self, sig: str, action: Dict, reward: float) -> None:
        # Online mean update: value <- value + (reward - value)/n
        k = self._key(action)
        self.counts.setdefault(sig, {})
        self.values.setdefault(sig, {})
        self.counts[sig][k] = self.counts[sig].get(k, 0) + 1
        n = self.counts[sig][k]
        old = self.values[sig].get(k, 0.0)
        self.values[sig][k] = old + (reward - old) / n


class Policy:
    """
    This is where I decide actions.
    I combine:
    - “playbook candidates” (restart/scale/clear_cache/limit_traffic)
    - safety guardrails (restart cooldown)
    - optional bandit learning to pick among candidates
    """

    def __init__(self, use_bandit: bool = True):
        self.use_bandit = use_bandit
        self.bandit = UCB1Bandit() if use_bandit else None
        self._cooldowns: Dict[str, int] = {}  # action_key -> next_allowed_t

    def _cooldown_ok(self, key: str, t: int) -> bool:
        # I prevent spammy restarts because that’s bad practice in real ops too.
        return t >= self._cooldowns.get(key, -10**9)

    def _set_cooldown(self, key: str, t: int, cd: int) -> None:
        self._cooldowns[key] = t + cd

    def decide(self, tel: Telemetry, rca: RCAHypothesis, t: int) -> Decision:
        """
        Given telemetry + RCA hypothesis, I build a candidate action list and choose one.

        I also compute an “incident signature” so the bandit can learn per-situation.
        """
        svc = rca.primary_service
        sig = f"{svc}|lat>{int(tel.latency_ms_p95['api']>220)}|err>{int(tel.error_rate['api']>0.06)}"

        # Start with noop so there’s always a “do nothing” option.
        candidates: List[Dict] = [{"type": "noop"}]

        # “Playbook-style” candidates based on the suspected service.
        if svc in ("api", "db"):
            candidates.append({"type": "restart", "service": svc})
            candidates.append({"type": "scale", "service": svc, "delta": 1})
        if svc == "cache":
            candidates.append({"type": "clear_cache"})
            candidates.append({"type": "restart", "service": "cache"})

        # Load shedding is a general safety action.
        candidates.append({"type": "limit_traffic", "factor": 0.7})

        # Apply restart cooldown.
        filtered: List[Dict] = []
        for a in candidates:
            if a["type"] == "restart":
                key = f"restart:{a['service']}"
                if self._cooldown_ok(key, t):
                    filtered.append(a)
            else:
                filtered.append(a)

        # Choose action.
        if self.use_bandit and self.bandit:
            chosen = self.bandit.select(sig, filtered)
            rationale = (
                f"I used a bandit to pick an action for signature={sig}. "
                f"My RCA guess was {svc} (conf={rca.confidence:.2f})."
            )
        else:
            # Simple deterministic fallback (still “playbook-y”).
            if svc == "cache":
                chosen = {"type": "clear_cache"}
                rationale = "I suspect cache issues, so I clear cache first."
            elif tel.cpu_util.get(svc, 0.0) > 0.85:
                chosen = {"type": "scale", "service": svc, "delta": 1}
                rationale = f"{svc} CPU looks high, so I scale it up."
            else:
                chosen = {"type": "restart", "service": svc}
                rationale = f"I restart {svc} as a general mitigation."

        # Enforce cooldown if we restarted.
        if chosen["type"] == "restart":
            self._set_cooldown(f"restart:{chosen['service']}", t, cd=20)

        return Decision(action=chosen, rationale=rationale)

    def learn(
        self,
        tel_before: Telemetry,
        tel_after: Telemetry,
        rca: RCAHypothesis,
        action: Dict,
        reward: float,
    ) -> None:
        """
        After taking an action, I use the reward as feedback.
        This is my “autonomous improvement” piece: I adapt action selection over time.
        """
        if not (self.use_bandit and self.bandit):
            return

        svc = rca.primary_service
        sig = f"{svc}|lat>{int(tel_before.latency_ms_p95['api']>220)}|err>{int(tel_before.error_rate['api']>0.06)}"
        self.bandit.update(sig, action, reward)