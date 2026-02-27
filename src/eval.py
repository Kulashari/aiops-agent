from __future__ import annotations
import argparse
import pandas as pd

from .env.env import AIOpsEnv
from .agent.agent import AIOpsAgent


def run_episode(seed: int, steps: int):
    """
    This is my batch-eval helper.
    I run one episode end-to-end and return a single result row dict.
    """
    env = AIOpsEnv(seed=seed, steps=steps)
    tel = env.reset(seed=seed)
    agent = AIOpsAgent(seed=seed)

    t = 0
    anomalies = 0
    slo_steps = 0

    while True:
        t += 1
        agent_step, step_res = agent.step(env, tel, t)
        tel = step_res.telemetry

        # Count how many steps the detector flagged anomaly.
        if agent_step.detection.is_anomaly:
            anomalies += 1

        # Count how many steps we were in SLO violation.
        if agent_step.slo_violation:
            slo_steps += 1

        if step_res.done:
            info = step_res.info
            break

    # Compute MTTR-like proxy only if an incident actually happened and recovered.
    first_slo = info.get("first_slo_violation_t")
    recovered = info.get("recovered_t")

    mttr = None
    if first_slo is not None and recovered is not None:
        mttr = recovered - first_slo

    fault = info["fault"]

    return {
        "seed": seed,
        "fault_type": fault.fault_type.value,
        "fault_service": fault.service,
        "fault_t_start": fault.t_start,
        "fault_duration": fault.duration,
        "fault_severity": fault.severity,
        "anomaly_steps": anomalies,
        "slo_steps": slo_steps,
        "first_slo": first_slo,
        "recovered": recovered,
        "mttr_proxy": mttr,
    }


def main():
    """
    This script is purely for experiments:
    I run many episodes and dump results to a CSV.
    The goal is to quantify behavior (even if it’s imperfect).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--out", type=str, default="results.csv")
    ap.add_argument("--seed0", type=int, default=1)
    args = ap.parse_args()

    rows = []
    for i in range(args.episodes):
        seed = args.seed0 + i
        rows.append(run_episode(seed=seed, steps=args.steps))

        # I print progress occasionally so it doesn’t look “stuck”.
        if (i + 1) % 5 == 0 or (i + 1) == args.episodes:
            print(f"Finished {i+1}/{args.episodes} episodes...")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # A tiny summary so I can quickly sanity check that MTTR exists for some cases.
    print(df.groupby(["fault_type", "fault_service"])["mttr_proxy"].mean().sort_values().head(10))
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()