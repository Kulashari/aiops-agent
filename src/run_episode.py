from __future__ import annotations
import argparse
from rich.console import Console
from rich.table import Table

from .env.env import AIOpsEnv
from .agent.agent import AIOpsAgent


def main():
    """
    This is my “demo runner”.
    I run exactly one episode and print a timeline so a reviewer can see:
    - what fault got injected
    - when the system violated SLO
    - what RCA I guessed
    - what actions I took
    - whether we recovered
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    console = Console()

    env = AIOpsEnv(seed=args.seed, steps=args.steps)
    tel = env.reset(seed=args.seed)
    agent = AIOpsAgent(seed=args.seed)

    fault = env.fault
    console.print(
        f"[bold]Injected fault:[/bold] {fault.fault_type.value} on [bold]{fault.service}[/bold] "
        f"at t={fault.t_start} for {fault.duration} steps (sev={fault.severity:.2f})"
    )

    timeline = []
    total_reward = 0.0
    t = 0

    while True:
        t += 1

        # Run one agent loop iteration.
        agent_step, step_res = agent.step(env, tel, t)
        total_reward += step_res.reward
        tel = step_res.telemetry
        timeline.append((agent_step, step_res))

        # If render is on, I print periodic updates (and also any “interesting” moments).
        if args.render and (t % 5 == 0 or agent_step.slo_violation or agent_step.detection.is_anomaly):
            console.print(
                f"t={t:03d} req={tel.req_rate:6.1f} api_lat={tel.latency_ms_p95['api']:7.1f}ms "
                f"api_err={tel.error_rate['api']*100:5.2f}% slo={tel.slo_violation} "
                f"RCA={agent_step.rca.primary_service}({agent_step.rca.confidence:.2f}) "
                f"act={agent_step.decision.action}"
            )

        if step_res.done:
            break

    # Pull incident timing info from the final step.
    first_slo = timeline[-1][1].info.get("first_slo_violation_t")
    recovered = timeline[-1][1].info.get("recovered_t")

    console.print("\n[bold]Incident Report[/bold]")
    console.print(f"First SLO violation at t={first_slo}")
    console.print(f"Recovered at t={recovered}")
    if first_slo is not None and recovered is not None:
        console.print(f"MTTR (proxy) = {recovered - first_slo} steps")
    console.print(f"Total reward = {total_reward:.2f}")

    # Summarize how many times each action was chosen.
    counts = {}
    for a, _ in timeline:
        k = a.decision.action["type"]
        counts[k] = counts.get(k, 0) + 1

    tab = Table(title="Action counts")
    tab.add_column("Action")
    tab.add_column("Count", justify="right")
    for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        tab.add_row(k, str(v))
    console.print(tab)


if __name__ == "__main__":
    main()