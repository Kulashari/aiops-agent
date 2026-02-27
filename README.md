<div align="center">

AIOps Agent Prototype 
=========================

</div>

This repo is a small prototype of an AIOps agent. Not production grade level but the goal is to show 
the core loop you'd expect from an autonomous operations system:

<b>Observe → Detect Problems → Predict The Cause → Choose a Suitable Action → Measure if Things Were Improved → Learn</b>

Everything runs in a local simulation so the program can be iterated quickly and be able to evaluate behavior across many scenarios.

<h2>Setup Instructions</h2>

<h3>1) Install dependencies</h3>

Make sure you have **Python 3.10+** and `pip` installed, then run from the project root:

```pip install -r requirements.txt```

<h3>2) Run a single simulation (demo)</h3>

This prints a timeline plus a final incident report:

```python -m src.run_episode --steps 240 --seed 7 --render```

<h3>3) Run multiple simulation evaluation (writes CSV report)</h3>

This runs 50 simulations and saves result:

```python -m src.eval --episodes 50 --steps 240 --out results.csv```

The ```results.csv``` file includes: 
* fault type/service/severity/timing
* number of stemps falgged as anomalies
* number of SLO violation steps
* first SLO violation time, recovery time

<h2>Architecture / Design</h2>
<h3>High level loop</h3>

At every timestep, the agent runs the same cycle: 

1. <b>Observe</b> current state from the system

2. <b>Detect</b> whether the state looks anomalous 
3. <b>Diagonse</b> the root cause (RCA hypothesis)
4. <b>Decide</b> an automated action 
5. <b>Act</b> by applying the action to the system
6. <b>Learn</b> from a reward signal (did things improve?)

<h3>Components</h3>

The implementation is split into small sections so each part is easy to explain and swap.

<b>Enviroment(</b>src/env/<b>)</b>

* ```system.py```: Simulates a microservice system:
    * services: api (user facing operations), db, cache
    * emits metrics: p95 latency, error rate, CPU, memory, request rate

* ```faults.py```: Defines fault types and samples one fault per episode

* ```env.py```: Wraps the system into an episode:
    * tracks incident start (first_slo) and recovery (recovered)
    * computes an a similar MTTR metric (mttr_proxy)
    * emits a simple reward signal for learning


<b>Agent(</b>src/agent/<b>)</b>

* ```detector.py```: Unsupervised problem detection (Isolation Forest per service)

* ```diagnoser.py```: RCA hypothesis (ranks anomaly and correlation to API symptoms)
* ```policy.py```: Action selection (playbooks + safetu guardrails+ optional bandit learning)
* ```agent.py```: Orchestartes the loop end to end

<b>Runners</b>

* ```run_episode.py```: Runs one simulation and prints a readable timeline (good for demos+debugging)
* ```eval.py```: Runs many simulations and writes the results to CSV format

<h2>AI Techniques / Decision Logic</h2>
<h3>1) Unsupervised anomaly detection (Isolation Forest)</h3>
I use isolation forest to score how off each service looks based on its metrics:

* festures per service: (p95 latency, error rate, CPU utilization, memory utilization)
* one model per service
* trained on early observations in the simulation

This is a strong choice for ops settings because annomaly detection is often weakly labeled in practice.

The detector prodcues: 
* per service anomaly scores
* a global anomaly score (mean across services)
* a boolean anomaly trigger based on thresholds

<h3>2) RCA hypothesis (explainable heuristics)</h3>

When the system looks unhealthy (anomaly trigger or SLO breach), I generate a root cause hypothesis by combinig:
* anomaly score ranking (what service looks most abnormal)
* correlation with API symptoms (what service metrics move with API latency/error)

Output:
* Suspected service (api/db/cache)
* confidence (heuristic)
* rationale (text explanation)

It doesn't include deep learning but it's easy to debug and starts off as a baseline in AIOps systems. 

<h3>3) Automated mitigation + lightweight learning (bandit)</h3>

The agent chooses an action from various avaliable methods that represent the actions to fix the anomaly: 
* ```restart(service)```
* ```scale (service, +1)```
* ```clear_cache()```
* ```limit_traffic(0.7)```
* ```noop```

To show autonomous adaptation without a heavy reinforcement learning infrastructure, I use a multi armed bandit (UCB algorithim) keyed by an incident signature (suspected service + if the API latenct/error is past threshold). Over multiple simulations, the bandit learns which action tend to output better reward for similar simulations.

I also added basic safety constraints: 
* restart cooldown (prevents restart spamming)
* replica limits (prevents infinite scaling)

<h2>Assumptions</h2>

I made the following assumptions to simplify the implementation process to keep the prototype small: 
* Fake enviroment: Observations are simulated and don't rely on real data
* One fault per simulation: makes runs easier to interpret
* Simple fault effects: faults are implemented as multipliers
* Simplified actions: restart/scale/etc. have immediate effects in the simulation
* SLO definition is fixed: incidents are defined by API latency/error thresholds
* Reward is shaped: I rewarded the model on healthy operations and penalize time spent in SLO violations

Because of these assumptions, the output isn't perfect but enough to demonstrate the AIOps loop and see the behaviour.

<h2>Future Improvements</h2>

Since I was in a big time crunch from this project, school, and other commitments, the solution isn't perfect.
But here's what I'd improve next time: 

<h3>1. Add logs and traces</h3>

* Generate log events, trace the anomaly and solution
* Make diagnosis reason reason over metrics + logs + traces

<h3>2. Stronger RCA</h3>

* Replace correlation heuristics with causal discovery / Bayesian reasoning
* Track dependency graphs and guess the casual influence

<h3>3. Better learning signal</h3>

* Use recovery based reward
* INtroduce contextual bandits or reinforcement learning with better state reprenstation

<h3>4. Baseline comparisons</h3>

* Compares against:
    * pure threshold rules (no ML)
    * anomaly only (no RCA)

<h3>5. Integration path</h3>

* Emit metrics in Prometheus like format
* Replace the simulator with real data pipeline