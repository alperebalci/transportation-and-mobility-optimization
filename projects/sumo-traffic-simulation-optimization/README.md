# SUMO Traffic Simulation Optimization

A closed-loop mobility project connecting traffic-control optimization with a SUMO/TraCI execution boundary.

The open-source CI path contains a deterministic two-approach queue simulator and finite signal-plan optimizer. The optional TraCI adapter applies the selected plan to a running SUMO traffic light.

Architecture:

`historical/forecast demand -> candidate signal plans -> simulator evaluation -> selected plan -> TraCI/SUMO -> observed queues -> re-optimization`

The queue model is deliberately not presented as SUMO itself. Its role is to keep optimization logic testable without requiring a graphical/network SUMO installation in CI. A production study should replace the surrogate evaluator with repeated SUMO runs, common random numbers, network-level delay/emission KPIs, and rolling-horizon control.

Natural extensions: multi-intersection coordination, MPC, constrained RL challenger policies, incident scenarios, emissions, and TraCI/libsumo performance comparison.
