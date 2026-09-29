"""Optional SUMO/TraCI adapter."""

from __future__ import annotations

from .control import SignalPlan


def apply_signal_plan(tls_id: str, plan: SignalPlan) -> None:
    try:
        import traci
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the optional SUMO/TraCI dependency") from exc
    logic=traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phases=list(logic.phases)
    if len(phases) < 2:
        raise RuntimeError("traffic light needs at least two phases")
    phases[0].duration=plan.green_a
    phases[1].duration=plan.green_b
    logic.phases=tuple(phases)
    traci.trafficlight.setProgramLogic(tls_id,logic)
