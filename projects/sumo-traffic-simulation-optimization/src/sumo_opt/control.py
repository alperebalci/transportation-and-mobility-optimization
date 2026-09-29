"""Traffic-signal optimization core independent of SUMO runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SignalPlan:
    green_a: int
    green_b: int

    @property
    def cycle(self) -> int:
        return self.green_a + self.green_b


def queue_step(
    queues: ArrayLike,
    arrivals: ArrayLike,
    plan: SignalPlan,
    saturation: float = 1.0,
) -> NDArray[np.float64]:
    q = np.asarray(queues, dtype=float)
    a = np.asarray(arrivals, dtype=float)
    if q.shape != (2,) or a.shape != (2,):
        raise ValueError("queues and arrivals must have shape (2,)")
    service = saturation * np.array([plan.green_a, plan.green_b], dtype=float)
    return np.maximum(q + a - service, 0.0)


def rollout_cost(
    arrivals: ArrayLike,
    plan: SignalPlan,
    *,
    initial: ArrayLike=(0.0,0.0),
    saturation: float=1.0,
) -> float:
    demand=np.asarray(arrivals,dtype=float)
    q=np.asarray(initial,dtype=float)
    total=0.0
    for row in demand:
        q=queue_step(q,row,plan,saturation)
        total += float(q.sum())
    return total


def optimize_plan(
    scenarios: list[NDArray[np.float64]],
    *,
    min_green: int=5,
    max_green: int=30,
    cycle: int=40,
) -> SignalPlan:
    """Minimize mean simulated queue cost over a finite signal-plan set."""
    best: tuple[float,SignalPlan] | None=None
    for ga in range(min_green,max_green+1):
        gb=cycle-ga
        if gb < min_green or gb > max_green:
            continue
        plan=SignalPlan(ga,gb)
        cost=float(np.mean([rollout_cost(s,plan) for s in scenarios]))
        if best is None or cost < best[0]:
            best=(cost,plan)
    if best is None:
        raise ValueError("no feasible signal plan")
    return best[1]
