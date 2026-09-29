import numpy as np

from sumo_opt.control import SignalPlan, optimize_plan, rollout_cost


def test_optimizer_prefers_more_green_for_heavier_approach() -> None:
    rng=np.random.default_rng(3)
    scenarios=[]
    for _ in range(20):
        a=np.column_stack([rng.poisson(8,size=8),rng.poisson(2,size=8)]).astype(float)
        scenarios.append(a)
    plan=optimize_plan(scenarios,min_green=10,max_green=30,cycle=40)
    assert plan.green_a > plan.green_b


def test_rollout_is_finite() -> None:
    demand=np.array([[4,2],[5,1],[3,2]],dtype=float)
    assert np.isfinite(rollout_cost(demand,SignalPlan(20,20)))
