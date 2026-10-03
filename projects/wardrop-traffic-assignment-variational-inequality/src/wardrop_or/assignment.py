"""Path-flow formulation of Wardrop equilibrium as a variational inequality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class AssignmentResult:
    path_flows: np.ndarray
    link_flows: np.ndarray
    link_costs: np.ndarray
    path_costs: np.ndarray
    total_travel_time: float
    wardrop_gap: float


@dataclass(frozen=True)
class PathNetwork:
    """Single-OD separable traffic-assignment model.

    Link latency is t_a(x) = intercept_a + slope_a * x ** power_a.
    The link-path incidence matrix has shape (n_links, n_paths).
    """

    demand: float
    incidence: np.ndarray
    intercept: np.ndarray
    slope: np.ndarray
    power: np.ndarray

    def __post_init__(self) -> None:
        a = np.asarray(self.incidence, dtype=float)
        b = np.asarray(self.intercept, dtype=float)
        s = np.asarray(self.slope, dtype=float)
        p = np.asarray(self.power, dtype=float)
        if self.demand <= 0:
            raise ValueError("demand must be positive")
        if a.ndim != 2 or b.shape != (a.shape[0],) or s.shape != b.shape or p.shape != b.shape:
            raise ValueError("Inconsistent link/path dimensions")
        if np.any(a < 0) or np.any(b < 0) or np.any(s < 0) or np.any(p < 1):
            raise ValueError("Incidence and latency parameters must be nonnegative; powers >= 1")
        object.__setattr__(self, "incidence", a)
        object.__setattr__(self, "intercept", b)
        object.__setattr__(self, "slope", s)
        object.__setattr__(self, "power", p)

    def link_flows(self, path_flows: np.ndarray) -> np.ndarray:
        return self.incidence @ np.asarray(path_flows, dtype=float)

    def link_costs(self, link_flows: np.ndarray) -> np.ndarray:
        x = np.asarray(link_flows, dtype=float)
        return self.intercept + self.slope * x**self.power

    def path_costs(self, path_flows: np.ndarray) -> np.ndarray:
        x = self.link_flows(path_flows)
        return self.incidence.T @ self.link_costs(x)

    def beckmann_potential(self, path_flows: np.ndarray) -> float:
        x = self.link_flows(path_flows)
        integral = self.intercept * x + self.slope * x ** (self.power + 1.0) / (
            self.power + 1.0
        )
        return float(integral.sum())

    def total_travel_time(self, path_flows: np.ndarray) -> float:
        x = self.link_flows(path_flows)
        return float(x @ self.link_costs(x))

    def marginal_social_costs(self, link_flows: np.ndarray) -> np.ndarray:
        x = np.asarray(link_flows, dtype=float)
        latency = self.link_costs(x)
        derivative = self.slope * self.power * np.where(
            x > 0.0, x ** (self.power - 1.0), 0.0
        )
        return latency + x * derivative

    def pigouvian_tolls(self, link_flows: np.ndarray) -> np.ndarray:
        x = np.asarray(link_flows, dtype=float)
        return self.marginal_social_costs(x) - self.link_costs(x)

    def _solve(self, objective) -> np.ndarray:
        n_paths = self.incidence.shape[1]
        x0 = np.full(n_paths, self.demand / n_paths)
        constraints = [{"type": "eq", "fun": lambda f: float(np.sum(f) - self.demand)}]
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[(0.0, self.demand)] * n_paths,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            raise RuntimeError(result.message)
        return np.maximum(np.asarray(result.x, dtype=float), 0.0)

    def wardrop_gap(self, path_flows: np.ndarray) -> float:
        flows = np.asarray(path_flows, dtype=float)
        costs = self.path_costs(flows)
        return float(flows @ costs - self.demand * costs.min())

    def _result(self, flows: np.ndarray) -> AssignmentResult:
        link_flows = self.link_flows(flows)
        link_costs = self.link_costs(link_flows)
        path_costs = self.incidence.T @ link_costs
        return AssignmentResult(
            path_flows=flows,
            link_flows=link_flows,
            link_costs=link_costs,
            path_costs=path_costs,
            total_travel_time=float(link_flows @ link_costs),
            wardrop_gap=self.wardrop_gap(flows),
        )

    def user_equilibrium(self) -> AssignmentResult:
        """Solve Beckmann's convex program, equivalent to Wardrop UE here."""
        return self._result(self._solve(self.beckmann_potential))

    def system_optimum(self) -> AssignmentResult:
        """Minimize aggregate travel time under the same demand conservation."""
        return self._result(self._solve(self.total_travel_time))


def pigou_network(demand: float = 1.0) -> PathNetwork:
    """Classical Pigou network: t1(x)=x and t2(x)=1."""
    return PathNetwork(
        demand=demand,
        incidence=np.eye(2),
        intercept=np.array([0.0, 1.0]),
        slope=np.array([1.0, 0.0]),
        power=np.array([1.0, 1.0]),
    )
