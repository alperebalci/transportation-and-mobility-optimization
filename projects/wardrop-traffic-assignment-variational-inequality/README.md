# Wardrop Traffic Assignment as a Variational Inequality

This project adds a class of Operations Research problems that is not naturally described as "minimize one planner objective and stop."

It implements:

- Wardrop user equilibrium (UE);
- the Beckmann convex potential formulation;
- a variational-inequality/Wardrop residual;
- system-optimal (SO) traffic assignment;
- marginal social costs and Pigouvian congestion tolls;
- price-of-anarchy analysis.

The benchmark uses the classical Pigou two-route network:

- route 1 latency: `t_1(x)=x`;
- route 2 latency: `t_2(x)=1`;
- demand: 1.

The known results are recovered numerically:

- user equilibrium: all flow on route 1, total travel time 1;
- system optimum: 0.5 / 0.5 split, total travel time 0.75;
- price of anarchy: `4/3`.

## Why this is a distinct OR methodology

At user equilibrium, no individual traveler can improve travel time by unilaterally changing routes. That decentralized equilibrium generally differs from the centralized system optimum.

For separable monotone link costs, Beckmann's transformation solves the equilibrium through a convex program, while the residual

```text
sum_p f_p c_p - demand * min_p c_p
```

is zero at a single-OD Wardrop equilibrium.

The project then computes marginal-external-cost tolls showing how pricing can align decentralized incentives with the system optimum.

## Run

```bash
python -m pip install -e '.[dev]'
pytest
```

## Scope

The current implementation is a transparent single-OD path-flow model. Multi-OD link-based Frank-Wolfe assignment, elastic demand, stochastic user equilibrium, dynamic traffic assignment, and general VI projection/extragradient methods remain natural extensions.
