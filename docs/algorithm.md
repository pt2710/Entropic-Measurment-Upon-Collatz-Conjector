# Algorithm Notes

This project investigates the classic Collatz iteration using an entropy inspired perspective. For each starting value (`Seed`) we generate its Collatz sequence until reaching the cycle `4 → 2 → 1`. Parity counts of even and odd steps are recorded and several quantities are derived:

- **$\hat{H}$** – a normalized measure based on parity imbalance that evolves along the sequence.
- **$\pi E$ norm** – combines parity counts into a single clustering statistic.

The output of the sweep is classified into six empirical "laws":

1. **Convergence** – sequences must end with $\hat{H}=0$.
2. **Dyadic Immediacy** – powers of two show distinct behaviour at the start of their trajectories.
3. **Start Plateaux** – only a limited set of initial $\hat{H}$ values occur.
4. **Monotone Spike** – the maximum $\hat{H}$ value always occurs after the first valid step.
5. **Elastic–$\pi$ Clustering** – seeds group naturally according to their $\pi E$ norm.
6. **Parity Neutrality** – parity codes distribute evenly between even, odd and neutral states.

See the main script for implementation details.
