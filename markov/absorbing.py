"""Exact answers for absorbing Markov chains.

An absorbing chain has at least one state you can enter but never leave.
Once the transition matrix is split into transient and absorbing parts
(the canonical form), three questions stop needing simulation:

1. How long until absorption, from each starting state?
2. Which absorbing state do we end up in, with what probability?
3. How many times do we visit each transient state along the way?

All three come from the fundamental matrix N = (I - Q)^-1, where Q is the
transient-to-transient block of the transition matrix.
"""

from __future__ import annotations

import numpy as np


class AbsorbingChain:
    """An absorbing Markov chain built from a full transition matrix.

    Parameters
    ----------
    P : array-like, shape (n, n)
        Row-stochastic transition matrix. Rows must sum to 1.
    states : list of str
        A label per state, in the same order as the rows of P.

    Absorbing states are detected automatically: any state whose row
    keeps all probability on itself (P[i, i] == 1).
    """

    def __init__(self, P, states):
        P = np.asarray(P, dtype=float)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("P must be a square matrix")
        if len(states) != P.shape[0]:
            raise ValueError("need one label per state")
        if not np.allclose(P.sum(axis=1), 1.0):
            raise ValueError("every row of P must sum to 1")

        self.P = P
        self.states = list(states)

        absorbing_mask = np.isclose(np.diag(P), 1.0)
        if not absorbing_mask.any():
            raise ValueError("no absorbing state found (no P[i, i] == 1)")
        if absorbing_mask.all():
            raise ValueError("every state is absorbing; nothing to analyze")

        self.transient_idx = np.where(~absorbing_mask)[0]
        self.absorbing_idx = np.where(absorbing_mask)[0]
        self.transient_states = [states[i] for i in self.transient_idx]
        self.absorbing_states = [states[i] for i in self.absorbing_idx]

        # Canonical form blocks: Q (transient -> transient), R (transient -> absorbing)
        self.Q = P[np.ix_(self.transient_idx, self.transient_idx)]
        self.R = P[np.ix_(self.transient_idx, self.absorbing_idx)]

        # Fundamental matrix. Entry N[i, j] is the expected number of visits
        # to transient state j when the chain starts in transient state i.
        self.N = np.linalg.inv(np.eye(len(self.transient_idx)) - self.Q)

    def expected_visits(self) -> np.ndarray:
        """Expected visits to each transient state, per starting state (N)."""
        return self.N

    def expected_steps(self) -> np.ndarray:
        """Expected number of steps before absorption, per starting state."""
        return self.N.sum(axis=1)

    def absorption_probabilities(self) -> np.ndarray:
        """Probability of ending in each absorbing state (B = N R).

        Rows are transient starting states, columns are absorbing states.
        """
        return self.N @ self.R

    def expected_cost(self, cost_per_state) -> np.ndarray:
        """Expected total cost accumulated before absorption.

        cost_per_state is the cost paid per visit to each transient state,
        in the order of `transient_states`. The result is N @ cost, one
        value per starting state.
        """
        cost = np.asarray(cost_per_state, dtype=float)
        if cost.shape != (len(self.transient_idx),):
            raise ValueError("need one cost per transient state")
        return self.N @ cost

    def simulate(self, start, n_runs=10_000, rng=None, max_steps=10_000):
        """Monte Carlo check on the analytic answers.

        Returns (absorption_counts, mean_steps): a dict mapping absorbing
        state label to the fraction of runs ending there, and the average
        number of steps before absorption.
        """
        rng = rng or np.random.default_rng(42)
        start_i = self.states.index(start) if isinstance(start, str) else start
        n = self.P.shape[0]
        absorbed_in = {s: 0 for s in self.absorbing_states}
        total_steps = 0
        for _ in range(n_runs):
            i, steps = start_i, 0
            while i not in self.absorbing_idx and steps < max_steps:
                i = rng.choice(n, p=self.P[i])
                steps += 1
            absorbed_in[self.states[i]] += 1
            total_steps += steps
        fractions = {s: c / n_runs for s, c in absorbed_in.items()}
        return fractions, total_steps / n_runs

    def summary(self) -> str:
        """A readable report of the three headline answers."""
        lines = []
        t_steps = self.expected_steps()
        B = self.absorption_probabilities()
        for r, s in enumerate(self.transient_states):
            dest = " · ".join(
                f"P({a}) = {B[r, c]:.4f}"
                for c, a in enumerate(self.absorbing_states))
            lines.append(f"start in {s}: {t_steps[r]:.2f} expected steps, {dest}")
        return "\n".join(lines)
