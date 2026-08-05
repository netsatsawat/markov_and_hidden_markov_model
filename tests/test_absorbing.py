"""Checks for the AbsorbingChain class.

Run: python tests/test_absorbing.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markov.absorbing import AbsorbingChain


# The loan portfolio chain used in the notebooks.
LOAN_P = np.array([
    [0.75, 0.06, 0.15, 0.04],   # Good Loan
    [0.05, 0.58, 0.02, 0.35],   # Risky Loan
    [0.00, 0.00, 1.00, 0.00],   # Paid Up (absorbing)
    [0.00, 0.00, 0.00, 1.00],   # Bad Loan (absorbing)
])
LOAN_STATES = ["GL", "RL", "PU", "BL"]


def test_detects_absorbing_states():
    chain = AbsorbingChain(LOAN_P, LOAN_STATES)
    assert chain.absorbing_states == ["PU", "BL"]
    assert chain.transient_states == ["GL", "RL"]


def test_absorption_probabilities_sum_to_one():
    chain = AbsorbingChain(LOAN_P, LOAN_STATES)
    B = chain.absorption_probabilities()
    assert np.allclose(B.sum(axis=1), 1.0)


def test_matches_long_run_matrix_powers():
    # Iterating the full matrix for a long time must agree with N R.
    chain = AbsorbingChain(LOAN_P, LOAN_STATES)
    B = chain.absorption_probabilities()
    P_inf = np.linalg.matrix_power(LOAN_P, 5000)
    for r, i in enumerate(chain.transient_idx):
        for c, j in enumerate(chain.absorbing_idx):
            assert abs(B[r, c] - P_inf[i, j]) < 1e-9


def test_matches_simulation():
    chain = AbsorbingChain(LOAN_P, LOAN_STATES)
    fractions, mean_steps = chain.simulate("RL", n_runs=20_000)
    B = chain.absorption_probabilities()
    t = chain.expected_steps()
    r = chain.transient_states.index("RL")
    assert abs(fractions["BL"] - B[r, chain.absorbing_states.index("BL")]) < 0.02
    assert abs(mean_steps - t[r]) < 0.1


def test_uniform_chain_matches_closed_form():
    # A linear chain of n steps that each succeed with probability p is the
    # simplest agent pipeline. End-to-end success must equal p ** n and, since
    # failures go undetected and every step always runs, a run always costs
    # n steps unless it fails early and absorbs.
    p, n = 0.85, 10
    states = [f"s{i}" for i in range(n)] + ["done", "failed"]
    P = np.zeros((n + 2, n + 2))
    for i in range(n):
        nxt = "done" if i == n - 1 else f"s{i + 1}"
        P[i, states.index(nxt)] = p
        P[i, states.index("failed")] = 1 - p
    P[states.index("done"), states.index("done")] = 1.0
    P[states.index("failed"), states.index("failed")] = 1.0

    chain = AbsorbingChain(P, states)
    B = chain.absorption_probabilities()
    success = B[0, chain.absorbing_states.index("done")]
    assert abs(success - p ** n) < 1e-12


def test_expected_cost_equals_weighted_visits():
    chain = AbsorbingChain(LOAN_P, LOAN_STATES)
    cost = np.array([2.0, 5.0])
    expected = chain.N @ cost
    assert np.allclose(chain.expected_cost(cost), expected)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
