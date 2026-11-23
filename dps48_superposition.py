"""Check exact counterexamples to additive superposition in the stored D(2) slices."""
from itertools import combinations
import json

import numpy as np

from audit import ROOT, modular_rank
from families import (FILE_NAMES, SUPPORTS, all_factors, common_coefficients,
                      exact_tensor, integer_evaluator, poly)
from reference.verify_certificate import jacobian_rows, require, target


def main():
    _, base = all_factors()
    base = base.astype(object)
    desired = 16**3 * target(4).astype(object)
    snapshots = []
    for file, support in zip(FILE_NAMES, SUPPORTS):
        chart = json.loads((ROOT / file).read_text())
        den = poly(chart["denominator"])
        part = np.array([poly(c) for c in chart["numerators"]], dtype=object).reshape(12, 3, 16)
        num = common_coefficients(den, part, base, np.array(support)-1)
        _, evaluate = integer_evaluator(den, num)
        scale, values = evaluate(1)
        require(scale and np.array_equal(exact_tensor(values), scale**3 * desired),
                "a single curve is undefined or fails Brent")
        snapshots.append((scale, values))

    failures = []
    for (i, (di, pi)), (j, (dj, pj)) in combinations(enumerate(snapshots, 1), 2):
        mixed = dj*pi + di*pj - (di*dj)*base
        residual = exact_tensor(mixed) - (di*dj)**3 * desired
        nz = np.argwhere(residual != 0)
        require(len(nz), f"pair {i}, {j} needs a polynomial compatibility check")
        failures.append({"charts": [i, j], "parameters": [1, 1],
                         "first_nonzero_coordinate_0_based": nz[0].tolist(),
                         "nonzero_brent_coordinates": len(nz)})
    support_ranks = []
    prime = 101
    for j, (scale, values) in enumerate(snapshots, 1):
        require(scale % prime, "sample has no good reduction at the certificate prime")
        reduced = (values % prime).astype(np.int32)
        require(np.all(np.any(reduced, axis=2)), "a sampled factor is zero")
        for i, support in enumerate(SUPPORTS, 1):
            if i == j:
                continue
            restricted = jacobian_rows(reduced[np.array(support)-1])
            rank = modular_rank(restricted, prime)
            require(rank == 552, f"restricted rank for support {i} on chart {j} is {rank}")
            support_ranks.append([i, j, rank])
    print(json.dumps({"DPS48_D2_additive_superposition": failures,
                      "generic_off_diagonal_support_ranks_mod_101": support_ranks}))


if __name__ == "__main__":
    main()
