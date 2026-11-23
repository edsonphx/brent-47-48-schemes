"""Reconstruct the three higher rational jet arrays, using optional python-flint."""
import argparse
from itertools import combinations_with_replacement, product
from pathlib import Path

import numpy as np
from flint import nmod_mat

from audit import ROOT, ModularEchelon, exact_product
from dps48_local import linear_coefficients, monomials
from reference.verify_certificate import jacobian_rows, require


PRIME = 1_000_000_007


def jacobian_minor(jac):
    span = ModularEchelon(jac.shape[1], 3)
    variables = []
    for i in np.argsort(np.count_nonzero(jac, axis=1), kind="stable"):
        if span.insert(jac[i]) is None:
            variables.append(i)
    equations = [bit.bit_length()-1 for bit in span.pivots]
    require(len(variables) == len(equations) == 2153, "incomplete pivot minor")
    return variables, equations


def residuals(jets, degree):
    words = list(combinations_with_replacement(range(10), degree))
    result = {w: np.zeros((16, 256), dtype=np.int64) for w in words}
    heights = [max(int(np.abs(x).max()) for x in step.values()) for step in jets]
    bound = 0
    for i, j, k in product(range(degree), repeat=3):
        if i+j+k != degree:
            continue
        # Bound even the sum of all coefficients contributing to all monomials.
        bound += (48 * len(jets[i])*len(jets[j])*len(jets[k])
                  * heights[i]*heights[j]*heights[k])
        require(bound < 2**63, "integer contraction could overflow")
        for b, second in jets[j].items():
            for c, third in jets[k].items():
                pair = (second[:, 1, :, None] * third[:, 2, None, :]).reshape(48, 256)
                for a, first in jets[i].items():
                    result[tuple(sorted(a+b+c))] += first[:, 0].T @ pair
    return np.stack([v.ravel() for v in result.values()])


def reconstruct():
    with np.load(ROOT / "inputs.npz", allow_pickle=False) as data:
        p = data["DPS48"].astype(np.int64)
    with np.load(ROOT / "rational-DPS48.npz", allow_pickle=False) as data:
        linear = linear_coefficients(p, data["kernel"])
        require(np.all(data["denominators"] == 1), "nonintegral quadratic corrections")
        second = -data["preimages"].astype(np.int64)
    pairs = [(j, i) for i in range(10) for j in range(i+1)]
    jets = [{(): p}, dict(zip(monomials(1), linear.reshape(-1, *p.shape))),
            dict(zip(pairs, second.reshape(-1, *p.shape)))]
    jac = jacobian_rows(p.astype(np.int32))
    variables, equations = jacobian_minor(jac)
    minor = jac[np.ix_(variables, equations)].T.astype(np.int64)
    matrix = nmod_mat((minor % PRIME).tolist(), PRIME)
    output = {}
    for degree, key in [(3, "cubic"), (4, "quartic"), (5, "quintic")]:
        residual = residuals(jets, degree)
        rhs = nmod_mat(((-residual[:, equations].T) % PRIME).tolist(), PRIME)
        solution = matrix.solve(rhs)
        centered = np.array(solution.tolist(), dtype=np.int64).T
        centered[centered > PRIME//2] -= PRIME
        lifts = np.zeros((len(residual), p.size), dtype=np.int64)
        lifts[:, variables] = centered
        require(np.array_equal(exact_product(lifts, jac), -residual),
                "modular reconstruction fails an integer equation")
        jets.append(dict(zip(monomials(degree), lifts.reshape(-1, *p.shape))))
        output[key] = lifts.astype(np.int32)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write to a new NPZ file")
    args = parser.parse_args()
    arrays = reconstruct()
    with np.load(ROOT / "dps48-jets-Q.npz", allow_pickle=False) as reference:
        for key, values in arrays.items():
            require(np.array_equal(values, reference[key]), f"{key} array differs")
    if args.output:
        with args.output.open("xb") as output:
            np.savez_compressed(output, **arrays)
    print("All three rational jet arrays reconstructed exactly.")
