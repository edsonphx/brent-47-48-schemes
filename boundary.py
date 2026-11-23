"""Exact Laurent identity, gauge rank, and boundary certificates for the DPS48 curve."""
from collections import Counter
from itertools import combinations, permutations, product
import json

import numpy as np

from audit import ROOT, ModularEchelon
from reference.verify_certificate import jacobian_rows, require, symmetry_directions, target


def determinant(matrix):
    """Bareiss determinant with unbounded Python integers."""
    rows = [[int(x) for x in row] for row in matrix]
    sign, previous = 1, 1
    for k in range(len(rows) - 1):
        pivot = next((i for i in range(k, len(rows)) if rows[i][k]), None)
        if pivot is None:
            return 0
        if pivot != k:
            rows[k], rows[pivot] = rows[pivot], rows[k]
            sign = -sign
        value = rows[k][k]
        for i in range(k + 1, len(rows)):
            for j in range(k + 1, len(rows)):
                rows[i][j] = (value * rows[i][j] - rows[i][k] * rows[k][j]) // previous
            rows[i][k] = 0
        previous = value
    return sign * rows[-1][-1]


def sum_tensor(terms):
    return np.einsum("ri,rj,rk->ijk", *terms.transpose(1, 0, 2), dtype=np.int64)


def term_multiset(terms, multiplier=1):
    return Counter((multiplier * sum_tensor(term[None])).tobytes() for term in terms)


def evaluate_polynomial(coeff, t):
    return coeff[0] + t * (coeff[1] + t * coeff[2])


def loops(factors):
    u, v, w = factors.swapaxes(0, 1)
    w = w.swapaxes(1, 2)
    return u @ v @ w, v @ w @ u, w @ u @ v


def laurent_identity(coeff):
    block = np.zeros((7, 16, 16, 16), dtype=np.int64)
    for i, j, k in product(range(3), repeat=3):
        block[i+j+k] += np.einsum("ri,rj,rk->ijk", coeff[i, :, 0], coeff[j, :, 1],
                                  coeff[k, :, 2], dtype=np.int64)
    expected = np.zeros_like(block)
    expected[3] = 8**3 * target(4)
    require(np.array_equal(block, expected), "Laurent Brent identity failed")


def gauge_rank(coeff):
    p = 16 * coeff[0] + 4 * coeff[1] + coeff[2]
    derivative = 4 * (coeff[2] - 16 * coeff[0])
    with np.load(ROOT / "inputs.npz", allow_pickle=False) as data:
        dps = data["DPS48"].astype(np.int64)
    require(term_multiset(p) == term_multiset(dps, 32**3 // 16),
            "t = 1/4 did not reproduce DPS48 term by term")
    jac = jacobian_rows(p.astype(np.int32))
    require(not np.any(derivative.ravel() @ jac.astype(np.int64)), "curve derivative failed")
    span = ModularEchelon(p.size, 3)
    for generator in symmetry_directions(p, True):
        span.insert(generator.ravel())
    require(len(span.pivots) == 141, "unexpected orbit rank")
    span.insert(derivative.ravel())
    require(len(span.pivots) == 142, "curve is tangent to the symmetry orbit")
    return 1


def all_wedges(coeff):
    counts = []
    for mode in range(3):
        matrices = coeff[:, :, mode].transpose(1, 2, 0)
        count = 0
        for i, j in combinations(range(48), 2):
            for a, b in combinations(range(16), 2):
                minor = (np.convolve(matrices[i, a], matrices[j, b])
                         - np.convolve(matrices[i, b], matrices[j, a]))
                if np.count_nonzero(minor) == 1 and np.any(minor % 3):
                    count += 1
                    break
            else:
                raise ValueError(f"No unit wedge for mode {mode}, terms {(i+1, j+1)}")
        counts.append(count)
    return counts


def mixed_det(four_matrices):
    return sum(determinant([[four_matrices[layer, row, col] for col, layer in enumerate(p)]
                            for row in range(4)]) for p in permutations(range(4)))


def pole_certificates(mats):
    # Input arrays evaluate 8t times each Appendix A factor. Each mixed
    # determinant product therefore equals (8t)^12 times the invariant.
    certificates = [
        ((4, 25, 28, 40), 10, 2**37),
        ((12, 15, 32, 38), 14, 2**45),
    ]
    for t in range(25):
        block = evaluate_polynomial(mats, t)
        for labels, degree, coefficient in certificates:
            array = block[np.array(labels) - 1]
            value = 1
            for mode in range(3):
                value *= mixed_det(array[:, mode])
            require(value == coefficient * t**degree, f"bad pole identity for {labels}")
    return [{"terms_1_based": list(ids), "laurent_power": deg - 12,
             "coefficient": co // 8**12} for ids, deg, co in certificates]


def lattice_certificates(mats):
    # The first 15 rows are word length 1 and the final row has word
    # length 2. After clearing t, the determinant has degree <= 102.
    row_indices = [list(range(15)), list(range(15)), [*range(13), 14, 15]]
    certificates = [(149, 51, 1), (149, 51, -1), (155, 52, -1)]
    # Trace of a product of three degree-at-most-two matrices has degree <= 6.
    weights = np.trace(loops(evaluate_polynomial(mats, 1).astype(object))[0],
                       axis1=1, axis2=2)
    require(set(weights) == {8**3, 2*8**3}, "unexpected summand traces")
    for t in range(103):
        group = loops(evaluate_polynomial(mats, t).astype(object))
        if t < 7:
            require(np.array_equal(np.trace(group[0], axis1=1, axis2=2), weights * t**3),
                    "nonconstant or nonunit summand trace")
        for mode, (power, degree, sign) in enumerate(certificates):
            loop = group[mode]
            basis = list(loop[row_indices[mode]]) + [loop[0] @ loop[1]]
            value = determinant([row.ravel() for row in basis])
            require(value == sign * 2**power * t**degree, f"bad matrix-algebra span in mode {mode}")
    return [{"vertex": i, "basis_size": 16, "normalized_determinant_t_power": deg - 51,
             "sign": sign, "normalized_determinant_two_power": power - 153}
            for i, (power, deg, sign) in enumerate(certificates)]


def main():
    coeff = np.load(ROOT / "lwh-laurent.npy", allow_pickle=False)
    require(coeff.shape == (3, 48, 3, 16) and coeff.dtype == np.int64
            and np.max(np.abs(coeff)) < 256, "unexpected Laurent coefficient format")
    laurent_identity(coeff)
    wedges = all_wedges(coeff)
    mats = coeff.reshape(3, 48, 3, 4, 4)
    report = {"family": "Li-Wang-Hu Appendix A, anchored at DPS48",
              "fields": ["Q", "F3"], "brent_equations": 4096,
              "transverse_curve_rank": gauge_rank(coeff),
              "unit_pair_wedges_by_mode": wedges,
              "endpoint_pole_invariants": pole_certificates(mats),
              "lattice_determinants": lattice_certificates(mats)}
    print(json.dumps(report))


if __name__ == "__main__":
    main()
