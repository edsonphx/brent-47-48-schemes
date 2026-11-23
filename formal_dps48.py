"""Lift the ten DPS48 tangent parameters through degree four over F3."""
from collections import defaultdict
from itertools import combinations, combinations_with_replacement, permutations
import json

import numpy as np

from audit import ROOT, ModularEchelon, quadratic, quadratic_columns
from reference.verify_certificate import jacobian_rows, require, symmetry_directions


def contract(a, b, c):
    return np.einsum("ri,rj,rk->ijk", a[:, 0], b[:, 1], c[:, 2],
                     dtype=np.int64).ravel()


def polar(point, u, v):
    return sum(contract(*triple) for triple in permutations((point, u, v)))


def output_coefficients():
    return defaultdict(lambda: np.zeros(4096, dtype=np.int64))


def formal_lift(point):
    jac = jacobian_rows(point.astype(np.int32))
    elimination = ModularEchelon(jac.shape[1], 3)
    tangent = ModularEchelon(point.size, 3)
    for direction in symmetry_directions(point, True):
        tangent.insert(direction.ravel())
    require(len(tangent.pivots) == 141, "unexpected DPS48 orbit rank")

    linear = []
    for i, row in enumerate(jac):
        relation = elimination.insert(row, i)
        if relation is not None:
            direction = elimination.relation(relation, point.size)
            if tangent.insert(direction) is None:
                linear.append(direction.reshape(point.shape))
    require(len(tangent.pivots) == 151 and len(linear) == 10,
            "wrong transverse basis size")
    require(len(elimination.pivots) == 2153, "wrong Brent Jacobian rank")

    def solve(residual, degree, monomial):
        relation = elimination.insert(residual, point.size)
        require(relation is not None, f"Degree {degree} failed at {monomial}")
        coeff = elimination.relation(relation, point.size + 1)
        require(coeff[-1] == 1, "invalid tagged Jacobian relation")
        return (coeff[:-1] % 3).reshape(point.shape)

    pairs = [(a, b) for b in range(len(linear)) for a in range(b + 1)]
    second = {ab: solve(r, 2, ab) for ab, r in zip(pairs, quadratic_columns(point, linear))}
    cubics = output_coefficients()
    for i, di in enumerate(linear):
        for jk, ejk in second.items():
            cubics[tuple(sorted((i, *jk)))] += polar(point, di, ejk)
    for index in combinations_with_replacement(range(len(linear)), 3):
        for a, b, c in sorted(set(permutations(index))):
            cubics[index] += contract(linear[a], linear[b], linear[c])
    third = {abc: solve(r, 3, abc) for abc, r in cubics.items()}

    quartics = output_coefficients()
    for i, di in enumerate(linear):
        for jkl, ejkl in third.items():
            quartics[tuple(sorted((i, *jkl)))] += polar(point, di, ejkl)
    for ab, eab in second.items():
        quartics[tuple(sorted((*ab, *ab)))] += quadratic(point, eab)
    for (ab, eab), (cd, ecd) in combinations(second.items(), 2):
        quartics[tuple(sorted((*ab, *cd)))] += polar(point, eab, ecd)
    for i, di in enumerate(linear):
        for j, dj in enumerate(linear):
            for kl, qkl in second.items():
                quartics[tuple(sorted((i, j, *kl)))] += (
                    contract(di, dj, qkl) + contract(di, qkl, dj) + contract(qkl, di, dj))
    for monomial, residual in quartics.items():
        solve(residual, 4, monomial)

    require((len(second), len(third), len(quartics)) == (55, 220, 715),
            "monomial count mismatch")
    return {"scheme": "DPS48", "field": "F3", "transverse_parameters": 10,
            "formal_lift_through_total_degree": 4,
            "preimages_by_degree": {"2": len(second), "3": len(third), "4": len(quartics)}}


def main():
    with np.load(ROOT / "inputs.npz", allow_pickle=False) as data:
        report = formal_lift(data["DPS48"].astype(np.int64) % 3)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
