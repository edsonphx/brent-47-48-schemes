"""Verify the rational sixth jet at DPS48 and the restricted local tangent ranks."""
from itertools import combinations_with_replacement, product
import json
from math import comb, prod

import numpy as np

from audit import ROOT, ModularEchelon
from families import SUPPORTS, all_factors
from reference.verify_certificate import jacobian_rows, require, symmetry_directions, target


def monomials(degree):
    return list(combinations_with_replacement(range(10), degree))


def linear_coefficients(point, kernel):
    span = ModularEchelon(point.size, 3)
    for direction in symmetry_directions(point, True):
        span.insert(direction.ravel())
    require(len(span.pivots) == 141, "incorrect orbit dimension")
    linear = [v for v in kernel if span.insert(v) is None]
    require(len(linear) == 10, "incorrect transverse basis")
    return np.stack(linear)


def homogeneous_value(coefficients, indices, node):
    weights = np.array([prod(node[i] for i in word) for word in indices], dtype=np.int64)
    support = np.flatnonzero(weights)
    return weights[support] @ coefficients[support]


def sixth_jet():
    with np.load(ROOT / "inputs.npz", allow_pickle=False) as data:
        point = data["DPS48"].astype(np.int64)
    with np.load(ROOT / "rational-DPS48.npz", allow_pickle=False) as data:
        linear = linear_coefficients(point, data["kernel"])
        require(np.all(data["denominators"] == 1), "expected integral quadratic corrections")
        quadratic = -data["preimages"].astype(np.int64)

    # The original quadratic file orders (j,i) by i first. Higher degrees use lex order.
    words = [monomials(1), [(j, i) for i in range(10) for j in range(i + 1)],
             monomials(3), monomials(4), monomials(5), monomials(6)]
    with np.load(ROOT / "dps48-jets-Q.npz", allow_pickle=False) as data:
        coefficients = [linear, quadratic, data["cubic"], data["quartic"], data["quintic"], data["sextic"]]
    heights = [int(np.abs(matrix.astype(np.int64)).max()) for matrix in coefficients]
    for d, matrix in enumerate(coefficients, 1):
        require(matrix.shape == (len(words[d-1]), point.size), "wrong jet array shape")
    order = len(words)
    # The sum of degree-d monomial weights is at most order**d.
    bounds = [int(np.abs(point).max())] + [h * order**d for d, h in enumerate(heights, 1)]
    triples = [[powers for powers in product(range(d+1), repeat=3) if sum(powers) == d]
               for d in range(order+1)]
    for terms in triples:
        bound = len(point) * sum(prod(bounds[k] for k in powers) for powers in terms)
        require(bound < 2**63, "integer contraction could overflow")
    # The stored representative has target scale 16. audit.py independently checks this.
    u, v, w = point.transpose(1, 0, 2)
    require(np.array_equal(u.T @ (v[:, :, None]*w[:, None, :]).reshape(48, 256),
                           16*target(4).reshape(16, 256)), "wrong constant tensor")
    for node_word in words[-1]:
        node = np.bincount(node_word, minlength=10)
        jets = [point] + [homogeneous_value(matrix, indices, node).reshape(point.shape)
                          for matrix, indices in zip(coefficients, words)]
        for d in range(1, order+1):
            value = np.zeros((16, 256), dtype=np.int64)
            for i, j, k in triples[d]:
                paired = jets[j][:, 1, :, None] * jets[k][:, 2, None, :]
                value += jets[i][:, 0].T @ paired.reshape(48, 256)
            require(not np.any(value), f"nonzero degree {d} at simplex node {node.tolist()}")
    return {"field": "Q", "transverse_variables": 10, "total_degree": order,
            "homogeneous_coefficients": list(map(len, words)), "simplex_nodes": len(words[-1]),
            "coefficient_heights": heights,
            "ambient_hilbert_prefix": [comb(150+d, d) for d in range(order+1)]}


def restricted_ranks():
    raw, at_two = all_factors()
    at_quarter = 16 * raw[0] + 4 * raw[1] + raw[2]
    result = {}
    for label, point in [("1/4", at_quarter), ("2", at_two)]:
        ranks = []
        for support in SUPPORTS:
            restricted = (point[np.array(support)-1] % 3).astype(np.int32)
            rows = jacobian_rows(restricted)
            echelon = ModularEchelon(rows.shape[1], 3)
            for row in rows:
                echelon.insert(row)
            ranks.append(len(echelon.pivots))
        require(ranks == [551]*4, "restricted Jacobian minor missing")
        result[label] = ranks
    return result


if __name__ == "__main__":
    print(json.dumps({"DPS48_rational_jet": sixth_jet(),
                      "twelve_summand_ranks_mod3": restricted_ranks()}))
