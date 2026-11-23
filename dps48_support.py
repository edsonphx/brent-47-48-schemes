"""Certify the minimum external support of a sixth tangent at D(2)."""
from itertools import combinations
import json

import numpy as np

from audit import ModularEchelon
from families import GROUPS, all_factors
from reference.verify_certificate import jacobian_rows, require, symmetry_directions


LINES = (
    (1, 5, 6, 7, 10, 14, 31, 36),
    (16, 20, 21, 23, 27, 29, 43, 46),
    (2, 8, 13, 18, 26, 30, 33, 39),
    (3, 11, 24, 37, 41, 42, 45, 47),
)
ACTIVE_EXTERNAL = (5, 6, 8, 10, 11, 13, 16, 18, 20, 23, 24, 26, 36, 37, 41, 46)


def span(rows):
    echelon = ModularEchelon(rows.shape[1], 3)
    for row in rows:
        echelon.insert(row)
    return echelon


def kernel(jacobian, variables):
    echelon = ModularEchelon(jacobian.shape[1], 3)
    vectors = []
    for i in variables:
        relation = echelon.insert(jacobian[i], tag=int(i))
        if relation is not None:
            vectors.append(echelon.relation(relation, jacobian.shape[0]))
    return len(echelon.pivots), np.array(vectors)


def remove_scalings(point, tangents):
    representatives = tangents.reshape(-1, *point.shape).copy()
    for term in range(len(point)):
        for mode in (0, 1):
            first = np.flatnonzero(point[term, mode])[0]
            coefficient = representatives[:, term, mode, first] * point[term, mode, first] % 3
            representatives[:, term, mode] -= coefficient[:, None] * point[term, mode]
            representatives[:, term, 2] += coefficient[:, None] * point[term, 2]
    return representatives.reshape(-1, point.size) % 3


def main():
    raw, base = all_factors()
    # The data use the uniform factor scale 16 at t=2.
    point = (base % 3).astype(np.int32)
    require(np.all(np.any(point, axis=2)), "a reduced factor is zero")
    jacobian = jacobian_rows(point)
    variables = np.arange(point.size).reshape(48, 48)
    core = sum(GROUPS, [])
    external = set(range(1, 49)) - set(core)
    require(set(sum(LINES, ())) == external and sum(map(len, LINES)) == len(external),
            "stationary lines do not partition the external terms")
    require(len(set(ACTIVE_EXTERNAL)) == 16 and set(ACTIVE_EXTERNAL) <= external,
            "invalid external support")

    rank, tangents = kernel(jacobian, variables.ravel())
    orbit = np.array(list(symmetry_directions(point, True))).reshape(-1, point.size)
    quotient = ModularEchelon(point.size, 3)
    basis = [v for v in remove_scalings(point, tangents) if quotient.insert(v) is None]
    require((rank, len(tangents), len(span(orbit).pivots), len(basis)) == (2153, 151, 141, 55),
            "unexpected tangent, orbit, or scaling quotient rank")
    basis = np.array(basis).reshape(55, 48, 48)

    _, core_tangents = kernel(jacobian, variables[np.array(core)-1].ravel())
    core_projected = remove_scalings(point, core_tangents)
    require(len(span(core_projected).pivots) == 4, "wrong core dimension modulo scalings")
    orbit_core = np.concatenate((orbit, core_tangents))
    require(len(span(orbit_core).pivots) == 145, "wrong core dimension modulo symmetries")
    # d(16 D(t))/dt = 2*(P_+ - P_-/t^2), and t=2 is a unit mod 3.
    derivative = (2 * (raw[2] - raw[0])).ravel() % 3
    require(not np.any((derivative @ jacobian) % 3), "published curve derivative is not tangent")
    known = span(np.vstack((orbit_core, derivative)))
    require(len(known.pivots) == 146, "incorrect span of known directions")

    covers = []
    for line in LINES:
        quartets = [
            set(group) for group in combinations(line, 4)
            if len(span(basis[:, np.array(group)-1].reshape(55, -1).T).pivots) == 51
        ]
        covered = sum(any(q.issubset(group) for q in quartets)
                      for group in combinations(line, 5))
        require((len(quartets), covered) == (40, 56), "invalid stationary quartet cover")
        covers.append([len(quartets), covered])

    selected = variables[np.array([*core, *ACTIVE_EXTERNAL])-1].ravel()
    restricted_rank, candidates = kernel(jacobian, selected)
    require((restricted_rank, len(candidates)) == (1467, 69), "incorrect restricted rank")
    witness = None
    for v in candidates:
        if known.insert(v) is None:
            require(witness is None, "more than one new direction on certified support")
            witness = v
    require(witness is not None and len(known.pivots) == 147, "missing sixth direction")
    require(not np.any((witness @ jacobian) % 3), "support witness is not tangent")
    touched = {i+1 for i in np.flatnonzero(np.any(witness.reshape(48, 48) % 3, axis=1))}
    require(touched & external == set(ACTIVE_EXTERNAL), "incorrect external witness support")
    print(json.dumps({
        "DPS48_D2_field": "F3", "minimum_external_support": 16,
        "rank51_quartets_and_covered_five_sets": covers,
        "support_restricted_jacobian_rank": restricted_rank,
        "known_span_rank": 146, "enlarged_span_rank": len(known.pivots),
        "active_external_terms_1_based": ACTIVE_EXTERNAL,
        "witness_sparse_coordinates_0_based": [[int(i), int(witness[i]) % 3]
                                              for i in np.flatnonzero(witness % 3)],
    }))


if __name__ == "__main__":
    main()
