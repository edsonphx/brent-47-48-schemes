"""Verify explicit binary and rational matrix multiplication orbit equivalences."""
from fractions import Fraction
from itertools import permutations
from collections import Counter
import json

import numpy as np

from audit import ROOT, ModularEchelon
from boundary import determinant
from reference.verify_certificate import require


def modes(point):
    matrices = point.reshape(-1, 3, 4, 4).copy()
    matrices[:, 2] = matrices[:, 2].swapaxes(-1, -2)
    return matrices


def binary_rank(matrix):
    span = ModularEchelon(4, 2)
    for row in matrix:
        span.insert(row)
    return len(span.pivots)


def binary_fingerprint(point):
    return sorted(Counter(tuple(sorted(binary_rank(m) for m in term))
                          for term in modes(point)).items())


COMMUTANT_TERMS = {
    "AT47": ((2, 15, 12), (2, 3, 4, 14), (2, 15, 6)),
    "KM47": ((2, 10, 8), (18, 6, 1), (2, 10, 4)),
    "C3": ((2, 11, 1), (2, 1, 3), (2, 8, 1)),
    "C4": ((4, 1, 9), (4, 6, 1), (4, 1, 2)),
    "C5": ((11, 47), (11, 47), (11, 6, 3)),
}


def check_commutants(data):
    units = np.eye(16, dtype=np.int64).reshape(16, 4, 4)
    for name, vertex_terms in COMMUTANT_TERMS.items():
        point = modes(data[name])
        require(np.all(np.any(point, axis=(2, 3))), "a binary factor is zero")
        for m, indices in enumerate(vertex_terms):
            terms = point[np.array(indices)-1]
            loops = terms[:, m] @ terms[:, (m+1) % 3] @ terms[:, (m+2) % 3]
            span = ModularEchelon(16, 2)
            for loop in loops:
                for row in (units @ loop - loop @ units).reshape(16, 16).T:
                    span.insert(row)
            require(len(span.pivots) == 15, f"{name}: non-scalar commutant at vertex {m}")
    print(json.dumps({"binary_scalar_commutant_terms_1_based": COMMUTANT_TERMS,
                      "commutator_rank_per_vertex": 15}))


def check(data, filename, prime):
    cert = json.loads((ROOT/filename).read_text())
    a = modes(data[cert["source"]])
    b = modes(data[cert["target"]])
    order = cert["mode_order"]
    require(tuple(order) in set(permutations(range(3))), "invalid mode permutation")
    swaps = sum(order[i] > order[j] for i in range(3) for j in range(i+1, 3)) % 2
    require(bool(swaps) == cert["transpose_all"], "invalid cyclic matrix action")
    a = a[:, order]
    if cert["transpose_all"]:
        a = a.swapaxes(-1, -2)
    perm = cert["term_permutation"]
    require(sorted(perm) == list(range(len(a))) and len(a) == len(b), "invalid term permutation")
    b = b[perm]
    matrices = np.array(cert["vertex_matrices"], dtype=np.int64)
    for g in matrices:
        require((determinant(g) % prime if prime else determinant(g)) != 0,
                "singular change of basis")
    if prime:
        scalars = np.ones((len(a), 3), dtype=np.int64)
    else:
        scalars = np.array([[Fraction(x) for x in row] for row in cert["scalars"]], dtype=object)
        require(np.all(scalars.prod(axis=1) == 1), "invalid term rescaling")
    for i in range(3):
        moved = (matrices[i] @ a[:, i])*scalars[:, i, None, None]
        expected = b[:, i] @ matrices[(i+1) % 3]
        require(not np.any((moved - expected) % prime if prime else moved - expected),
                "orbit certificate failed")
    print(json.dumps({"field": f"F{prime}" if prime else "Q", "source": cert["source"],
                      "target": cert["target"], "equivalence": True,
                      "matched_terms": len(a), "mode_order_zero_based": order}))


def main():
    with np.load(ROOT/"inputs.npz", allow_pickle=False) as data:
        check_commutants(data)
        check(data, "gamma-equivalence.json", 2)
        check(data, "dps48-equivalence.json", None)
        points = ["AT47", "KM47", "C3", "C4", "C5"]
        fingerprints = {k: binary_fingerprint(data[k]) for k in points}
        require(len(set(map(tuple, fingerprints.values()))) == len(points),
                "rank triple invariant did not separate the binary reps")
        print(json.dumps({"distinct_binary_orbits": len(points), "fingerprints": fingerprints}))


if __name__ == "__main__":
    main()
