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
        check(data, "gamma-equivalence.json", 2)
        check(data, "dps48-equivalence.json", None)
        points = ["AT47", "KM47", "C3", "C4", "C5"]
        fingerprints = {k: binary_fingerprint(data[k]) for k in points}
        require(len(set(map(tuple, fingerprints.values()))) == len(points),
                "rank triple invariant did not separate the binary reps")
        print(json.dumps({"distinct_binary_orbits": len(points), "fingerprints": fingerprints}))


if __name__ == "__main__":
    main()
