"""Replay exact finite-field ranks and rational invariant fingerprints."""
from collections import Counter
from fractions import Fraction
from itertools import combinations
from math import gcd
import json

import numpy as np

from audit import ROOT, ModularEchelon, audit, Input
from equivalence import modes
from reference.verify_certificate import jacobian_rows, require, symmetry_directions, target


def tensor(terms):
    return np.einsum("ri,rj,rk->ijk", *terms.transpose(1, 0, 2), dtype=np.int64)


def exact_rank(matrix):
    rows = [[int(x) for x in row] for row in matrix]
    h = 0
    for j in range(len(rows[0])):
        i = next((i for i in range(h, len(rows)) if rows[i][j]), None)
        if i is None:
            continue
        rows[h], rows[i] = rows[i], rows[h]
        a = rows[h][j]
        for i in range(h+1, len(rows)):
            b = rows[i][j]
            if not b:
                continue
            g = gcd(a,b)
            rows[i] = [(a//g)*x - (b//g)*y for x,y in zip(rows[i], rows[h])]
        h += 1
    return h


def loop_profile(matrix, scale):
    m = matrix.astype(object)
    power = m
    traces = []
    for _ in range(4):
        traces.append(int(np.trace(power)))
        power = power @ m
    multiplicity, remainder = divmod(traces[0], scale)
    # Power sums determine the characteristic polynomial by Newton's identities.
    require(not remainder and 0 <= multiplicity <= 4
            and traces == [multiplicity*scale**k for k in range(1, 5)],
            "unexpected characteristic polynomial")
    def ranks(a, complement):
        power = a
        parts = []
        for _ in range(4):
            parts.append(exact_rank(power) - complement)
            power = power @ a
        return tuple(parts)
    return (multiplicity, ranks(m, multiplicity),
            ranks(m - scale*np.eye(4, dtype=object), 4-multiplicity))


def fingerprint(point, scale):
    matrices = modes(point)
    first, second, third = matrices.swapaxes(0, 1).astype(object)
    loops = np.stack((first@second@third, second@third@first, third@first@second), axis=1)
    values = [(tuple(sorted(exact_rank(m) for m in part)),
               tuple(sorted(loop_profile(m, scale) for m in loop)))
              for part, loop in zip(matrices, loops)]
    return [[k, v] for k,v in sorted(Counter(values).items())]


def pair_data(point):
    if max(abs(int(x)) for x in point.flat) >= 2048:
        point = point.astype(object)
    first, second = np.triu_indices(len(point), 1)
    a, b = point[first], point[second]
    n = (a*a).sum(axis=-1) * (b*b).sum(axis=-1)
    d = (a*b).sum(axis=-1)**2
    return first, second, n, d


def exact_collision(point):
    i, j, n, d = pair_data(point)
    require(np.all(n>0), "zero summand factor")
    def rational(x):
        if isinstance(x, (int, np.integer)):
            return Fraction(int(x))
        return Fraction(int(x.numerator), int(x.denominator))
    best = None
    for k in range(len(i)):
        losses = sorted((1 - rational(d0)/rational(n0), m)
                        for m,(n0,d0) in enumerate(zip(n[k],d[k])))
        value = losses[0][0] + losses[1][0]
        if best is None or value < best[0]:
            best = (value, int(i[k])+1, int(j[k])+1, [losses[0][1]+1,losses[1][1]+1])
    return best


def field_scan(point, scale, name):
    q = (point % 3).astype(np.int32)
    require(scale % 3 and np.array_equal(tensor(q)%3, target(4)*scale%3), "invalid F3 reduction")
    rank = ModularEchelon(4096, 3)
    for row in jacobian_rows(q):
        rank.insert(row)
    orbit = ModularEchelon(q.size, 3)
    for row in symmetry_directions(q, True):
        orbit.insert(row.ravel())
    rank, orbit = len(rank.pivots), len(orbit.pivots)
    result = {"scheme": name, "jacobian_rank_F3": rank,
              "tangent_dimension_F3": 2304-rank, "orbit_rank_F3": orbit}
    if rank < 2153:
        result["obstruction_audit"] = audit(Input(name, q, scale % 3), "F3")
    return result


def shared_factors_mod3(point):
    counts = []
    for f in point.swapaxes(0, 1):
        signatures = Counter()
        for row in f%3:
            nz = np.flatnonzero(row)
            require(len(nz)>0, "factor zero over F3")
            signatures[tuple(row * row[nz[0]] % 3)] += 1
        counts.append(sum(k*(k-1)//2 for k in signatures.values()))
    return counts


def main():
    source = json.loads((ROOT/"scan-inputs.json").read_text())
    with np.load(ROOT/"scan-inputs.npz", allow_pickle=False) as data:
        base = np.load(ROOT/"lwh-laurent.npy", allow_pickle=False)
        tables = {name:(data[name], source[name]["target_scale"]) for name in source}
        tables["LWH(t=1)"] = (base.sum(axis=0), 8**3)
        for name, (point, scale) in tables.items():
            require(point.shape == (48,3,16) and int(np.abs(point).max()) < 2048,
                    "unexpected scheme shape or coefficient bound")
            require(np.array_equal(tensor(point), target(4)*scale), f"{name}: invalid Q identity")
            result = field_scan(point, scale, name)
            result["ordinary_flip_pair_counts_F3_by_mode"] = shared_factors_mod3(point)
            if result["orbit_rank_F3"] == 141:
                # There are 48 matrix-group and 96 scaling parameters and
                # at least three scalar-kernel dimensions. Reduction proves
                # the lower bound 141 in characteristic zero.
                result["stabilizer_dimension_Q_full_group"] = 3
                result["stabilizer_dimension_Q_effective_group"] = 0
            best, i, j, selected = exact_collision(point)
            result["least_sample_score"] = str(best)
            result["least_pair_1_based"] = [i,j]
            result["least_modes_1_based"] = selected
            result["rational_fingerprint"] = fingerprint(point, scale)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
