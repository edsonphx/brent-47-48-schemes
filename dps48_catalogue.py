"""Exact equivalence classes and chart membership of the 21 rational scan tables."""
from collections import Counter
from fractions import Fraction
from itertools import combinations
from math import gcd, prod
import json

import numpy as np

from boundary import determinant, loops
from equivalence import modes
from families import (ROOT, FILE_NAMES, SUPPORTS, all_factors, common_coefficients,
                      integer_evaluator, poly, sp)
from reference.verify_certificate import require


def terms(point, scale):
    result = Counter()
    for term in point:
        primitive = []
        weight = Fraction(1, scale)
        for factor in term:
            row = list(map(int, factor))
            divisor = gcd(*row)
            require(divisor != 0, "zero factor")
            if next(x for x in row if x) < 0:
                divisor = -divisor
            primitive.append(tuple(x//divisor for x in row))
            weight *= divisor
        result[tuple(primitive), weight] += 1
    return result


def trace_maximum(point, scale):
    values = []
    for loop in loops(point.astype(object).reshape(48, 3, 4, 4)):
        traces = loop.reshape(48, 16) @ loop.swapaxes(1, 2).reshape(48, 16).T
        values.append(max(abs(int(traces[i, j])) for i, j in combinations(range(48), 2)))
    return Fraction(max(values), scale**2)


def check_map(source, source_scale, dest, dest_scale, cert):
    order, perm = cert["order"], cert["term_permutation"]
    require(order in ([0, 1, 2], [1, 2, 0], [2, 0, 1]), "unexpected mode rotation")
    require(sorted(perm) == list(range(48)), "not a term permutation")
    a = modes(source)[:, order].astype(object)
    b = modes(dest)[perm].astype(object)
    g = np.array(cert["vertex_matrices"], dtype=object)
    require(g.shape == (3, 4, 4) and all(determinant(m) for m in g), "singular basis change")
    scalars = [[Fraction(v) for v in row] for row in cert["scalars"]]
    require(len(scalars) == 48, "invalid scaling table")
    for i, layer in enumerate(scalars):
        require(len(layer) == 3 and prod(layer) == Fraction(dest_scale, source_scale),
                "incorrect raw-data term scale")
        for m, value in enumerate(layer):
            require(np.array_equal(value.numerator*g[m]@a[i,m],
                                   value.denominator*b[i,m]@g[(m+1)%3]),
                    "failed rational factor equality")


def main():
    meta = json.loads((ROOT/"scan-inputs.json").read_text())
    maps = json.loads((ROOT/"dps48-catalogue-maps.json").read_text())
    with np.load(ROOT/"scan-inputs.npz", allow_pickle=False) as data:
        tables = {name: (data[name], entry["target_scale"]) for name, entry in meta.items()}
    raw, base = all_factors()
    tables["LWH(t=1)"] = (raw.sum(axis=0), 8**3)
    require(len(tables) == 21, "unexpected catalogue")
    common = set(tables) - {"LWH(t=1)", "Agarwal-D2", "Agarwal-replacement"}
    require(len(common) == 18 and set(maps) == common, "incomplete orbit certificate")
    with np.load(ROOT/"inputs.npz", allow_pickle=False) as inputs:
        dps = inputs["DPS48"]
    for name in sorted(common):
        check_map(dps, 16, *tables[name], maps[name])
    maxima = {name: trace_maximum(*table) for name, table in tables.items()}
    require(all(maxima[n] == 2 for n in common), "unexpected DPS48 trace maximum")
    require([maxima[n] for n in ("LWH(t=1)", "Agarwal-D2", "Agarwal-replacement")]
            == [4, 8, 17], "orbit separators failed")
    require(terms(base, 16**3) == terms(*tables["Agarwal-D2"]), "wrong D(2) identification")
    chart = json.loads((ROOT/FILE_NAMES[0]).read_text())
    den = poly(chart["denominator"])
    part = np.array([poly(c) for c in chart["numerators"]], dtype=object).reshape(12,3,16)
    numerator = common_coefficients(den, part, base, np.array(SUPPORTS[0])-1)
    degree, evaluate = integer_evaluator(den, numerator)
    # On the unordered chart the involution permutes whole summand tensors.
    # Cross-multiplication gives coordinate polynomials of degree at most bound.
    bound = 3 * (degree + int(den.degree()))
    permutation = list(range(48))
    for a, b in ((12, 32), (19, 22)):
        permutation[a-1], permutation[b-1] = b-1, a-1
    for h in range(1, bound + 2):
        d, first = evaluate(h)
        e, second = evaluate(-2-h)
        require(d != 0 and e != 0, "involution interpolation node is a pole")
        for i, j in enumerate(permutation):
            require(terms(first[i:i+1], (16*d)**3) == terms(second[j:j+1], (16*e)**3),
                    "chart involution failed on a summand")
    matched = []
    for h in (sp.Rational(-1, 2), sp.Rational(-3, 2)):
        divisor, values = evaluate(h)
        require(divisor != 0, "pole at proposed chart parameter")
        if terms(values, (16*divisor)**3) == terms(*tables["Agarwal-replacement"]):
            matched.append(str(h))
    require(len(matched) == 2, "no termwise match with the Agarwal replacement")
    print(json.dumps({"DPS48_catalogue_orbit_sizes": [18, 1, 1, 1],
                      "max_absolute_pair_loop_traces": [2, 4, 8, 17],
                      "first_chart_involution": {"h": "-2-h", "swaps_1_based": [[12,32], [19,22]],
                                                 "interpolation_bound": bound},
                      "Agarwal_replacement_first_chart_at_t2": matched}))


if __name__ == "__main__":
    main()
