"""Search a finite rational grid entirely on the verified curves."""
from fractions import Fraction
from collections import Counter
from math import lcm
import json

import numpy as np
import sympy as sp

from families import (FILE_NAMES, SUPPORTS, all_factors, common_coefficients,
                      exact_tensor, integer_evaluator, poly)
from audit import ROOT
from scan import exact_collision, field_scan, fingerprint
from reference.verify_certificate import require, target


def valuation3(n):
    if n == 0:
        raise ValueError("zero valuation")
    v = 0
    while n % 3 == 0:
        n //= 3
        v += 1
    return v


def good_reduction(point):
    out = np.empty(point.shape, dtype=np.int64)
    for i,term in enumerate(point):
        valuations = []
        for row in term:
            values = [valuation3(abs(a.numerator))-valuation3(a.denominator) for a in row if a]
            require(values, "curve grid produced a zero factor")
            valuations.append(min(values))
        if sum(valuations)<0:
            return None
        for k,(row,v) in enumerate(zip(term,valuations)):
            d = Fraction(3)**v
            for j,x in enumerate(row):
                u = x/d
                out[i,k,j] = (u.numerator*pow(u.denominator,-1,3)) % 3
        out[i,2] *= (sum(valuations) == 0)
    return out % 3


def canonical_reduction(point):
    return tuple(sorted(tuple(row) for row in point.reshape(48,48)))


def main():
    _,base = all_factors()
    tested = set()
    ranks=Counter()
    parameters = [sp.Rational(s,d) for s,d in (
        (-8,1),(-4,1),(-3,1),(-5,2),(-2,1),(-3,2),(-1,1),(-1,2),
        (0,1),(1,2),(1,1),(2,1),(4,1),(8,1))]
    for c,file in enumerate(FILE_NAMES):
        d=json.loads((ROOT/file).read_text())
        denominator=poly(d["denominator"])
        factors=np.array([poly(a) for a in d["numerators"]], dtype=object).reshape(12,3,16)
        numerators=common_coefficients(denominator,factors,base,np.array(SUPPORTS[c])-1)
        _,evaluate=integer_evaluator(denominator,numerators)
        best=None
        eligible=0
        reductions=0
        for h in parameters:
            d,n=evaluate(h)
            if not d:
                continue
            require(np.array_equal(exact_tensor(n), d**3*16**3*target(4).astype(object)),
                    "off-variety point in rational grid")
            score=exact_collision(n)
            candidate=(score[0],str(h),score[1:3],score[3])
            winner=best is None or candidate<best
            if winner:
                best=candidate
                ds=[lcm(*(int(sp.denom(x)) for x in n[:,m].flat)) for m in range(3)]
                integer=np.stack([np.vectorize(int,otypes=[object])(n[:,m]*ds[m])
                                  for m in range(3)],axis=1)
                best_fp=fingerprint(integer,int(d**3*16**3*ds[0]*ds[1]*ds[2]))
            eligible+=1
            scale=Fraction(int(sp.numer(d*16)),int(sp.denom(d*16)))
            normalized=np.array([Fraction(int(sp.numer(x)),int(sp.denom(x)))/scale for x in n.flat],
                                dtype=object).reshape(n.shape)
            q=good_reduction(normalized)
            if q is None:
                continue
            signature=canonical_reduction(q)
            if signature not in tested:
                result=field_scan(q,1,f"chart {c+1} h={h}")
                ranks[result["jacobian_rank_F3"]]+=1
                if result.get("obstruction_audit"):
                    print(json.dumps(result),flush=True)
                tested.add(signature)
                reductions+=1
        print(json.dumps({"chart":c+1,"exact_rational_samples":eligible,
                          "new_reductions_audited":reductions,
                          "least_score":str(best[0]),"parameter":best[1],
                          "pair_1_based":best[2],"mode_pair_1_based":best[3],
                          "rational_fingerprint_at_minimum":best_fp}),flush=True)
    print(json.dumps({"unique_grid_reductions_by_displayed_coefficients":len(tested),
                      "grid_jacobian_ranks_F3":dict(sorted(ranks.items()))}))


if __name__=="__main__":
    main()
