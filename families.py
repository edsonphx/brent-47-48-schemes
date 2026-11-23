"""Exact coefficient, boundary and transport checks for four rational DPS48 curves."""
from itertools import combinations
from math import lcm
import json

import numpy as np
import sympy as sp

from audit import ROOT, ModularEchelon
from boundary import determinant, evaluate_polynomial, loops
from equivalence import modes
from scan import exact_collision, field_scan, fingerprint
from reference.verify_certificate import require, symmetry_directions, target

H, T = sp.symbols("h t")
GROUPS = [[4,17,34,40], [9,25,28,35], [12,19,22,32], [15,38,44,48]]
SUPPORTS = [sum(x, []) for x in combinations(GROUPS, 3)]
FILE_NAMES = ["rational12.json", "rational-chart2.json", "rational-chart3.json",
              "rational-chart4.json"]
WEIGHTED = set(GROUPS[3])
# Agarwal's mode order, followed by group numbers for fixed, bottom and arm.
CHARTS = [((0,1,2),(1,2,0)), ((0,2,1),(1,3,0)),
          ((0,2,1),(2,0,3)), ((0,1,2),(2,1,3))]


def poly(values):
    return sp.Poly.from_list([sp.Rational(v) for v in reversed(values)], H, domain=sp.QQ)


def all_factors():
    p = np.load(ROOT/"lwh-laurent.npy", allow_pickle=False)
    return p, p[0]+2*p[1]+4*p[2]


def exact_tensor(point):
    u,v,w = point.swapaxes(0,1)
    return (u[:,:,None,None]*v[:,None,:,None]*w[:,None,None,:]).sum(axis=0)


def common_coefficients(den, part, base, indices):
    num = np.empty(base.shape, dtype=object)
    for i in np.ndindex(base.shape):
        num[i] = int(base[i])*den
    num[indices] = part
    return num


def integer_evaluator(den, num):
    polynomials=list(dict.fromkeys([den,*num.flat]))
    degree=max(int(f.degree()) for f in polynomials if not f.is_zero)
    scale=lcm(*(int(sp.denom(c)) for f in polynomials for c in f.all_coeffs()))
    arrays={f:[int(scale*f.nth(i)) for i in range(degree+1)] for f in polynomials}

    def evaluate(parameter):
        a,b = int(sp.numer(parameter)),int(sp.denom(parameter))
        weights=[a**i*b**(degree-i) for i in range(degree+1)]
        values={f:sum(x*y for x,y in zip(coeff,weights)) for f,coeff in arrays.items()}
        return values[den], np.array([values[f] for f in num.flat],
                                     dtype=object).reshape(num.shape)
    return degree,evaluate


def check_identity(den, num, base):
    degree,evaluate=integer_evaluator(den,num)
    bound=3*degree
    for x in range(bound+1):
        d,n=evaluate(x)
        require(np.array_equal(exact_tensor(n), d**3*16**3*target(4).astype(object)),
                f"Brent polynomial identity fails at interpolation node {x}")
    d,n=evaluate(0)
    require(np.array_equal(n, base.astype(object)*d) and d!=0,
            "curve is unanchored")
    return bound


def gcds(values):
    g = sp.Poly(0,H,domain=sp.QQ)
    for a in values:
        g = g.gcd(a)
        if g.degree() == 0:
            return g
    return g


def outside(g, den):
    require(not g.is_zero, "collinearity or zero condition holds identically")
    while (d := g.gcd(den)).degree() > 0:
        g = g.exquo(d)
    return g


def wedge(a,b):
    return (a[i]*b[j]-a[j]*b[i] for i,j in combinations(range(16),2))


def regular_boundary(den, num):
    primitive = num.copy().reshape(-1,16)
    for factor in primitive:
        g = gcds(factor)
        require(outside(g, den).degree() == 0, "zero factor at regular parameter")
        if g.degree():
            factor[:] = [x.exquo(g) for x in factor]
    for a,b in combinations(primitive.reshape(num.shape),2):
        minors = [gcds(wedge(a[i],b[i])) for i in range(3)]
        for x,y in combinations(minors,2):
            require(outside(x.gcd(y),den).degree() == 0, "collision at regular parameter")


def valuation(p, q):
    if p.is_zero:
        return None, p
    if q is None:
        return -int(p.degree()), p.LC()
    k = 0
    while p.rem(q).is_zero:
        k += 1
        p = p.exquo(q)
    return k,p.rem(q)


def limit_data(den,num,q):
    den_order,dc = valuation(den,q)
    limits = np.empty(num.shape,dtype=object)
    orders = []
    for i,term in enumerate(num):
        total = 0
        for m,part in enumerate(term):
            v = [valuation(x,q) for x in part]
            lowest = min(k for k,_ in v if k is not None)
            total += lowest-den_order
            limits[i,m] = [r if k==lowest else 0 for k,r in v]
        orders.append(total)
    return orders,limits,dc


def is_zero(value, q):
    return value == 0 if q is None else not value


def normalize_limit(limits,dc,q):
    # For a rational endpoint evaluate constants in the residue field.
    scalar = dc if q is None else dc.nth(0)
    raw = np.array([(x if q is None or x == 0 else x.nth(0))/scalar
                    for x in limits.flat],dtype=object).reshape(limits.shape)
    d = [lcm(*(int(sp.denom(x)) for x in raw[:,i].flat)) for i in range(3)]
    point = np.stack([np.vectorize(int,otypes=[object])(raw[:,i]*d[i])
                      for i in range(3)],axis=1)
    scale = 16**3 * d[0]*d[1]*d[2]
    require(np.array_equal(exact_tensor(point.astype(object)),scale*target(4).astype(object)),
            "finite specialization fails exact tensor check")
    return point,scale


def excluded_parameters(den,num):
    result = []
    for q in [p for p,_ in den.factor_list()[1]]+[None]:
        orders,limits,dc = limit_data(den,num,q)
        name = str(q.as_expr()) if q is not None else "infinity"
        row = {"endpoint":name,"negative_term_orders":sum(k<0 for k in orders)}
        if min(orders) >= 0:
            require(max(orders) == 0, "finite endpoint has zero summand")
            for a,b in combinations(limits,2):
                uncollinear = 0
                for i in range(3):
                    minors = ((v if q is None or v == 0 else v.rem(q))
                              for v in wedge(a[i],b[i]))
                    uncollinear += any(not is_zero(v,q) for v in minors)
                require(uncollinear >= 2, "finite endpoint has a two-factor collision")
            if q is None or q.degree()==1:
                point,scale = normalize_limit(limits,dc,q)
                row["exact_least_pair_score"] = str(exact_collision(point)[0])
                if scale % 3:
                    row.update(field_scan(point,scale,f"boundary {name}"))
                    row["rational_fingerprint"] = fingerprint(point,scale)
        result.append(row)
    return result


def rational_frame(factors, cols, rows=None):
    E = sp.Matrix.hstack(*[factors[i-1] for i in cols])
    select = list(E.T.rref()[1]) if rows is None else [i-1 for i in rows]
    require(len(select) == len(cols), "incomplete frame")
    complement = [i for i in range(16) if i not in select]
    return E.row_join(sp.eye(16)[:,complement])


def transport_frames(raw):
    coeff = np.stack([modes(p).reshape(48,3,16) for p in raw],axis=0)
    f = [[sp.Matrix((2*(coeff[0,i,m]/T + coeff[1,i,m] + coeff[2,i,m]*T)).tolist())
          for i in range(48)] for m in range(3)]
    constant = [[a.subs(T,2) for a in m] for m in f]
    weights = T/2
    Bset,Brows = [4,9,12,15,17,22,25,28], [1,2,3,5,6,8,9,10]
    Cset,Crows = [4,9,12,15,17,19,22,34], [1,2,3,5,6,7,9,11]
    B = rational_frame(f[1],Bset,Brows)
    C = rational_frame(f[2],Cset,Crows)
    for frame, rows, coefficient, exponent in (
            (B,Brows,-1024*16**8,4), (C,Crows,sp.Rational(16**8,262144),-3)):
        padded = (frame[[i-1 for i in rows],:8]*T).applyfunc(sp.expand)
        for i in range(17):
            require(padded.subs(T,i).det() == coefficient*i**(8+exponent),
                    "published core minor has changed")
    maps = [sp.eye(16), B*B.subs(T,2).inv(),
            C*sp.diag(*([1/weights if x in WEIGHTED else 1 for x in Cset]+[1]*8))
             * C.subs(T,2).inv()]
    for i in sum(GROUPS,[]):
        for m in range(3):
            weight = weights if m==2 and i in WEIGHTED else 1
            require(all(sp.cancel(x)==0 for x in maps[m]*constant[m][i-1]*weight-f[m][i-1]),
                    "wrong core transport formula")
    return f,constant,maps


def chart_maps(chart,const):
    mode_order,(fi,bo,ar) = CHARTS[chart]
    fwt,bwt,awt = [T/2 if j==3 else sp.S.One for j in (fi,bo,ar)]
    Hs = [sp.eye(16) for _ in range(3)]
    second,third = mode_order[1:]
    B = rational_frame(const[second],GROUPS[fi]+GROUPS[bo])
    C = rational_frame(const[third],GROUPS[bo]+GROUPS[ar])
    Hs[second] = B*sp.diag(*([fwt]*4+[bwt]*4+[1]*8))*B.inv()
    Hs[third] = C*sp.diag(*([1]*4+[awt/bwt]*4+[1]*8))*C.inv()
    return Hs


def chart_transport(chart, sample, f, const, core_maps):
    Hs=chart_maps(chart,const)
    indices = SUPPORTS[chart]
    out = [(a*b).subs(T,sample) for a,b in zip(core_maps,Hs)]
    for i in indices:
        ratios = []
        for m in range(3):
            lhs = out[m]*const[m][i-1]
            rhs = f[m][i-1].subs(T,sample)
            k = next(j for j in range(16) if rhs[j])
            factor = lhs[k]/rhs[k]
            require(factor!=0 and lhs==factor*rhs,
                    f"invalid rescaling after chart transport: chart {chart}, term {i}, mode {m}")
            ratios.append(factor)
        require(sp.prod(ratios)==1,"chart transport does not preserve summand tensor")
    # All three factors are proportional termwise symbolically, with product one.
    for i in indices:
        weights = []
        for m in range(3):
            lhs = core_maps[m]*Hs[m]*const[m][i-1]
            rhs = f[m][i-1]
            k = next(j for j in range(16) if rhs[j]!=0)
            w = sp.cancel(lhs[k]/rhs[k])
            require(w!=0 and all(sp.cancel(a-w*b)==0 for a,b in zip(lhs,rhs)),
                    "symbolic chart transport failed")
            weights.append(w)
        require(sp.cancel(sp.prod(weights)-1)==0,"symbolic term weight product failed")
    return out


def unit_laurent(value):
    n,d=sp.fraction(sp.cancel(value))
    return n!=0 and len(sp.Poly(n,T).terms())==len(sp.Poly(d,T).terms())==1


def verify_cross_boundary(chart,part,f,const,core_maps):
    core=set(sum(GROUPS,[]))
    active=SUPPORTS[chart]
    Hs=chart_maps(chart,const)
    outsiders=set(range(1,49))-core
    separated={i:0 for i in outsiders}
    cycle=modes(part).reshape(12,3,16)
    for m in range(3):
        span=sp.Matrix.hstack(*[const[m][j-1] for j in active])
        tests=span.T.nullspace()
        require(len(tests)==8,"unexpected active factor span")
        for r in tests:
            for a in cycle[:,m]:
                require(sum((r[k]*x for k,x in enumerate(a)),sp.Poly(0,H)).is_zero,
                        "moving factor left the transported support")
        for i in core-set(active):
            orig=const[m][i-1]
            moved=Hs[m]*orig
            k=next(j for j in range(16) if orig[j])
            require(unit_laurent(moved[k]/orig[k]) and
                    all(sp.cancel(a*orig[k]-b*moved[k])==0 for a,b in zip(moved,orig)),
                    "fixed core term is not transported projectively")
        inverse=(core_maps[m]*Hs[m]).inv()
        for i in outsiders:
            outside=inverse*f[m][i-1]
            separated[i]+=any(unit_laurent(r.dot(outside)) for r in tests)
    require(min(separated.values())>=2, "fixed and moving factor spans may meet in two modes")


def verify_fixed_loops(raw):
    block=raw.reshape(3,48,3,4,4)
    signs=[(1,-1,-1),(-1,-1,-1),(-1,-1,-1),(-1,-1,1)]
    high=[(157,53),(159,54),(157,53),(157,53)]
    charts=[]
    base=loops(evaluate_polynomial(block,1).astype(object))
    for c,support in enumerate(SUPPORTS):
        fixed=[i for i in range(48) if i+1 not in support]
        chosen=[]
        for cycle in base:
            span=ModularEchelon(16,3)
            chosen.append([i for i in fixed if span.insert(cycle[i].ravel().astype(np.int64)) is None])
        require(all(len(idx)==15 for idx in chosen) and all(i in fixed for i in (0,1,2,5,26)),
                "unexpected constant term basis")
        for t in range(103):
            group=loops(evaluate_polynomial(block,t).astype(object))
            for m,loop in enumerate(group):
                flattened=[loop[i].ravel() for i in chosen[m]]
                flattened.append((loop[0]@loop[1]).ravel())
                power,exponent=high[c] if m==2 else (152,51)
                require(determinant(flattened)==signs[c][m]*2**power*t**exponent,
                        "fixed loop lattice certificate failed")
            if t<13:
                # All loops here are multiplied by (8t)^3.
                require(np.trace(group[2][0]@group[2][26])==-65536*t**5
                        and np.trace(group[2][2]@group[2][5])==1048576*t**7,
                        "fixed trace pole certificate failed")
        charts.append([[i+1 for i in mode] for mode in chosen])
    return charts


def mod3(a):
    v = [int(sp.numer(x))*pow(int(sp.denom(x)),-1,3)%3 for x in a.flat]
    return np.array(v,dtype=np.int64).reshape(a.shape)


def independent_tangents(raw, definitions):
    f,const,core_maps = transport_frames(raw)
    for chart,(_,part) in enumerate(definitions):
        verify_cross_boundary(chart,part,f,const,core_maps)
    results = {}
    for t in (sp.Rational(1,4), sp.Rational(2)):
        same = np.array([[list(f[m][i].subs(T,t)) for m in range(3)]
                         for i in range(48)],dtype=object)
        D = np.array([[list(f[m][i].diff(T).subs(T,t)) for m in range(3)]
                        for i in range(48)],dtype=object)
        span = ModularEchelon(2304,3)
        p = mod3(modes(same).reshape(48,3,16))
        for v in symmetry_directions(p,True):
            span.insert(v.ravel())
        require(len(span.pivots)==141, "rank loss in transported symmetry directions")
        span.insert(mod3(modes(D)).ravel())
        for chart,(den,num) in enumerate(definitions):
            active = np.array(SUPPORTS[chart])-1
            trans = chart_transport(chart,t,f,const,core_maps)
            dh = np.array([sp.cancel((n.diff().eval(0)*den.eval(0)-n.eval(0)*den.diff().eval(0))
                                    /den.eval(0)**2) for n in num.flat],
                          dtype=object).reshape(num.shape)
            dh = modes(dh).reshape(12,3,16)
            variation = np.zeros_like(same)
            for k,i in enumerate(active):
                for m in range(3):
                    rhs = sp.Matrix(same[i,m])
                    row = next(j for j in range(16) if rhs[j])
                    ratio = (trans[m]*const[m][i])[row]/rhs[row]
                    variation[i,m] = list(trans[m]*sp.Matrix(dh[k,m])/ratio)
            span.insert(mod3(modes(variation)).ravel())
        results[str(t)] = len(span.pivots)-141
    return results


def main():
    raw,base = all_factors()
    definitions = []
    for chart,filename in enumerate(FILE_NAMES):
        d = json.loads((ROOT/filename).read_text())
        require(d["support_1based"]==SUPPORTS[chart] and d["factor_scale"]==16,
                "unexpected chart metadata")
        den = poly(d["denominator"])
        part = np.array([poly(a) for a in d["numerators"]],dtype=object).reshape(12,3,16)
        if chart==0:
            numerators=part.ravel()
            y=numerators[32].as_expr()/den.as_expr()
            t=numerators[447].as_expr()/den.as_expr()
            require(sp.cancel(y-8*((1+H)**2-1))==0 and
                    sp.cancel(t+16*H/((H+1)*(H+2)))==0 and
                    sp.cancel(t*t*y*y+8*t*t*y-512*t*y-4096*t-2048*y)==0,
                    "first rationalization certificate failed")
        num = common_coefficients(den,part,base,np.array(SUPPORTS[chart])-1)
        degree = check_identity(den,num,base)
        regular_boundary(den,num)
        result = {"rational_curve_chart":chart+1,"fixed_summands":36,"identity_degree_bound":degree,
                  "regular_pairs_by_mode_pair":1128,"finite_divisor_cases":excluded_parameters(den,num)}
        print(json.dumps(result),flush=True)
        definitions.append((den,part))
    rank = independent_tangents(raw,definitions)
    require(set(rank.values())=={5},f"unexpected transverse rank {rank}")
    print(json.dumps({"exact_two_parameter_families":4,"joint_transverse_rank_at":rank,
                      "fixed_loop_bases":verify_fixed_loops(raw),
                      "fixed_trace_poles":["-1/(4t)","4t"],
                      "zero_and_two_factor_boundary":"excluded over Qbar for all four surfaces"}))


if __name__=="__main__":
    main()
