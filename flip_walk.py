"""Deterministic one-extra-summand F3 flip walk with exact Hensel tests."""
import json
from fractions import Fraction
from itertools import combinations
from math import gcd, isqrt, lcm
from random import Random

import numpy as np

from audit import ROOT, ModularEchelon
from scan import field_scan, tensor
from reference.verify_certificate import jacobian_rows, require, target


def canonical(point):
    point = point % 3
    point = point[np.all(np.any(point, axis=-1), axis=1)].copy()
    for mode in (0, 1):
        first = np.argmax(point[:, mode] != 0, axis=1)
        signs = point[np.arange(len(point)), mode, first]
        point[:, mode] *= signs[:, None]
        point[:, 2] *= signs[:, None]
    return point % 3


def key(point):
    return tuple(sorted(bytes(row) for row in point.astype(np.uint8).reshape(-1,48)))


def form(v):
    for x in v:
        if x:
            return bytes((v*x % 3).astype(np.uint8))
    return b""


def pairs(point):
    shared = {}
    for mode in range(3):
        buckets = {}
        for i, v in enumerate(point[:, mode]):
            buckets.setdefault(form(v), []).append(i)
        for b in buckets.values():
            if len(b)>1:
                for i,j in combinations(b,2):
                    shared.setdefault((i,j), []).append(mode)
    return shared


def split_seeds(point):
    seeds = []
    for mode in range(3):
        fs = point[:, mode]
        forms = {form(v) for v in fs}
        for i,j in combinations(range(len(point)),2):
            for sign in (1,2):
                overlap = form((fs[i] + sign*fs[j])%3)
                if overlap and overlap in forms:
                    seeds.append((i,j,mode,sign))
    return seeds


def other(mode):
    return (mode+1)%3, (mode+2)%3


def plus(point, i, j, mode, sign):
    copy = point.copy()
    b = point[i].copy()
    copy[i, mode] += sign*point[j,mode]
    b[mode] = -sign*point[j,mode]
    return canonical(np.concatenate((copy,b[None])))


def merge(point, i, j, modes):
    b = point[j].copy()
    dst = point[i].copy()
    scales = [int(point[i,m,np.argmax(point[i,m] != 0)]) *
              int(b[m,np.argmax(point[i,m] != 0)]) % 3 for m in modes]
    remaining = next(m for m in range(3) if m not in modes)
    dst[remaining] += scales[0]*scales[1]*b[remaining]
    out = point.copy()
    out[i] = dst
    return canonical(np.delete(out,j,axis=0))


def flip(point, i, j, mode, sign):
    u,v = other(mode)
    out = point.copy()
    k = int(np.argmax(point[i,mode] != 0))
    orientation = int(point[i,mode,k])*int(point[j,mode,k]) % 3
    out[j,mode] = point[i,mode]
    out[j,u] *= orientation
    b = out[j,u].copy()
    out[i,u] += sign*b
    out[j,v] -= sign*point[i,v]
    return canonical(out)


def lift_levels(point):
    echelon = ModularEchelon(4096,3)
    for i,row in enumerate(jacobian_rows(point.astype(np.int32))):
        echelon.insert(row, i)
    current = point.copy()
    previous = 3
    witnessed = []
    for level in (9,27):
        residual = (tensor(current) - target(4)).ravel()
        require(not np.any(residual%previous), "incorrect incoming lift")
        relation = echelon.insert(residual//previous, point.size)
        if relation is None:
            return witnessed, None, "no lift of tested representative to " + str(level)
        correction = echelon.relation(relation, point.size+1)[:-1].reshape(point.shape)
        current = (current + previous*correction)%level
        require(not np.any((tensor(current)-target(4))%level), "invalid Hensel correction")
        witnessed.append(level)
        previous = level
    return witnessed, current, None


def rationalize(point, modulus):
    bound = isqrt(modulus//2)
    tiny = {}
    for d in range(1,bound+1):
        if gcd(d,modulus)!=1:
            continue
        for n in range(-bound,bound+1):
            if gcd(abs(n),d)==1:
                tiny[(n*pow(d,-1,modulus))%modulus] = Fraction(n,d)
    if any(int(a) not in tiny for a in point.ravel()):
        return None
    result = np.array([tiny[int(a)] for a in point.ravel()],dtype=object).reshape(point.shape)
    ds = [lcm(*(a.denominator for a in result[:,i].ravel())) for i in range(3)]
    exact = np.stack([(result[:,i]*ds[i]).astype(np.int64) for i in range(3)],axis=1)
    return (exact,ds) if np.array_equal(tensor(exact),ds[0]*ds[1]*ds[2]*target(4)) else None


def walk(initial, seed, episodes, depth):
    rng = Random(seed)
    moves = split_seeds(initial)
    rng.shuffle(moves)
    found = {}
    visited = set()
    trials = 0
    for move in moves[:episodes]:
        point = plus(initial,*move)
        for _ in range(depth):
            sig = key(point)
            if sig not in visited:
                visited.add(sig)
            moves_here = pairs(point)
            for (i,j), ms in moves_here.items():
                if len(ms)<2:
                    continue
                q = merge(point,i,j,ms[:2])
                require(np.array_equal(tensor(q)%3,target(4)%3), "incorrect flip-walk merge")
                signature = key(q)
                if signature != key(initial):
                    found[signature] = q
            if not moves_here:
                break
            i,j = rng.choice(list(moves_here))
            point = flip(point,i,j,rng.choice(moves_here[i,j]),rng.choice((1,2)))
            trials += 1
            if len(point)!=49:
                if len(point)<=48 and key(point)!=key(initial):
                    found[key(point)] = point
                break
    return len(moves), len(visited), trials, found


def main():
    with np.load(ROOT/"scan-inputs.npz",allow_pickle=False) as data:
        meta = json.loads((ROOT/"scan-inputs.json").read_text())
        starts = ("rational","Agarwal-replacement")
        all_new = {}
        for i,name in enumerate(starts):
            q = data[name]%3
            q[:,2] *= pow(meta[name]["target_scale"],-1,3)
            q = canonical(q)
            require(np.array_equal(tensor(q)%3,target(4)%3), "invalid starting solution")
            levels,lift,failure = lift_levels(q)
            exact=rationalize(lift,27) if lift is not None else None
            available,visited,attempts,found = walk(q, 9101+i, 40, 12)
            print(json.dumps({"walk_seed": name, "available_plus_moves_with_overlap": available,
                              "distinct_states_visited": visited, "flip_steps": attempts,
                              "table_endpoints_other_than_start": len(found),
                              "canonical_control_hensel_levels": levels,
                              "canonical_control_small_rational_reconstruction": exact is not None,
                              "canonical_control_failure": failure}))
            all_new.update(found)
    for p in list(all_new.values())[:12]:
        require(np.array_equal(tensor(p)%3,target(4)%3), "returned candidate is invalid")
        levels, lift, note = lift_levels(p)
        exact = rationalize(lift,27) if lift is not None else None
        result = {"walk_candidate_length": len(p), "hensel_levels": levels,
                  "hensel_note": note, "exact_rational_hit_by_small_reconstruction": exact is not None}
        if len(p)==48:
            result.update(field_scan(p,1,"walk endpoint"))
        print(json.dumps(result))
        if exact is not None and len(p)<=47:
            raise RuntimeError("47-term exact candidate needs SMS export")


if __name__ == "__main__":
    main()
