"""Optional numerical collision search in a fixed DPS48 gauge (requires SciPy)."""
import argparse
import json
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr

from audit import ROOT, ModularEchelon
from reference.verify_certificate import symmetry_directions, target, require

PAIR_I, PAIR_J = np.triu_indices(48, 1)
EXPECTED = target(4).ravel()
COORDS = np.arange(4096).reshape(16, 16, 16)


def residual(point):
    return np.einsum("ri,rj,rk->ijk", *point.transpose(1, 0, 2), optimize=True).ravel() - EXPECTED


def free_coordinates(point):
    span = ModularEchelon(point.size, 3)
    for row in symmetry_directions(point, scales=True):
        span.insert(row.ravel())
    free = np.ones(point.size, dtype=bool)
    free[[i.bit_length() - 1 for i in span.pivots]] = False
    return np.flatnonzero(free)


def jacobian(point, free):
    a, b, c = point.transpose(1, 0, 2)
    values = np.array([np.einsum("ri,rj->rij", *m) for m in ((b, c), (a, c), (a, b))])
    selected = np.arange(point.size).reshape(point.shape)
    rows = np.concatenate([np.broadcast_to(np.moveaxis(COORDS, mode, 0),
                           (48, 16, 16, 16)).ravel() for mode in range(3)])
    cols = np.concatenate([np.repeat(selected[:, mode].ravel(), 256) for mode in range(3)])
    data = np.repeat(values[:, :, None, :, :], 16, axis=2).ravel()
    jac = csr_matrix((data, (rows, cols)), shape=(4096, point.size))
    jac.eliminate_zeros()
    return jac[:, free]


def objective(point):
    u, v = point[PAIR_I], point[PAIR_J]
    dot = np.sum(u*v, axis=-1)
    lengths = np.sum(u*u, axis=-1)*np.sum(v*v, axis=-1)
    if np.any(lengths < 1e-20):
        return None
    wedge = 1 - dot*dot/lengths
    both = np.argsort(wedge, axis=-1, kind="stable")[:, :2]
    scores = np.take_along_axis(wedge, both, axis=-1).sum(axis=-1)
    j = np.argmin(scores)
    return float(scores[j]), int(PAIR_I[j]), int(PAIR_J[j]), tuple(int(x) for x in both[j])


def gradient(point, a, b, modes):
    g = np.zeros_like(point)
    for k in modes:
        u, v = point[a, k], point[b, k]
        dot, nu, nv = u@v, u@u, v@v
        g[a, k] = -2*dot/(nu*nv)*(v - dot/nu*u)
        g[b, k] = -2*dot/(nu*nv)*(u - dot/nv*v)
    return g.ravel()


def least_squares(j, rhs):
    return lsmr(j, rhs, atol=1e-11, btol=1e-11, maxiter=240)[0]


def search(integer, denominators, steps):
    point = integer / np.asarray(denominators)[None, :, None]
    free = free_coordinates(integer % 3)
    require(len(free) == 2163, "expected a full-rank gauge")
    initial = objective(point)
    jac = jacobian(point, free)
    # Check that the independent Jacobian generator tracks a finite difference.
    check = free[len(free)//2]
    shifted = point.copy()
    shifted.ravel()[check] += 1
    require(np.max(np.abs(residual(shifted) - residual(point) -
                          jac[:, len(free)//2].toarray().ravel())) < 1e-10,
            "numerical Jacobian construction disagrees with the tensor")
    accepted = 0
    residual_max = 0.0
    for _ in range(steps):
        value, a, b, modes = objective(point)
        g = gradient(point, a, b, modes)[free]
        tangent = g - jac.T @ least_squares(jac.T, g)
        if np.linalg.norm(tangent) < 1e-10:
            break
        delta = 0.08 * tangent/np.linalg.norm(tangent)
        for _ in range(4):
            trial = point.copy()
            trial.ravel()[free] -= delta
            for _ in range(4):
                r = residual(trial)
                if np.max(np.abs(r)) <= 2e-10:
                    break
                trial.ravel()[free] -= least_squares(jacobian(trial, free), r)
            err = float(np.max(np.abs(residual(trial))))
            score = objective(trial)
            if err <= 2e-10 and score is not None and score[0] < value - 1e-10:
                point = trial
                jac = jacobian(point, free)
                residual_max = max(residual_max, err)
                accepted += 1
                break
            delta /= 2
        else:
            break
    return {"starting_score": initial[0], "score": objective(point)[0],
            "lowest_pair_0_based": objective(point)[1:],
            "accepted_steps": accepted, "max_accepted_residual": residual_max}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--schemes", nargs="+", default=[
        "rational", "accurate", "sparse", "Moran-Schwartz-Yuan", "Agarwal-replacement"])
    args = parser.parse_args()
    meta = json.loads((ROOT/"scan-inputs.json").read_text())
    with np.load(ROOT/"scan-inputs.npz", allow_pickle=False) as data:
        for name in args.schemes:
            if name=="LWH(t=1)":
                point=np.load(ROOT/"lwh-laurent.npy",allow_pickle=False).sum(axis=0)
                scales=[8,8,8]
            else:
                point=data[name]
                scales=meta[name]["factor_denominators"]
            print(json.dumps({"scheme": name, **search(point,scales,args.steps)}), flush=True)


if __name__ == "__main__":
    main()
