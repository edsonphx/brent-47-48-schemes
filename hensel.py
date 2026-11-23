"""Generate and replay exact first Hensel obstruction certificates."""
import argparse
import json

import numpy as np

from audit import ROOT, ModularEchelon
from reference.verify_certificate import jacobian_rows, require, target


def residual(terms):
    return (np.einsum("ri,rj,rk->ijk", *terms.transpose(1, 0, 2),
                      dtype=np.int64) - target(4)).ravel()


def obstruction(transposed_jacobian, rhs):
    echelon = ModularEchelon(transposed_jacobian.shape[0], 2)
    for equation, row in enumerate(transposed_jacobian.T):
        relation = echelon.insert(row, tag=equation)
        if relation is not None:
            witness = echelon.relation(relation, len(rhs))
            if int(witness @ rhs) % 2:
                return np.flatnonzero(witness).tolist()
    return None


def cokernel_projection(transposed_jacobian, rhs):
    echelon = ModularEchelon(len(rhs), 2)
    for column in transposed_jacobian:
        echelon.insert(column)
    residue = echelon.bits(rhs)
    for pivot, (column, _) in sorted(echelon.pivots.items()):
        if residue & pivot:
            residue ^= column
    support = [i for i in range(len(rhs)) if residue >> i & 1]
    return len(echelon.pivots), support


def check(name, terms, support):
    require(terms.shape == (47, 3, 16) and np.isin(terms, [0, 1]).all(),
            f"{name}: invalid binary coefficient array")
    discrepancy = residual(terms)
    require(not np.any(discrepancy % 2), f"{name}: invalid binary decomposition")
    jac = jacobian_rows(terms.astype(np.int32))
    rhs = (-discrepancy // 2) % 2
    if support is None:
        support = obstruction(jac, rhs)
    require(isinstance(support, list) and support == sorted(set(support))
            and all(type(i) is int and 0 <= i < len(rhs) for i in support),
            f"{name}: missing or malformed obstruction")
    require(not np.any(jac[:, support].sum(axis=1) % 2),
            f"{name}: witness does not annihilate J")
    require(int(rhs[support].sum()) % 2 == 1, f"{name}: witness does not separate rhs")
    rank, projection = cokernel_projection(jac, rhs)
    require(rank == 2117 and bool(projection), f"{name}: unexpected cokernel projection")
    return {"witness_support": support, "projection_support": projection, "rank": rank}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    entries = json.loads((ROOT / "inputs.json").read_text())
    path = ROOT / "hensel-witnesses.json"
    expected = None if args.generate else json.loads(path.read_text())
    witnesses = {}
    with np.load(ROOT / "inputs.npz", allow_pickle=False) as data:
        for entry in entries:
            if entry["fields"] != ["F2"]:
                continue
            name = entry["name"]
            cert = check(name, data[name], None if expected is None else expected[name]["witness_support"])
            witnesses[name] = cert
            print(json.dumps({"scheme": name, "mod4": "obstructed", "mod8": "obstructed",
                              "Z2": "obstructed", "cokernel_dimension": 4096 - cert["rank"],
                              "witness_weight": len(cert["witness_support"]),
                              "projection_weight": len(cert["projection_support"]),
                              "projection_support": cert["projection_support"]}))
    if expected is None:
        path.write_text(json.dumps(witnesses, indent=2) + "\n")
    else:
        require(expected == witnesses, "unexpected witness names")


if __name__ == "__main__":
    main()
