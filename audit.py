"""Exact tangent/orbit and quadratic obstruction audit in F2, F3, and Q."""
import argparse
from dataclasses import dataclass
from functools import reduce
from math import gcd
from pathlib import Path
import json

import numpy as np

from reference.verify_certificate import jacobian_rows, require, symmetry_directions, target


ROOT = Path(__file__).resolve().parent


class ModularEchelon:
    def __init__(self, width, prime):
        self.width = width
        self.mask = (1 << width) - 1
        self.prime = prime
        self.pivots = {}

    @staticmethod
    def bits(array):
        return int.from_bytes(np.packbits(array, bitorder="little").tobytes(), "little")

    def insert(self, row, tag=None):
        row = np.asarray(row) % self.prime
        a = self.bits(row == 1)
        b = self.bits(row == 2) if self.prime == 3 else 0
        if tag is not None:
            a |= 1 << (self.width + tag)
        while leading := (a | b) & self.mask:
            bit = leading & -leading
            pivot = self.pivots.get(bit)
            if pivot is None:
                self.pivots[bit] = (a, b) if a & bit else (b, a)
                return None
            c, d = pivot
            if self.prime == 2:
                a ^= c
            else:
                if a & bit:
                    c, d = d, c
                az, cz = ~(a | b), ~(c | d)
                a, b = ((a & cz) | (az & c) | (b & d),
                        (b & cz) | (az & d) | (a & c))
        return a >> self.width, b >> self.width

    def relation(self, residue, count):
        length = (count + 7) // 8
        return sum(j * np.unpackbits(np.frombuffer(v.to_bytes(length, "little"), dtype=np.uint8),
                                     bitorder="little")[:count]
                   for j, v in enumerate(residue, 1)).astype(np.int64)


class RationalEchelon:
    def __init__(self, width):
        self.width = width
        self.pivots = {}

    @staticmethod
    def primitive(row):
        divisor = reduce(gcd, row.values(), 0)
        return {i: v // divisor for i, v in row.items()} if divisor > 1 else row

    def insert(self, vector, tag=None):
        row = {int(i): int(vector[i]) for i in np.flatnonzero(vector)}
        if tag is not None:
            row[self.width + tag] = 1
        while row:
            lead = min(row)
            if lead >= self.width:
                break
            pivot = self.pivots.get(lead)
            if pivot is None:
                self.pivots[lead] = self.primitive(row)
                return None
            a, b = row[lead], pivot[lead]
            g = gcd(a, b)
            scale, multiple = b // g, a // g
            if scale != 1:
                row = {i: v * scale for i, v in row.items()}
            for i, v in pivot.items():
                delta = row.get(i, 0) - multiple * v
                if delta:
                    row[i] = delta
                else:
                    row.pop(i, None)
            if abs(scale) != 1:
                row = self.primitive(row)
        return row

    def relation(self, residue, count):
        output = np.zeros(count, dtype=np.int64)
        for i, value in self.primitive(residue).items():
            output[i - self.width] = value
        return output


@dataclass(frozen=True)
class Input:
    name: str
    factors: np.ndarray
    target_scale: int


def quadratic(point, direction):
    a, b, c = point.transpose(1, 0, 2)
    u, v, w = direction.transpose(1, 0, 2)
    return sum(np.einsum("si,sj,sk->ijk", *parts, dtype=np.int64)
               for parts in ((u, v, c), (u, b, w), (a, v, w))).ravel()


def quadratic_columns(point, complement):
    squares = [quadratic(point, v) for v in complement]
    return [squares[i] if i == j else
            quadratic(point, u + complement[j]) - squares[i] - squares[j]
            for i, u in enumerate(complement) for j in range(i + 1)]


def modular_rank(rows, prime):
    rows = rows.copy() % prime
    rank = 0
    for column in range(rows.shape[1]):
        candidates = np.flatnonzero(rows[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + candidates[0]
        rows[[pivot, rank]] = rows[[rank, pivot]]
        normalized = rows[rank] * pow(int(rows[rank, column]), -1, prime) % prime
        active = rank + 1 + np.flatnonzero(rows[rank+1:, column])
        rows[active] = (rows[active] - rows[active, column, None] * normalized) % prime
        rank += 1
        if rank == len(rows):
            break
    return rank


def exact_product(left, right):
    bound = lambda x: int(np.abs(x).max()) if x.size else 0
    require(left.shape[-1] * bound(left) * bound(right) < 2**63, "integer product can overflow")
    right = right.astype(np.int64)
    output = np.empty((len(left), right.shape[1]), dtype=np.int64)
    for i, row in enumerate(left):
        support = np.flatnonzero(row)
        output[i] = row[support] @ right[support]
    return output


def audit_rational_certificate(point, jac, orbit_generators, file):
    with np.load(file, allow_pickle=False) as source:
        kernel = source["kernel"]
        lifts = source["preimages"]
        denominators = source["denominators"]
    lower = ModularEchelon(jac.shape[1], 3)
    for row in jac:
        lower.insert(row)
    jacobian_rank = len(lower.pivots)
    require(kernel.shape == (point.size - jacobian_rank, point.size),
            "wrong rational kernel shape")
    require(modular_rank(kernel, 5) == len(kernel), "rational kernel fails independence check")
    require(not np.any(exact_product(kernel, jac)), "invalid exact kernel vector")

    orbit_span = RationalEchelon(point.size)
    quotient = ModularEchelon(point.size, 3)
    for vector in orbit_generators:
        orbit_span.insert(vector)
        quotient.insert(vector)
    orbit_rank = len(orbit_span.pivots)
    require(orbit_rank == len(quotient.pivots), "orbit reduction loses rank")
    complement = [v.reshape(point.shape) for v in kernel if quotient.insert(v) is None]
    require(len(quotient.pivots) == len(kernel), "rational complement is incomplete")
    quadrics = np.asarray(quadratic_columns(point, complement))
    require(lifts.shape == (len(quadrics), point.size)
            and denominators.shape == (len(quadrics),)
            and np.all(denominators > 0), "invalid quadratic preimage shape")
    require(int(np.abs(quadrics).max()) * int(denominators.max()) < 2**63,
            "scaled quadratic can overflow")
    require(np.array_equal(exact_product(lifts, jac), denominators[:, None] * quadrics),
            "invalid quadratic preimage")
    return jacobian_rank, len(kernel), orbit_rank, len(complement), 0


def audit_by_elimination(point, jac, orbit_generators, prime, make):
    j_elim = make(jac.shape[1])
    if prime is None:
        columns = np.argsort(np.count_nonzero(jac, axis=0), kind="stable")
        row_order = np.argsort(np.count_nonzero(jac, axis=1), kind="stable")
    else:
        columns = slice(None)
        row_order = range(len(jac))
    kernel = []
    for i in row_order:
        relation = j_elim.insert(jac[i, columns], tag=int(i))
        if relation is not None:
            kernel.append(j_elim.relation(relation, point.size))
    jacobian_rank = len(j_elim.pivots)
    require(len(kernel) == point.size - jacobian_rank, "kernel dimension mismatch")
    products = exact_product(np.array(kernel).reshape(-1, point.size), jac)
    require(not np.any(products if prime is None else products % prime), "invalid kernel vector")

    quotient = make(point.size)
    for row in orbit_generators:
        quotient.insert(row)
    orbit = len(quotient.pivots)
    complement = [v.reshape(point.shape) for v in kernel if quotient.insert(v) is None]
    require(len(quotient.pivots) == len(kernel), "orbit and kernel dimension disagree")
    if complement:
        scale = max(int(np.abs(v).max()) for v in complement)
        bound = int(np.abs(point).max())
        require(6*len(point)*bound*(2*scale)**2 < 2**63, "quadratic buffer overflow")
    for obstruction in quadratic_columns(point, complement):
        j_elim.insert(obstruction[columns])
    return jacobian_rank, len(kernel), orbit, len(complement), len(j_elim.pivots) - jacobian_rank


def audit(input_, field):
    prime = {"F2": 2, "F3": 3, "Q": None}[field]
    make = RationalEchelon if prime is None else lambda width: ModularEchelon(width, prime)
    point = input_.factors.astype(np.int64)
    expected = target(4).astype(np.int64) * input_.target_scale
    if prime:
        require(input_.target_scale % prime != 0, "scaling denominator not invertible")
        point %= prime
    bound = int(np.abs(point).max())
    require((bound+1)**2 <= np.iinfo(np.int16).max
            and len(point)*(bound+1)**3 < np.iinfo(np.int32).max, "Jacobian buffer overflow")
    residual = np.einsum("si,sj,sk->ijk", *point.transpose(1, 0, 2), dtype=np.int64) - expected
    require(not np.any(residual if prime is None else residual % prime), "Brent identity failed")

    jac = jacobian_rows(point.astype(np.int32))
    orbit_generators = np.array(list(symmetry_directions(point, scales=True))).reshape(-1, point.size)
    products = exact_product(orbit_generators, jac)
    require(not np.any(products if prime is None else products % prime), "invalid orbit direction")
    file = ROOT / f"rational-{input_.name}.npz"
    if prime is None and file.exists():
        ranks = audit_rational_certificate(point, jac, orbit_generators, file)
    else:
        ranks = audit_by_elimination(point, jac, orbit_generators, prime, make)
    jacobian_rank, tangent, orbit, transverse, obstruction_rank = ranks
    return {
        "scheme": input_.name, "field": field, "brent_equations_checked": 4096,
        "jacobian_rank": jacobian_rank, "tangent_dimension": tangent,
        "orbit_differential_rank": orbit, "transverse_dimension": transverse,
        "quadratic_obstruction_rank": obstruction_rank,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", choices=("F2", "F3", "Q"))
    args = parser.parse_args()
    # A pure-square obstruction survives in characteristic two on this input.
    one = np.array([[[0], [0], [1]]], dtype=np.int64)
    direction = np.array([[[1], [1], [0]]], dtype=np.int64)
    require(quadratic_columns(one, [direction])[0].item() % 2 == 1,
            "characteristic-two square term was lost")
    with np.load(ROOT / "inputs.npz", allow_pickle=False) as data:
        for entry in json.loads((ROOT / "inputs.json").read_text()):
            for field in entry["fields"]:
                if args.field is None or field == args.field:
                    item = Input(entry["name"], data[entry["name"]], entry["target_scale"])
                    print(json.dumps(audit(item, field)), flush=True)


if __name__ == "__main__":
    main()
