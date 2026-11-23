"""Independent row-wise ranks of the binary Jacobian and its arithmetic augmentation."""
from itertools import product
from pathlib import Path
import json

import numpy as np


def rank(rows):
    pivots = {}
    for row in rows:
        while row:
            lead = row.bit_length()
            if lead not in pivots:
                pivots[lead] = row
                break
            row ^= pivots[lead]
    return len(pivots)


def check(point):
    # Use Python integers for row bitsets. No matrix or witness from the main checker is read.
    layers = point.tolist()
    rows, augmented = [], []
    for i, j, k in product(range(16), repeat=3):
        target = int(i % 4 == j // 4 and k == 4*(i // 4) + j % 4)
        residual = -target
        row = 0
        for s, (a, b, c) in enumerate(layers):
            u, v, w = a[i], b[j], c[k]
            residual += u*v*w
            bits = ((v*w << i) | (u*w << (16+j)) | (u*v << (32+k)))
            row |= bits << (48*s)
        if residual % 2:
            raise ValueError("invalid binary tensor")
        rows.append(row)
        augmented.append(row | (((residual // 2) & 1) << (48*len(layers))))
    ranks = [rank(rows), rank(augmented)]
    if ranks != [2117, 2118]:
        raise ValueError(f"unexpected ranks {ranks}")
    return ranks


if __name__ == "__main__":
    with np.load(Path(__file__).with_name("inputs.npz"), allow_pickle=False) as data:
        names = ["AT47", "KM47", "C3", "C4", "C5", "Gamma47-B"]
        print(json.dumps({"independent_binary_row_ranks": {s: check(data[s]) for s in names}}))
