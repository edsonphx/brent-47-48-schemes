# Rigidity of 47-term schemes and rational deformations of a 48-term scheme for 4×4 matrix multiplication

This is an offline research package for the ordered Brent scheme of \(4\times4\) matrix multiplication. It includes an editable LaTeX manuscript, compiled PDF, coefficient and change-of-basis certificates, exact rational curves, and field audits. The [Strassen² companion paper](https://doi.org/10.5281/zenodo.24449376) studies the recursive 49-term point.

The binary results concern five symmetry orbits and every public representative explicitly matched to them in [Tomar, Appendix B.1](https://doi.org/10.5281/zenodo.23604141). For DPS48 the full local Krull dimension in characteristic zero is bounded by **143 and 151**. Each of four twelve-term restrictions has exact local dimension **25**, and each symmetry-saturated surface closure has dimension **143**. The unrestricted local ring agrees with the polynomial ring on 151 variables through degree four over the rationals. Its Hilbert function in these degrees is **1, 151, 11,476, 585,276, 22,533,126**.

The [previously archived version](https://doi.org/10.5281/zenodo.24458306) bears the earlier title *Exact local and arithmetic geometry of 47- and 48-term schemes for 4×4 matrix multiplication*. Cite the Git revision alongside the DOI when using the new rational fourth-order certificate.

## Replay

Run `bash reproduce.sh` with Python 3.9 or later, NumPy, and SymPy installed. The verifier uses no network access. It checks the SHA-256 manifest, runs the independent certificate checks concurrently, and compares the resulting JSON lines against the frozen output. `paper.tex` uses standard AMS LaTeX packages and an inline bibliography.

The optional `slide.py` needs SciPy. It is a floating-point experiment with a fixed symmetry gauge and a Gauss–Newton residual tolerance. The recorded run covered all 21 displayed tables and accepted no projected step at the specified tolerance. The exact searches and replay use rational identities, matrix ranks modulo a specified prime, and polynomial certificates.

Two shorter checks are available. `python3 binary_crosscheck.py` independently builds every equation row and checks the binary and augmented ranks using Python integer arithmetic. `python3 dps48_local.py` verifies the rational ten-parameter fourth jet at 715 interpolation nodes and computes the four restricted Jacobian ranks at both specified base points. The coefficient archive `dps48-jets-Q.npz` stores the cubic and quartic arrays as signed integers in lexicographic monomial order; the linear and quadratic arrays come from the existing rational certificate.

With the optional python-flint package, run `python3 regenerate_dps48_jets.py` to rebuild the cubic and quartic coefficients from the quadratic data and compare them entry by entry. The default verifier checks the polynomial identities directly and does not need FLINT.

The reviewer draft also proves the exact minimum external support 16 over the rationals at D(2). The integer witness in `dps48-external-Q.json` has 442 nonzero coordinates. `python3 dps48_support.py` checks the exact tensor derivative, the support, and the independent reduction. Run `python3 dps48_catalogue.py` for the 21-table classification. The four full-symmetry classes have sizes 18, 1, 1 and 1. Each has a representative on chart one, including the published twelve-term replacement at both `(2,-1/2)` and `(2,-3/2)`.

## Exact results

- **Five binary orbits.** The six input tables all have Jacobian rank 2,117 and orbit differential rank 139. Their five distinct orbits are separated by sorted rank-triple multisets. Small vertex commutator matrices independently certify the orbit dimension 139 for the five representatives. The explicit Gamma47-B to C5 map includes a cyclic rotation, three binary basis changes, and a term permutation. The six distinct parity-witness presentations include two for the same orbit.
- **Mixed characteristic.** The parity checks certify that the first Hensel lift fails. The augmented linear rank is 2,118, which gives embedding dimension 139 in the integral local ring. Its characteristic-two fiber already has dimension 139. Regularity and this dimension equality force 2 to vanish locally, excluding even ramified lifts at all five orbits.
- **Rational orbit equivalence.** The file `dps48-equivalence.json` contains the explicit rational transformation from DPS48 to DPS48-accurate. It verifies termwise after a cyclic mode rotation, layer rescalings and a permutation. This identifies their full symmetry orbits. Ordered irreducible component membership outside this equivalence remains open beyond the explicit orbit relation.
- **Four exact rational surfaces.** Each reconstruction fixes 36 terms of \(D(2)\) and varies a different union of three of Agarwal's four core groups. The rational univariate formulas are checked coefficientwise through degree-bounded exact interpolation in all 4,096 equations. The Laurent normal form transports each of the four curves along \(D(t)\), giving four separate two-parameter identities. Their four directions and the published curve direction are jointly independent modulo the symmetry tangent space at \(t=1/4\) and at \(t=2\).
- **Boundary exclusion.** In the affine closure of the full symmetry saturation of each surface over \(\overline{\mathbb Q}\), there is no zero summand and no pair with proportional factors in two modes. The proof uses gcds and valuations on the reconstructed curves, transported support separation in two modes for each outside term, and Laurent-unit loop bases from the fixed terms to control singular gauges. Two fixed-term trace invariants, \(-1/(4t)\) and \(4t\), exclude finite limits at the ends of the original \(t\)-parameter. See the manuscript for the valuation-arc argument. The original Li–Wang–Hu curve has its separate Cycle 90 boundary certificate over \(\overline{\mathbb Q}\) and \(\overline{\mathbb F}_3\).
- **Remaining scope.** The ten-dimensional transverse space at DPS48 has simultaneous lifts through degree four over both \(\mathbb Q\) and \(\mathbb F_3\). The four exact surfaces integrate five independent directions individually and do not provide a common five- or ten-parameter parametrization. The full DPS48 component and its boundary remain open. No 47-term rational decomposition was found.

## Version 1.1.0

At the ordered point D(2) over the field with three elements, at least **16 of the 32 external summands** must move to obtain a tangent class outside the five known transverse directions. The core may vary freely. The bound is attained, and it concerns first-order deformations.

Run \`python3 dps48_support.py\` to regenerate the rank and covering certificates and print a sparse tangent witness. The script checks **40 rank-51 quartets** and **56 covered five-element sets** in each of the four rows of the external partition. Run \`python3 dps48_superposition.py\` for exact counterexamples to all six additive pairings of the stored core curves at D(2). It also certifies that a generic point of each curve has only rescaling tangents on any of the other three stored supports. Both checks run in the offline replay.

The results in this release concern 4×4 multiplication. The original v1.0.0 artifact is archived under [DOI 10.5281/zenodo.24454159](https://doi.org/10.5281/zenodo.24454159).

## Scan and search record

The exact scan includes 21 displayed rational tables, counting ordinary basis versions, alternative encodings restored to ordinary coordinates, and published replacements. All have Jacobian rank 2,153 and orbit differential rank 141 over \(\mathbb F_3\). The stabilizer has dimension three over \(\mathbb Q\) for the full connected acting group and dimension zero after quotienting by its scalar kernel. The fingerprints report each sorted factor-rank triple and the Jordan data of the three cyclic loop matrices. They are invariants, with no general converse. The published Agarwal replacement has rank-three factors and is separated from the rank-one/rank-two fingerprint of the DPS presentations.

Every scanned reduction has zero ordinary flip pairs. The package tests a bounded search through 49 terms using additive plus moves followed by ordinary flips. The two starting reductions reach mod 9 and mod 27. The walk found no new 48-term endpoint in the recorded search, so there was no new endpoint to lift.

For the collision search, the displayed score is the sum of the two smallest normalized squared wedge norms of a pair, minimized over all pairs. Normalization removes individual factor scalings. The numeric value still depends on the chosen matrix basis. The lowest exact value in the recorded rational grid is \(3744529211/14227290942803\) at \(h=8\) in the first chart, for pair \((28,40)\) and modes \(2,3\). This is the minimum over the sampled points. The algebraic surface certificates, rather than this positive number, rule out collisions everywhere on the covered closures.

## Sources and provenance

- [Tomar's binary representatives and checks](https://doi.org/10.5281/zenodo.23652770)
- [Kauers–Moosbauer flip graph and coefficient tables](https://github.com/jakobmoosbauer/flips/tree/e31a0a0f0d2577cee5da047ca7dcae0c61992e40). The repository has the 47-term mod-two and 49-term characteristic-zero \(4\times4\) tables.
- [Dumas–Pernet–Sedoglavic rational construction](https://arxiv.org/abs/2506.13242)
- [PLinOpt natural and alternative bases](https://github.com/jgdumas/plinopt/tree/282b712d095ee64c4d83da22801acba223bbae96/data), including the sparse and canonical presentations described in the [recursive implementation paper](https://arxiv.org/abs/2609.12027).
- [Moran–Schwartz–Yuan, Appendix A](https://arxiv.org/html/2602.13171v1#A1)
- [Li–Wang–Hu, Appendix A](https://arxiv.org/html/2607.15069v2#A1)
- [Agarwal's core normal form and twelve-term replacement](https://arxiv.org/html/2609.13596#S8), with [coefficient evidence](https://doi.org/10.5281/zenodo.22684265).
- [Arai–Ichikawa–Hukushima on plus transitions](https://arxiv.org/abs/2312.16960)

`inputs.json` and `scan-inputs.json` specify the provenance and scaling conventions of the frozen coefficients. An alternative encoding is composed as \(L_{\rm alt}L_{\rm cob},R_{\rm alt}R_{\rm cob},P_{\rm cob}P_{\rm alt}\), and the resulting 48-term tensor is checked directly. In the chart JSON, polynomial coefficients are listed in ascending degree, and the 12 moving terms are stored as 16 times their true factor vectors after dividing by the provided common polynomial denominator. The last factor is in direct output order. All files are read without pickling.
