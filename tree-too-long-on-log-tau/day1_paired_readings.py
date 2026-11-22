"""Day 1 - Paired readings.

The noise of a rejection share project ran six masses from the same studies
and the same warmed states and still found gaps of -0.20 to +0.125 a set,
no tighter than two independent sets, because the chains parted after one
transition. The tool this project needs is a reading where two samplers
differ only in what is being compared. This file hands NUTS its random
numbers from outside: the momentum's normals, a direction coin per
doubling, a uniform per leaf numbered by its place in the whole tree and a
uniform per merge, all from a stream fixed by (seed, transition), so two
samplers on one seed use the same number at every decision they both
reach. Then the mass matrix's NUTS after the 100 window, per-chain and
pooled over a study's chains, on two fresh sets of 40 studies, with the
pooled mass run on the per-chain run's seed and on a second one. Written
before the run: on one seed the two masses' chains stay close or meet, a
study's verdict agrees between them far more often than between two
seeds, and the sd of a paired gap falls well under day 4's.

0. The handed-in sampler is still the sampler. On the ten-dimensional
   normal under the exact mass, variance over truth 0.997 to 1.003, mean
   within 0.002 of a scale, and one sampler twice on one seed differs by
   exactly 0.

1. Nothing ever meets. Not one chain in 4000 on the normal, at any of 1 to
   200 transitions, gets within 1e-6 of its partner, even with both on the
   exact mass up to a lognormal of 0.3. What one seed keeps there is a
   correlation of +0.65 between the two draws at 200; on two seeds it is
   -0.005. Against unit mass it is -0.015 on one seed, nothing: a mass that
   turns a coordinate at a different rate sends the same momentum somewhere
   else, and the tree's doublings and U-turns cut it differently from there.

2. On eight schools the pair holds a fixed distance. Per-chain against
   pooled on one seed is 0.19 apart in log tau (median) after one
   transition, 0.36 after ten and 0.37 at the end; on different seeds 0.70,
   0.83 and 0.82, and the same for one mass on two seeds. So common numbers
   keep the two chains at under half the unpaired distance, and the
   distance stops growing after ten transitions rather than shrinking.
   Wrong that they would meet.

3. A study's verdict does not see it. The joint verdict disagrees between
   the two masses on 31.2% of the 80 studies paired and 30.0% unpaired, and
   between two seeds of one mass on 28.7%. The sd of a gap at 40 studies is
   0.088 paired, 0.086 unpaired, 0.085 for the noise alone, and it takes 125,
   119 and 115 studies to read a 0.1 gap at 2 sd. On log tau alone pairing
   buys a little, 25.0% against 28.7% disagreeing, sd 0.077 against 0.082,
   94 studies against 108. Wrong: the paired sd is day 4's.

4. Why. The band on a study reads the ranks of 500 simulations, and a rank
   is the count of 99 draws below the truth. Paired or not, a chain's ranks
   track each other at +0.98 against +0.96 on log tau, because where the
   truth sits sets most of a rank and the sampler only the rest. What
   decides a borderline study is that rest, and the rest at 0.37 apart is as
   free as at 0.82. The 0.4 the band sits on is not moved by pairing.

5. The masses themselves, on these 80. Per-chain 0.425 / 0.575 joint, pooled
   0.525 / 0.400 on the same seed and 0.350 / 0.550 on the other, log tau
   alone 0.41 per-chain against 0.29 and 0.28 pooled, gap +0.125 paired and
   +0.137 unpaired. Day 4's log tau gap of 0.09 to 0.14 again, now with the
   same studies, the same states and the same numbers, so it is the mass and
   not the draw. lag-1 on log tau +0.40 / +0.43 per-chain, +0.35 to +0.38
   pooled, 38 steps against 40.3.

So the paired design this project was to stand on saves nothing on the
joint and an eighth of the studies on log tau. Day 2 reads the tree's
length on log tau per transition, which needs no pairing; day 3's cut tree
gets its studies from the count, 120 or more, not from common numbers.

NumPy only. Fixed seeds. About 2750 seconds, 720 of them the normal.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mass-matrix-from-warmup"))
import day1_diagonal_mass_sweep as m1  # noqa: E402  (the warmup and lag1; puts the older projects on the path)
from day1_diagonal_mass_sweep import d1, d2, es, KEYS, draw_studies, params  # noqa: E402

# The mass matrix modules reseed on import; reseed again so these studies are fresh.
es.RNG = np.random.default_rng(211)
d1.RNG = np.random.default_rng(212)
d2.RNG = np.random.default_rng(213)

L, SIZES = d1.L, d1.SIZES
EPS, STUDIES, PASS, BURN, LAG_CHAINS = m1.EPS, m1.STUDIES, m1.PASS, m1.BURN, m1.LAG_CHAINS
MAX_DEPTH = 8
LEAVES = 2 ** MAX_DEPTH - 1       # one uniform for every leaf the deepest tree can grow
SETS = 2
# Each run is (label, which mass, noise seed). Runs that share a seed share every random number, chain by chain.
RUNS = [("per-chain A", "chain", "A"), ("pooled A", "pooled", "A"), ("pooled B", "pooled", "B")]


def noise(seed, t, c, d):
    """Every random number one NUTS transition of c chains can use, from a stream fixed by (seed, t).

    The momentum's standard normals, a direction coin per doubling, a
    uniform per leaf for the progressive draw inside a subtree, numbered by
    the leaf's place in the whole tree, and a uniform per doubling for the
    merge. Two samplers handed the same (seed, t) use the same numbers at
    every decision they both reach, whatever their mass; where one tree
    grows a leaf the other does not, that leaf's number goes unused.
    """
    g = np.random.default_rng([seed, t])
    return {"xi": g.standard_normal((c, d)), "dir": g.random((c, MAX_DEPTH)),
            "leaf": g.random((c, LEAVES)), "merge": g.random((c, MAX_DEPTH))}


def nuts_crn(z, eps, minv, u, pot_grad=None):
    """The mass matrix project's NUTS (multinomial draw) with its random numbers handed in.

    Line for line the tree of mass matrix day 2 under a per-chain diagonal
    mass; the only change is that the momentum, the direction coins and the
    three kinds of uniform come from u = noise(seed, t, ...) instead of a
    generator, so a sampler is a deterministic map of (state, mass, u).

    Returns (z, depth reached, leapfrog steps taken, diverged).
    """
    pot, grad = pot_grad if pot_grad is not None else es.MODELS["noncentred"]
    c, d = z.shape
    p0 = u["xi"] / np.sqrt(minv)
    g0 = grad(z)
    h0 = pot(z) + 0.5 * (minv * p0 * p0).sum(1)
    zm, pm, gm = z.copy(), p0.copy(), g0.copy()
    zp, pp, gp = z.copy(), p0.copy(), g0.copy()
    prop, lw, rho = z.copy(), np.zeros(c), p0.copy()
    live = np.ones(c, bool)
    depth = np.zeros(c, np.int32)
    steps = np.zeros(c, np.int32)
    diverged = np.zeros(c, bool)
    for j in range(MAX_DEPTH):
        fwd = u["dir"][:, j] < 0.5
        f = fwd[:, None]
        zs, ps, gs = np.where(f, zp, zm), np.where(f, pp, pm), np.where(f, gp, gm)
        e = np.where(fwd, eps, -eps)[:, None]
        ok = live.copy()
        s_lw = np.full(c, -np.inf)
        s_prop = zs.copy()
        S = np.zeros((c, d))
        ck_p = np.empty((j + 1, c, d))
        ck_S = np.empty((j + 1, c, d))
        for n in range(2 ** j):
            p1 = ps - 0.5 * e * gs
            z1 = zs + e * minv * p1
            g1 = grad(z1)
            p1 = p1 - 0.5 * e * g1
            step = ok[:, None]
            zs, ps, gs = np.where(step, z1, zs), np.where(step, p1, ps), np.where(step, g1, gs)
            steps += ok
            dh = pot(zs) + 0.5 * (minv * ps * ps).sum(1) - h0
            bad = ok & (~np.isfinite(dh) | (np.nan_to_num(dh, nan=0.0, posinf=0.0) > 1000))
            diverged |= bad
            ok &= ~bad
            leaf = np.where(np.isfinite(dh), -dh, -np.inf)
            grown = np.logaddexp(s_lw, leaf)
            take = ok & (np.log(u["leaf"][:, 2 ** j - 1 + n]) < leaf - grown)
            s_prop = np.where(take[:, None], zs, s_prop)
            s_lw = np.where(ok, grown, s_lw)
            for k in range(1, j + 1):
                if n % 2 ** k == 0:
                    ck_p[k], ck_S[k] = ps, S
            S = S + ps
            for k in range(1, j + 1):
                if (n + 1) % 2 ** k == 0:
                    r = S - ck_S[k]
                    ok &= ((r * minv * ck_p[k]).sum(1) > 0) & ((r * minv * ps).sum(1) > 0)
        merge = ok
        mf, mb = (merge & fwd)[:, None], (merge & ~fwd)[:, None]
        zp, pp, gp = np.where(mf, zs, zp), np.where(mf, ps, pp), np.where(mf, gs, gp)
        zm, pm, gm = np.where(mb, zs, zm), np.where(mb, ps, pm), np.where(mb, gs, gm)
        take = merge & (np.log(u["merge"][:, j]) < s_lw - lw)
        prop = np.where(take[:, None], s_prop, prop)
        lw = np.where(merge, np.logaddexp(lw, s_lw), lw)
        rho = np.where(merge[:, None], rho + S, rho)
        depth = np.where(merge, j + 1, depth)
        live = merge & ((rho * minv * pm).sum(1) > 0) & ((rho * minv * pp).sum(1) > 0)
        if not live.any():
            break
    return prop, depth, steps, diverged


def lockstep(z, minvs, seeds, n, pot_grad=None, every=None):
    """Run several samplers from one start for n transitions, one noise stream per distinct seed.

    Returns the final states and, at the transitions in every, each run's
    states, so the distance between two runs can be read as it grows.
    """
    zs = [z.copy() for _ in minvs]
    snaps = {}
    for t in range(n):
        us = {s: noise(s, t, z.shape[0], z.shape[1]) for s in set(seeds)}
        zs = [nuts_crn(x, EPS if pot_grad is None else 0.2, m, us[s], pot_grad)[0] for x, m, s in zip(zs, minvs, seeds)]
        if every is not None and t + 1 in every:
            snaps[t + 1] = [x.copy() for x in zs]
    return zs, snaps


def check_on_normal(chains=4000, n=200, burn=50):
    """Ten-dimensional normal, scales 0.5 to 5, eps 0.2: is the handed-in sampler still a sampler, and what pairs.

    a. exact mass, a fresh seed per transition: variance over truth and mean
       over scale, as on mass matrix day 2.
    b. the same sampler twice on one seed: the largest gap, which has to be 0.
    c. exact mass against exact mass times a per-chain lognormal of sd 0.3, the
       spread of a window's estimate: the share of chains within 1e-6 of each
       other and the correlation of their draws, on one seed and on two.
    d. unit mass against exact mass, the same.
    """
    s = np.linspace(0.5, 5.0, 10)
    pg = (lambda x: 0.5 * ((x / s) ** 2).sum(1), lambda x: x / s ** 2)
    exact = np.tile(s ** 2, (chains, 1))
    g = np.random.default_rng(214)
    z0 = s * g.standard_normal((chains, 10))
    out = {}

    (za,), snaps = lockstep(z0, [exact], [2141], burn + n, pg, every=set(range(burn + 1, burn + n + 1)))
    draws = np.stack([snaps[t][0] for t in sorted(snaps)])
    out["a"] = (draws.var((0, 1)) / s ** 2, draws.mean((0, 1)) / s)

    (x, y), _ = lockstep(z0, [exact, exact], [2142, 2142], 20, pg)
    out["b"] = np.abs(x - y).max()

    rough = exact * np.exp(0.3 * g.standard_normal((chains, 10)))
    unit = np.ones((chains, 10))
    marks = (1, 5, 20, 50, 200)
    for key, other in (("c", rough), ("d", unit)):
        rows = []
        for seeds in ((2143, 2143), (2143, 2144)):
            _, snaps = lockstep(z0, [exact, other], list(seeds), max(marks), pg, every=set(marks))
            met = [(np.abs(snaps[t][0] - snaps[t][1]).max(1) < 1e-6).mean() for t in marks]
            a, b = snaps[max(marks)]
            corr = np.mean([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(10)])
            rows.append((met, corr))
        out[key] = rows
    return out, marks


def pooled(minv):
    """Each study's inverse mass averaged over its chains and handed back to every one, as noise of a share day 4."""
    nmax = SIZES[-1]
    return np.repeat(minv.reshape(STUDIES, nmax, -1).mean(1), nmax, axis=0)


def hits(rr, stats, crits):
    """Per study: rejected by the band on any of the four parameters (the joint), and on log tau alone."""
    band, crit = stats["band"], crits["band"]
    alone = np.stack([band(d1.rank_counts(rr[:, :, p])) > crit for p in range(rr.shape[2])], 1)
    return alone.any(1), alone[:, 1]


def sbc_lockstep(z, truth, minvs, seeds):
    """BURN transitions then L with the ranks, every run from the same warmed state, the runs in lockstep.

    Returns per run the ranks, the pilots for the lag-1, mean steps, and the
    median distance in log tau to every other run after 1, 10 and BURN + L
    transitions with the share of chains that have met (all ten coordinates
    within 1e-6).
    """
    r, k = z.shape[0], len(minvs)
    zs = [z.copy() for _ in range(k)]
    below = [np.zeros((r, 4), np.int32) for _ in range(k)]
    kept = [np.empty((L, LAG_CHAINS, 3)) for _ in range(k)]
    steps = [0.0] * k
    dist = {}
    marks = (1, 10, BURN + L)
    for t in range(BURN + L):
        us = {s: noise(s, t, r, z.shape[1]) for s in set(seeds)}
        for i in range(k):
            zs[i], _, st, _ = nuts_crn(zs[i], EPS, minvs[i], us[seeds[i]])
            if t >= BURN:
                steps[i] += st.mean() / L
                p = params("noncentred", zs[i])
                kept[i][t - BURN] = p[:LAG_CHAINS][:, [0, 1, 3]]
                below[i] += p < truth
        if t + 1 in marks:
            for i in range(k):
                for j in range(i + 1, k):
                    gap = np.abs(zs[i] - zs[j])
                    dist[i, j, t + 1] = (np.median(gap[:, 1]), (gap.max(1) < 1e-6).mean())
    return below, kept, steps, dist, marks


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    nmax = SIZES[-1]
    r = STUDIES * nmax

    print("0. ten-dimensional normal, scales 0.5 to 5, eps 0.2, 4000 chains, the sampler with its numbers handed in")
    chk, marks = check_on_normal()
    vr, mz = chk["a"]
    print(f"  a. exact mass, fresh numbers: variance / truth {vr.min():.3f} to {vr.max():.3f}, "
          f"mean / scale within {np.abs(mz).max():.3f}")
    print(f"  b. one sampler twice on one seed: largest gap {chk['b']:.1e}")
    for key, lab in (("c", "exact against exact x lognormal(0.3)"), ("d", "exact against unit")):
        for (met, corr), how in zip(chk[key], ("one seed", "two seeds")):
            print(f"  {key}. {lab}, {how}: met after " + " / ".join(f"{t}" for t in marks) + " transitions "
                  + " / ".join(f"{m:.3f}" for m in met) + f", correlation of draws at {marks[-1]} {corr:+.3f}")
    print(f"  ({time.time() - t0:.0f} s)", flush=True)

    stats, crits = {}, {}
    s, _ = d2.shape_stats(nmax)
    stats = {k: s[k] for k in KEYS}
    crits = {k: v for k, v in d2.critical_values(dict(s, n=nmax)).items() if k in KEYS}

    print(f"\npaired readings: non-centred NUTS at eps {EPS}, max depth {MAX_DEPTH}, {SETS} sets of {STUDIES} x {nmax}, "
          f"mass from the 25 + 50 + 100 windows, per-chain or pooled over a study; runs on seed A share every number")
    joint = {lab: [] for lab, _, _ in RUNS}
    lt = {lab: [] for lab, _, _ in RUNS}
    ranks = {lab: [] for lab, _, _ in RUNS}
    for rep in range(SETS):
        truth, y = draw_studies(r)
        es.Y = y
        z, minv, _ = m1.warmup(r)
        minvs = [minv if kind == "chain" else pooled(minv) for _, kind, _ in RUNS]
        seeds = [2150 + 10 * rep + (0 if sd == "A" else 1) for _, _, sd in RUNS]
        below, kept, steps, dist, dmarks = sbc_lockstep(z, truth, minvs, seeds)
        print(f"\nset {rep + 1}  ({time.time() - t0:.0f} s)")
        for i, (lab, _, _) in enumerate(RUNS):
            rr = below[i].reshape(STUDIES, nmax, 4)
            hj, hl = hits(rr, stats, crits)
            joint[lab].append(hj)
            lt[lab].append(hl)
            ranks[lab].append(below[i])
            lag = m1.lag1(kept[i])
            print(f"  {lab:12s} joint {hj.mean():.3f}  log tau {hl.mean():.3f}  steps {steps[i]:5.1f}  "
                  f"lag-1 mu {lag[0]:+.2f} log tau {lag[1]:+.2f}")
        for (i, j, t), (med, met) in sorted(dist.items()):
            print(f"  {RUNS[i][0]} / {RUNS[j][0]} after {t:3d}: median |gap| in log tau {med:.4f}, met {met:.3f}")

    print(f"\npairs over the {SETS * STUDIES} studies: the gap in share, the share of studies the two disagree on, "
          f"the sd of a gap at {STUDIES} studies from that, and the correlation of the chains' log tau ranks")
    pairs = [("per-chain A", "pooled A", "paired, two masses"), ("per-chain A", "pooled B", "unpaired, two masses"),
             ("pooled A", "pooled B", "one mass, two seeds")]
    for a, b, how in pairs:
        for name, src in (("joint", joint), ("log tau", lt)):
            x, w = np.concatenate(src[a]), np.concatenate(src[b])
            gap = x.mean() - w.mean()
            disc = (x != w).mean()
            sd40 = np.sqrt(max(disc - gap ** 2, 0.0) / STUDIES)
            need = int(np.ceil((2 * np.sqrt(max(disc - gap ** 2, 1e-9)) / 0.1) ** 2))
            print(f"  {how:21s} {name:7s} gap {gap:+.3f}  disagree {disc:.3f}  sd of a gap at {STUDIES} {sd40:.3f}  "
                  f"studies to read 0.1 at 2 sd {need}")
        ra, rb = np.concatenate(ranks[a])[:, 1], np.concatenate(ranks[b])[:, 1]
        print(f"  {how:21s} rank correlation on log tau {np.corrcoef(ra, rb)[0, 1]:+.3f}")

    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
