"""Day 2 - Where the tree stops.

The mass matrix project guessed the tree runs too long on log tau's side,
so the draw comes back toward where log tau started and the lag-1 stays
positive. Day 1 showed pairing does not help a verdict, and this question
needs none: it is about one tree at a time. So day 1's tree is traced.
Every merged leaf keeps its mu and log tau at its signed step from the
start, a subtree that is not merged is wiped from the record, and the
draw's step is carried through the progressive and merge choices. On one
set of 40 studies x 500 from one warmed state, 50 burn then 99 traced
transitions, under the per-chain mass after the 100 window, the same
pooled over a study, and unit mass. Written before the run: log tau turns
early inside the tree, the draw lands past that turn on most transitions,
and the studies rejected on log tau are the ones where it does so most.

0. The traced tree is the tree. On the same numbers it gives day 1's draws
   and step counts exactly (gap 0), and the draw sits on the kept
   trajectory for every chain.

1. Log tau does turn early. Under the per-chain mass it turns inside the
   tree on 0.81 of transitions, a median 10 steps out (one unit), on a tree
   of 34.4, and the draw lands past the turn on 0.61. Pooled 0.85 / 10 /
   0.64, unit 0.87 / 9 / 0.66. mu is the contrast: under either adapted
   mass it turns on 0.70 and the draw is past on 0.49; under unit mass it
   turns on 0.22 and is past on 0.15, the too-short tree of the trajectory
   project again, lag-1 +0.74.

2. The tree does not look at log tau. Its length runs from 37 steps when
   log tau starts below -1 to 33 above 2, while log tau's turn comes at 11
   steps in the neck and 5 or 6 at the top. So the overshoot is worst
   where log tau is large, not in the neck: past the turn 0.47 below -1,
   0.63 to 0.65 above 1. The draw's |jump| is 0.56 to 0.69 of the furthest
   log tau got in every bin, and the mean jump pulls back to the middle,
   +1.22 from below -1 and -0.92 from above 2.

3. Wrong on the part that matters. The 19 studies rejected on log tau
   under the per-chain mass draw past the turn on 0.605 of transitions and
   the 21 that pass on 0.605; jump over furthest 0.596 and 0.596. Pooled
   0.637 / 0.639, unit 0.660 / 0.659. And the mass that fails least on log
   tau, pooled at 0.300 against 0.475, overshoots more, not less. A draw
   past the turn is followed by a jump of -0.31 along its direction against
   -0.40 for one before it, so it does not even come back more.

4. So the tree overshoots log tau on most transitions, under every mass,
   on every study alike, and that is not what the band reads. Whatever
   fails a study on log tau is not in the tree's shape on the average
   transition. The joint is 0.575 / 0.400 / 0.950 and log tau 0.475 / 0.300
   / 0.375, lag-1 on log tau +0.41 / +0.36 / +0.37, the same order as day 1.

Day 3 cuts the tree anyway, depth 3 to 6 and the draw from the first half,
on 120 or more studies: if overshooting is harmless, a shorter tree should
leave log tau's share where it is and only cost mu.

NumPy only. Fixed seeds. About 1470 seconds, a third a mass.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import day1_paired_readings as p1  # noqa: E402  (noise, nuts_crn, pooled, hits; puts the older projects on the path)
from day1_paired_readings import m1, d1, d2, es, KEYS, draw_studies, params  # noqa: E402

es.RNG = np.random.default_rng(221)
d1.RNG = np.random.default_rng(222)
d2.RNG = np.random.default_rng(223)

L, SIZES, EPS, STUDIES, BURN, LAG_CHAINS = p1.L, p1.SIZES, p1.EPS, p1.STUDIES, p1.BURN, p1.LAG_CHAINS
MAX_DEPTH = p1.MAX_DEPTH
W = 2 ** MAX_DEPTH                  # the furthest a leaf can sit from the start, either way
BINS = (-np.inf, -1.0, 0.0, 1.0, 2.0, np.inf)   # log tau at the start of a transition
COORDS = ((0, "mu"), (1, "log tau"))
RUNS = [("per-chain", "chain", 2210), ("pooled", "pooled", 2211), ("unit", "unit", 2212)]


def nuts_traced(z, eps, minv, u):
    """nuts_crn with the trajectory kept: every merged leaf's mu and log tau at its signed step from the start.

    The tree is day 1's line for line. Beside it, a leaf grown forward in a
    doubling sits at hi + 1 + n and one grown backward at lo - 1 - n, and a
    subtree that is not merged (a U-turn inside it, a divergence) is wiped
    from the record, so what is left is the trajectory the draw was made
    from. The position of the draw is carried through the progressive and
    the merge choices the same way as the state.

    Returns (z, steps, traj (c, 2W+1, 2) with nan off the tree, lo, hi, draw position).
    """
    pot, grad = es.MODELS["noncentred"]
    c, d = z.shape
    rows = np.arange(c)
    cols = np.arange(2 * W + 1)
    traj = np.full((c, 2 * W + 1, 2), np.nan)
    traj[:, W] = z[:, :2]
    lo, hi = np.zeros(c, np.int64), np.zeros(c, np.int64)
    p0 = u["xi"] / np.sqrt(minv)
    g0 = grad(z)
    h0 = pot(z) + 0.5 * (minv * p0 * p0).sum(1)
    zm, pm, gm = z.copy(), p0.copy(), g0.copy()
    zp, pp, gp = z.copy(), p0.copy(), g0.copy()
    prop, lw, rho = z.copy(), np.zeros(c), p0.copy()
    prop_at = np.zeros(c, np.int64)
    live = np.ones(c, bool)
    steps = np.zeros(c, np.int32)
    for j in range(MAX_DEPTH):
        fwd = u["dir"][:, j] < 0.5
        f = fwd[:, None]
        zs, ps, gs = np.where(f, zp, zm), np.where(f, pp, pm), np.where(f, gp, gm)
        e = np.where(fwd, eps, -eps)[:, None]
        ok = live.copy()
        s_lw = np.full(c, -np.inf)
        s_prop = zs.copy()
        s_at = np.where(fwd, hi, lo)
        S = np.zeros((c, d))
        ck_p = np.empty((j + 1, c, d))
        ck_S = np.empty((j + 1, c, d))
        for n in range(2 ** j):
            p_ = ps - 0.5 * e * gs
            z1 = zs + e * minv * p_
            g1 = grad(z1)
            p_ = p_ - 0.5 * e * g1
            step = ok[:, None]
            zs, ps, gs = np.where(step, z1, zs), np.where(step, p_, ps), np.where(step, g1, gs)
            steps += ok
            at = np.where(fwd, hi + 1 + n, lo - 1 - n)
            dh = pot(zs) + 0.5 * (minv * ps * ps).sum(1) - h0
            bad = ok & (~np.isfinite(dh) | (np.nan_to_num(dh, nan=0.0, posinf=0.0) > 1000))
            ok &= ~bad
            traj[rows, W + at] = np.where(ok[:, None], zs[:, :2], np.nan)
            leaf = np.where(np.isfinite(dh), -dh, -np.inf)
            grown = np.logaddexp(s_lw, leaf)
            take = ok & (np.log(u["leaf"][:, 2 ** j - 1 + n]) < leaf - grown)
            s_prop = np.where(take[:, None], zs, s_prop)
            s_at = np.where(take, at, s_at)
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
        a = np.where(fwd, hi + 1, lo - 2 ** j)
        b = np.where(fwd, hi + 2 ** j, lo - 1)
        wipe = live & ~merge
        span = wipe[:, None] & (cols[None, :] >= (W + a)[:, None]) & (cols[None, :] <= (W + b)[:, None])
        traj[span] = np.nan
        mf, mb = (merge & fwd)[:, None], (merge & ~fwd)[:, None]
        zp, pp, gp = np.where(mf, zs, zp), np.where(mf, ps, pp), np.where(mf, gs, gp)
        zm, pm, gm = np.where(mb, zs, zm), np.where(mb, ps, pm), np.where(mb, gs, gm)
        hi = np.where(merge & fwd, hi + 2 ** j, hi)
        lo = np.where(merge & ~fwd, lo - 2 ** j, lo)
        take = merge & (np.log(u["merge"][:, j]) < s_lw - lw)
        prop = np.where(take[:, None], s_prop, prop)
        prop_at = np.where(take, s_at, prop_at)
        lw = np.where(merge, np.logaddexp(lw, s_lw), lw)
        rho = np.where(merge[:, None], rho + S, rho)
        live = merge & ((rho * minv * pm).sum(1) > 0) & ((rho * minv * pp).sum(1) > 0)
        if not live.any():
            break
    return prop, steps, traj, lo, hi, prop_at


def first_turn(x):
    """Steps from the start to a coordinate's first turn, forward and backward, W + 1 where it never turns.

    x is (c, 2W+1) along the trajectory, start at column W. Forward, the turn
    is the first step whose increment has the opposite sign to the first
    one; backward the same, walking the other way.
    """
    inc = np.diff(x, axis=1)
    out = []
    for seq in (inc[:, W:], -inc[:, :W][:, ::-1]):
        flip = np.sign(seq[:, 1:]) == -np.sign(seq[:, :1])
        out.append(np.where(flip.any(1), flip.argmax(1) + 1, W + 1))
    return out


def read(z0, traj, lo, hi, at):
    """Per chain and per coordinate: the jump at the draw, the furthest the tree went, and whether the draw is past the turn."""
    c = z0.shape[0]
    rec = {"len": hi - lo, "at": at}
    for k, name in COORDS:
        x = traj[:, :, k]
        dx = np.abs(x - z0[:, k:k + 1])
        tf, tb = first_turn(x)
        side = np.where(at >= 0, tf, tb)
        reach = np.where(at >= 0, hi, -lo)
        rec[name] = {"jump": x[np.arange(c), W + at] - z0[:, k], "far": np.nanmax(dx, 1),
                     "turned": side <= reach, "past": np.abs(at) > side, "turn": side}
    return rec


def check(z, minv):
    """The traced tree is the tree: same draws and steps as day 1's nuts_crn on the same numbers."""
    u = p1.noise(2290, 0, z.shape[0], z.shape[1])
    a, _, sa, _ = p1.nuts_crn(z, EPS, minv, u)
    b, sb, traj, lo, hi, at = nuts_traced(z, EPS, minv, u)
    on_tree = np.isfinite(traj[np.arange(z.shape[0]), W + at, 1]).mean()
    return np.abs(a - b).max(), np.abs(sa - sb).max(), on_tree, np.abs(traj[np.arange(z.shape[0]), W + at] - b[:, :2]).max()


def run(z, truth, minv, seed):
    """BURN transitions then L kept, recording the tree on every kept one. Ranks for the band as in day 1."""
    r = z.shape[0]
    below = np.zeros((r, 4), np.int32)
    kept = np.empty((L, LAG_CHAINS, 3))
    acc = {"start": [], "len": [], "at": []}
    for _, name in COORDS:
        for key in ("jump", "far", "turned", "past", "turn"):
            acc[name, key] = []
    for t in range(BURN + L):
        u = p1.noise(seed, t, r, z.shape[1])
        if t < BURN:
            z = p1.nuts_crn(z, EPS, minv, u)[0]
            continue
        z0 = z
        z, _, traj, lo, hi, at = nuts_traced(z, EPS, minv, u)
        rec = read(z0, traj, lo, hi, at)
        acc["start"].append(z0[:, 1])
        acc["len"].append(rec["len"])
        acc["at"].append(rec["at"])
        for _, name in COORDS:
            for key in ("jump", "far", "turned", "past", "turn"):
                acc[name, key].append(rec[name][key])
        p = params("noncentred", z)
        kept[t - BURN] = p[:LAG_CHAINS][:, [0, 1, 3]]
        below += p < truth
    return {k: np.stack(v, 1) for k, v in acc.items()}, below, kept


def report(lab, acc, below, kept, stats, crits):
    nmax = SIZES[-1]
    rr = below.reshape(STUDIES, nmax, 4)
    hj, hl = p1.hits(rr, stats, crits)
    lag = m1.lag1(kept)
    print(f"\n{lab}: joint {hj.mean():.3f}  log tau {hl.mean():.3f}  steps {acc['len'].mean():.1f}  "
          f"lag-1 mu {lag[0]:+.2f} log tau {lag[1]:+.2f}")
    for _, name in COORDS:
        a = {k: acc[name, k] for k in ("jump", "far", "turned", "past", "turn")}
        t = a["turn"][a["turned"]]
        print(f"  {name:7s} turns inside the tree {a['turned'].mean():.3f}, median steps to it {np.median(t):5.1f}, "
              f"draw past it {a['past'].mean():.3f}; |jump| {np.abs(a['jump']).mean():.3f} of furthest "
              f"{a['far'].mean():.3f} ({np.abs(a['jump']).mean() / a['far'].mean():.2f})")
    s = acc["start"]
    lt = {k: acc["log tau", k] for k in ("jump", "far", "turned", "past", "turn")}
    print(f"  by log tau at the start:   share  steps  turned  to turn  past   |jump|  furthest  jump/far  mean jump")
    for lo_, hi_ in zip(BINS[:-1], BINS[1:]):
        m = (s >= lo_) & (s < hi_)
        if m.sum() < 100:
            continue
        tt = lt["turn"][m & lt["turned"]]
        print(f"   [{lo_:+5.1f}, {hi_:+5.1f})            {m.mean():.3f}  {acc['len'][m].mean():5.1f}  "
              f"{lt['turned'][m].mean():.3f}   {np.median(tt) if tt.size else np.nan:5.1f}   {lt['past'][m].mean():.3f}  "
              f"{np.abs(lt['jump'][m]).mean():.3f}   {lt['far'][m].mean():.3f}     {np.abs(lt['jump'][m]).mean() / lt['far'][m].mean():.2f}"
              f"     {lt['jump'][m].mean():+.3f}")
    # does a study's log tau verdict follow how often its draws land past the turn
    per = lt["past"].reshape(STUDIES, nmax, -1).mean((1, 2))
    rat = (np.abs(lt["jump"]) / np.maximum(lt["far"], 1e-12)).reshape(STUDIES, nmax, -1).mean((1, 2))
    if hl.any() and (~hl).any():
        print(f"  studies rejected on log tau ({hl.sum()}) / passed ({(~hl).sum()}): draw past the turn "
              f"{per[hl].mean():.3f} / {per[~hl].mean():.3f}, jump over furthest {rat[hl].mean():.3f} / {rat[~hl].mean():.3f}")
    # past the turn and the next transition: does a draw past the turn come back toward where it started
    past = lt["past"][:, :-1]
    nxt = lt["jump"][:, 1:] * np.sign(lt["jump"][:, :-1])
    print(f"  next jump along this one's direction: after a draw past the turn {nxt[past].mean():+.3f}, "
          f"before it {nxt[~past].mean():+.3f}")
    return hj.mean(), hl.mean()


def main():
    t0 = time.time()
    np.seterr(all="ignore")
    nmax = SIZES[-1]
    r = STUDIES * nmax
    s, _ = d2.shape_stats(nmax)
    stats = {k: s[k] for k in KEYS}
    crits = {k: v for k, v in d2.critical_values(dict(s, n=nmax)).items() if k in KEYS}

    truth, y = draw_studies(r)
    es.Y = y
    z, minv, _ = m1.warmup(r)
    print(f"warmed {r} chains ({time.time() - t0:.0f} s)")
    gz, gs, on, gt = check(z, minv)
    print(f"check: traced against day 1's tree, largest gap in the draw {gz:.1e}, in steps {gs}, "
          f"draw on the kept trajectory {on:.3f}, trajectory at the draw against the draw {gt:.1e}")

    masses = {"chain": minv, "pooled": p1.pooled(minv), "unit": np.ones_like(minv)}
    print(f"\nwhere the tree stops: non-centred NUTS at eps {EPS}, max depth {MAX_DEPTH}, {STUDIES} x {nmax}, "
          f"{BURN} burn then {L} transitions traced, one warmed state")
    for lab, kind, seed in RUNS:
        acc, below, kept = run(z.copy(), truth, masses[kind], seed)
        report(lab, acc, below, kept, stats, crits)
        print(f"  ({time.time() - t0:.0f} s)", flush=True)
    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
