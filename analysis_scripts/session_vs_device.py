"""July vs August: which chain parameters are DEVICE-fixed and which are SESSION?

THE QUESTION. A fingerprinting dataset is only valid if the features it carries
belong to the radio rather than to the afternoon it was captured. We have two
independent captures of the same 23 radios -- originals 2026-07-22..08-03
(radio_characterisation.json) and a re-capture 2026-08-18..21
(generalisation_work/repeat_log.json) -- so every run-level parameter can be
decomposed into device, session and run-to-run parts.

WHY IT MATTERS NOW. Single-session discriminability is NOT evidence of device
identity: with one capture per radio the "between-radio" spread is device PLUS
session state (cabling, position, temperature), and the two are not separable.
The bottom-up session reports that our fitted ISI taps carry 0.0% device
variance across sessions while reading 96-99% device WITHIN a session. This
script is the independent check of that class of claim, on the measured
parameters; the fitted blocks (taps, PA) need a refit and are handled separately.

STATISTIC. For each parameter and config: m[radio, session] = mean over runs.
    device  = var over radios of the per-radio session-mean
    session = mean over radios of the within-radio between-session var
    device share = device / (device + session)
    replication r = corr(m[:, july], m[:, august]) over radios
A device-fixed parameter has share -> 1 and r -> +1. A session-dominated one has
share -> 0 and r -> 0. Permutation null: shuffle radio labels independently
within each session, which destroys any device pairing while preserving both
marginal distributions.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from datetime import datetime
import os as _os

import numpy as np

BASE = Path(__file__).resolve().parent.parent
ORIG = BASE / "radio_characterisation.json"
REP = BASE / "repeat_log.json"
NPERM = 2000
CONFIGS = ["g77_433", "g77_915", "g77_2400", "g89_433", "g89_915", "g89_2400"]

# fig2_chain block -> parameters measured per run. Only fields present in BOTH
# logs are usable; the repeat log carries 12 extra (am_am_*, scd_*) that the
# originals do not, so those cannot be compared across sessions at all.
PARAMS = [
    ("CFO",            "cfo_ppm",               "ppm"),
    ("Sampling clock", "clock_mismatch_ppm",    "ppm"),
    ("Sampling clock", "clock_phase_mean_samp", "samp"),
    ("IQ imbalance",   "iq_amp_db",             "dB"),
    ("IQ imbalance",   "iq_phase_deg",          "deg"),
    ("TX leakage",     "lo_leakage_dbc",        "dBc"),
    ("RX DC",          "rx_dc_frac",            "-"),
    ("AWGN",           "snr_mean_db",           "dB"),
    ("Phase noise",    "phase_var_deg",         "deg"),
    ("Phase noise",    "phase_dev_kurtosis",    "-"),
    ("PA (proxy)",     "amp_var_pct",           "%"),
    ("PA (proxy)",     "amp_dev_kurtosis",      "-"),
    ("Setup",          "tx_temp_c",             "C"),
    ("Setup",          "rx_temp_c",             "C"),
]


# ── which radio/sessions may take part in the cross-day comparison ──────────
#
# 1. BAD CARRIER. Four radios were recorded before the campaign settled on its
#    carriers, at 2400.0/433.0 instead of 2450.0/433.92, and apparently without
#    a GPSDO. Their July capture is not a valid baseline for anything, and two
#    of them (30EAE54 at 28 d, 30ECB6B at 27 d) would otherwise sail through the
#    time filter below on invalid data.
#
# 2. TWO LEAKED AUGUST SESSIONS. 171025 (30EAE27) and 152456 (30ECB6B) are dated
#    2026-08-18 and live in D:/repeat -- they are August captures sitting inside
#    the July log. Both belong to radios already excluded by rule 1.
BAD_JULY_SESSIONS = {"172217",   # 30EAE27, 2400.0/433.0
                     "111617",   # 30EAE54, 2400.0/433.0
                     "181829",   # 30ECB6B, 2400.0/433.0
                     "193436",   # 30ECBAD, 2400.0/433.0
                     "171025",   # 30EAE27, actually 2026-08-18
                     "152456"}   # 30ECB6B, actually 2026-08-18

# 3. MINIMUM SEPARATION. A "cross-day" result means nothing if the two captures
#    are days apart. 20 days keeps 30BF779 (exactly 20), which every fitted-block
#    figure in the paper uses; the next radios down are the 3 August batch at
#    15-18 days. NOTE that this excludes one capture DAY wholesale rather than a
#    random subset, and ISI taps were separately shown to track capture day.
MIN_GAP_DAYS = int(_os.environ.get("SG_MIN_GAP_DAYS", "20"))


# CAPTURE DATES, BAKED IN. The originals live in the sweep DIRECTORY NAMES on
# the capture drive (the logs carry no timestamp), which is not shipped with
# this repository -- so the dates are recorded here instead. Format is
#   radio: (july capture, august re-capture)
# and the four bad-carrier July sessions are already excluded, which is why
# 30EAE27 and 30ECB6B are absent: neither has a usable July baseline.
CAPTURE_DATES = {
    "30BF779": ("20260729", "20260818"),
    "30BF795": ("20260803", "20260818"),
    "30BF796": ("20260728", "20260818"),
    "30BF7A4": ("20260724", "20260819"),
    "30BF7AB": ("20260724", "20260819"),
    "30BF7B6": ("20260724", "20260821"),
    "30BF7BE": ("20260728", "20260819"),
    "30BF7C1": ("20260803", "20260819"),
    "30EAE34": ("20260803", "20260819"),
    "30EAE54": ("20260818", "20260820"),
    "30EAE76": ("20260727", "20260821"),
    "30ECB66": ("20260727", "20260820"),
    "30ECB67": ("20260727", "20260821"),
    "30ECB71": ("20260803", "20260820"),
    "30ECB80": ("20260724", "20260819"),
    "30ECB81": ("20260724", "20260820"),
    "30ECB84": ("20260803", "20260821"),
    "30ECBAD": ("20260819", "20260820"),
    "30ECBBC": ("20260724", "20260821"),
    "30ECBC5": ("20260727", "20260820"),
}


def capture_gaps():
    """{radio: days between its July capture and its August re-capture}."""
    return {r: (datetime.strptime(b, "%Y%m%d")
                - datetime.strptime(a, "%Y%m%d")).days
            for r, (a, b) in CAPTURE_DATES.items()}


def load():
    """{(radio, cfg, session_label): {param: [per-run values]}}"""
    out = defaultdict(lambda: defaultdict(list))
    pat = re.compile(r"^TX([0-9A-Za-z]+)_RXBB60_(\d+)/g(\d+)_(\d+)$")
    for v in json.loads(ORIG.read_text()).values():
        if not isinstance(v, dict):
            continue
        m = pat.match(str(v.get("session", "")))
        if not m:
            continue
        if m.group(2) in BAD_JULY_SESSIONS:
            continue
        key = (m.group(1), f"g{m.group(3)}_{m.group(4)}", "july")
        for _, p, _u in PARAMS:
            val = (v.get("cfo_hz") / (v["fc_hz"] * 1e-6)
                   if p == "cfo_ppm" and v.get("fc_hz") else v.get(p))
            if val is not None and np.isfinite(val):
                out[key][p].append(float(val))
    patr = re.compile(r"^TX([0-9A-Za-z]+)_RXBB60_REPEAT/(g\d+_\d+)/run_\d+$")
    for k, v in json.loads(REP.read_text()).items():
        m = patr.match(k)
        if not m or not isinstance(v, dict):
            continue
        key = (m.group(1), m.group(2), "august")
        for _, p, _u in PARAMS:
            val = (v.get("cfo_hz") / (v["fc_hz"] * 1e-6)
                   if p == "cfo_ppm" and v.get("fc_hz") else v.get(p))
            if val is not None and np.isfinite(val):
                out[key][p].append(float(val))

    # Drop BOTH sessions of any radio whose two captures sit too close together,
    # so decompose() never sees a half-pair.
    gaps = capture_gaps()
    if gaps:
        too_close = {r for r, g in gaps.items() if g < MIN_GAP_DAYS}
        absent = {r for r, _c, _s in out} - set(gaps)
        for key in [k for k in out if k[0] in (too_close | absent)]:
            del out[key]
        kept = sorted({k[0] for k in out})
        print(f"  cross-day filter: {len(kept)} radios at >= {MIN_GAP_DAYS} days"
              f"   (dropped {len(too_close)} too close, {len(absent)} unpaired)")
    return out


def decompose(D, cfg, param, rng):
    rads = sorted({r for (r, c, s) in D if c == cfg})
    A, B, W = [], [], []
    for r in rads:
        a = D.get((r, cfg, "july"), {}).get(param, [])
        b = D.get((r, cfg, "august"), {}).get(param, [])
        if len(a) < 2 or len(b) < 2:
            continue
        A.append(np.mean(a)); B.append(np.mean(b))
        W.append(0.5 * (np.var(a, ddof=1) + np.var(b, ddof=1)))
    if len(A) < 4:
        return None
    A, B = np.asarray(A), np.asarray(B)
    # session variance: with two observations per radio, 0.5*(a-b)^2 IS the
    # unbiased two-sample variance (sum of squared deviations from their mean,
    # over dof 1).
    ses = float(np.mean(0.5 * (A - B) ** 2))
    # THE RAW DEVICE TERM IS BIASED UP BY HALF THE SESSION VARIANCE. The
    # per-radio session-MEAN still carries session noise: var(0.5(A+B)) = d +
    # S/2. So the raw ratio has a null of 0.5/1.5 = 1/3, NOT 0 -- confirmed by
    # permutation, whose null median lands at 0.318-0.344 for every parameter.
    # Reading a raw 0.39 as "some device content" is wrong; it is the null.
    # Same class of error as the 1/sqrt(n) null in the discriminability work.
    dev_biased = float(np.var(0.5 * (A + B), ddof=1))
    dev = dev_biased - ses / 2.0                  # unbiased device variance
    share_raw = dev_biased / (dev_biased + ses) if (dev_biased + ses) > 0 else np.nan
    # TRUNCATE AT ZERO, standard for variance components: the unbiased estimate
    # of a variance that is truly zero is negative half the time. rx_temp_c --
    # the shared receiver, which CANNOT carry per-device information -- lands at
    # -0.72 uncorrected, which is the estimator behaving correctly on a true
    # zero, not a meaningful negative. The untruncated value stays in share_raw
    # and in dev for anyone who needs it.
    share_untrunc = dev / (dev + ses) if (dev + ses) > 0 else np.nan
    # Display value truncates at zero (standard for variance components: the
    # unbiased estimate of a true zero is negative half the time). The P-VALUE
    # MUST USE THE UNTRUNCATED STATISTIC -- truncating the observed value while
    # the permutation null stays untruncated lets the observed beat a null that
    # is half negative, and rx_temp_c (the SHARED receiver, which cannot carry
    # device information) flipped from 0/6 to a spurious 6/6 "device" when I
    # made exactly that mistake.
    share = float(max(share_untrunc, 0.0)) if np.isfinite(share_untrunc) else np.nan
    r = float(np.corrcoef(A, B)[0, 1]) if A.std() > 0 and B.std() > 0 else np.nan
    # permutation: break the device pairing, keep both marginals
    null = np.empty(NPERM)
    for i in range(NPERM):
        a2, b2 = rng.permutation(A), rng.permutation(B)
        s2 = np.mean(0.5 * (a2 - b2) ** 2)
        d2 = np.var(0.5 * (a2 + b2), ddof=1) - s2 / 2.0
        null[i] = d2 / (d2 + s2) if (d2 + s2) > 0 else np.nan
    pval = float(np.mean(null >= share_untrunc))
    return dict(n=len(A), device=dev, session=ses, share=share,
                share_raw=share_raw, r=r, p=pval,
                run=float(np.mean(W)), july=A, august=B, rads=rads[:len(A)])


if __name__ == "__main__":
    D = load()
    rng = np.random.default_rng(0)
    nj = len({r for (r, c, s) in D if s == "july"})
    na = len({r for (r, c, s) in D if s == "august"})
    print(f"\n  july: {nj} radios   august: {na} radios   "
          f"{NPERM} permutations\n")
    res = {}
    for cfg in CONFIGS:
        print(f"  {cfg}")
        print(f"    {'block':<16}{'parameter':<22}{'share':>8}{'r':>8}"
              f"{'p':>8}{'dev sd':>11}{'ses sd':>11}{'run sd':>11}")
        for blk, p, unit in PARAMS:
            out = decompose(D, cfg, p, rng)
            if out is None:
                continue
            res[(cfg, p)] = out
            flag = "" if out["p"] < 0.05 else "   <- NOT device"
            print(f"    {blk:<16}{p:<22}{out['share']:>8.3f}{out['r']:>+8.2f}"
                  f"{out['p']:>8.3f}{np.sqrt(out['device']):>11.4g}"
                  f"{np.sqrt(out['session']):>11.4g}"
                  f"{np.sqrt(out['run']):>11.4g}{flag}")
        print()
    np.save(BASE / "scratchpad" / "session_vs_device.npy",
            np.array([{k: {kk: vv for kk, vv in v.items()} for k, v in res.items()}],
                     dtype=object))
    print(f"  saved {len(res)} (config, parameter) results")
