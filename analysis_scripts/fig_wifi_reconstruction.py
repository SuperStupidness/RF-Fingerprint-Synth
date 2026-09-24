#!/usr/bin/env python3
"""802.11g OFDM reconstruction against the real capture: PSD, time domain and
constellation, each beside its error.

    python analysis_scripts/fig_wifi_reconstruction.py

Writes figures/27_wifi_recon.png / .pdf (the name the paper cites).

Reads data/wifi_recon_30BF795_89_2400.npz, packaged here because the
captures it was extracted from are not shipped. It carries the real burst
average and the reconstruction through the full impairment chain, both at
40 MS/s, plus the scalars the panels annotate:

    ts_real, ts_recon   real RX burst average, and its reconstruction
    fs, guard           sample rate, and the edge guard the fit excluded
    nkept               bursts in the coherent average (sets the noise floor)
    floor               averaged noise floor, 10^(-SNR/20)/sqrt(nkept)
    aa_bw               receiver anti-alias bandwidth, drawn as +/-aa_bw/2

Demodulation uses the project's VALIDATED receiver, wifi_ofdm.py (vendored
beside this script): L-LTF detect, LTS channel estimate, zero-forcing
equalise, per-symbol pilot common-phase tracking. Cross-checked against the
MATLAB WLAN Toolbox reference to ~1e-8; the TX itself demodulates to 0.000%
EVM through it, which is the sanity check.

READING IT: with pilot tracking in place the burst-repetitive phase transient
is removed by the receiver, so it does not appear in the constellation. It is
a waveform-fidelity term, not something a receiver would see.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "analysis_scripts"))
from wifi_ofdm import demod_wifi, _DATA_K            # noqa: E402

NPZ = BASE / "data" / "wifi_recon_30BF795_89_2400.npz"
OUT = BASE / "figures" / "27_wifi_recon"
TAG = "30BF795 / wifi_89_2400"
WINDOW = 600            # samples shown in the time-domain panels
MUTED = "#5A636E"       # slate, for the anti-alias edge markers


def psd(v, fs):
    f, P = scipy.signal.welch(v, fs=fs, nperseg=2048, noverlap=1024,
                              return_onesided=False)
    return np.fft.fftshift(f) / 1e6, np.fft.fftshift(P)


def receive(v40):
    """The validated 802.11a/g receiver, on a 40 MS/s burst (decimated x2)."""
    r = demod_wifi(v40[::2])
    if not r["found"]:
        raise SystemExit("receiver did not detect a burst")
    eq = r["eq"].reshape(r["n_symbols"], len(_DATA_K))
    ref = (np.sign(eq.real) + 1j * np.sign(eq.imag)) / np.sqrt(2)
    return eq, ref, r["evm_pct"]


def main():
    d = np.load(NPZ)
    fs, g = float(d["fs"]), int(d["guard"])
    nk, floor = int(d["nkept"]), float(d["floor"])
    edge = float(d["aa_bw"]) / 2e6
    real, recon = d["ts_real"], d["ts_recon"]
    sl = slice(g, -g)

    f, Pr = psd(real[sl], fs)
    _, Pc = psd(recon[sl], fs)
    occ = np.abs(f) <= 9.0
    ndb = lambda P: 10 * np.log10(P / P[occ].mean())

    Ey, Ry, evm_y = receive(real)
    Ec, Rc, evm_c = receive(recon)
    evm_sc = lambda E, R: 100 * np.sqrt(
        np.mean(np.abs(E - R) ** 2, 0) / np.mean(np.abs(R) ** 2))

    fig, ax = plt.subplots(3, 2, figsize=(7.6, 7.5))

    # [0, 0] PSD - full burst
    ax[0, 0].plot(f, ndb(Pr), lw=1.0, color="C0")
    ax[0, 0].plot(f, ndb(Pc), lw=1.0, color="C1", alpha=0.85)
    ax[0, 0].set_ylim(-140, 15)
    ax[0, 0].set_xlabel("MHz")
    ax[0, 0].set_ylabel("PSD (dB re in-band)")
    ax[0, 0].set_title("PSD - full burst")
    ax[0, 0].grid(alpha=0.3)

    # [0, 1] PSD error. The vertical pair is the receiver anti-alias edge;
    # slate dash-dot so it reads apart from the black dashed mean error.
    ax[0, 1].plot(f, ndb(Pc) - ndb(Pr), lw=0.8, color="C3")
    ax[0, 1].axhline(0, color="k", lw=0.7)
    for s in (1, -1):
        ax[0, 1].axvline(s * edge, color=MUTED, ls="-.", lw=0.9)
    ax[0, 1].set_ylim(-20, 20)
    ax[0, 1].set_xlabel("MHz")
    ax[0, 1].set_ylabel("recon - real (dB)")
    ax[0, 1].set_title("PSD error")
    ax[0, 1].grid(alpha=0.3)

    # [1, 0] Time domain
    i0 = len(real) // 2
    z = slice(i0, i0 + WINDOW)
    t = np.arange(WINDOW) / fs * 1e6
    ax[1, 0].plot(t, real[z].real, lw=0.9, color="C0")
    ax[1, 0].plot(t, recon[z].real, lw=0.9, color="C1", alpha=0.85)
    ax[1, 0].set_xlabel("time in burst (us)")
    ax[1, 0].set_ylabel("in-phase")
    ax[1, 0].set_title(f"time domain, {WINDOW} samples")
    ax[1, 0].grid(alpha=0.3)

    # [1, 1] Time-domain error
    err = np.abs(real - recon)
    ax[1, 1].plot(t, err[z], lw=0.8, color="C3")
    ax[1, 1].axhline(err[sl].mean(), color="k", ls="--", lw=0.7)
    ax[1, 1].axhline(floor, color="C2", ls=":", lw=1.0)
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylim(1e-3, 0.5)
    ax[1, 1].set_xlabel("time in burst (us)")
    ax[1, 1].set_ylabel("|recon - real|")
    ax[1, 1].set_title("time-domain error")
    ax[1, 1].grid(alpha=0.3)

    # [2, 0] Constellation
    for E, c in ((Ey, "C0"), (Ec, "C1")):
        v = E.ravel()[::3]
        ax[2, 0].scatter(v.real, v.imag, s=1, alpha=0.05, color=c)
    ax[2, 0].set_aspect("equal")
    ax[2, 0].set_xlim(-1.8, 1.8)
    ax[2, 0].set_ylim(-1.8, 1.8)
    ax[2, 0].set_xlabel("I")
    ax[2, 0].set_ylabel("Q")
    ax[2, 0].grid(alpha=0.3)
    ax[2, 0].set_title("constellation, validated RX, pilot-tracked\n"
                       f"real {evm_y:.2f}%   recon {evm_c:.2f}%")

    # [2, 1] EVM per subcarrier
    ax[2, 1].plot(_DATA_K, evm_sc(Ey, Ry), "o-", ms=3, lw=0.9, color="C0")
    ax[2, 1].plot(_DATA_K, evm_sc(Ec, Rc), "s-", ms=3, lw=0.9, color="C1",
                  alpha=0.85)
    ax[2, 1].set_xlabel("subcarrier index")
    ax[2, 1].set_ylabel("EVM (%)")
    ax[2, 1].set_title("constellation error per subcarrier")
    ax[2, 1].grid(alpha=0.3)

    # Everything that carries text is 10 pt; the suptitle keeps its hierarchy
    # through weight rather than size.
    for a in ax.flat:
        a.tick_params(axis="both", labelsize=10)
        a.xaxis.label.set_size(10)
        a.yaxis.label.set_size(10)
        a.title.set_size(10)

    fig.suptitle(f"real RX vs reconstruction - 802.11g OFDM, {TAG}",
                 fontsize=10, fontweight="bold", y=0.992)

    # 6-element single horizontal legend across the top
    fig.legend(
        handles=[
            Line2D([], [], color="C0", lw=1.6, label="real RX"),
            Line2D([], [], color="C1", lw=1.6, label="reconstruction"),
            Line2D([], [], color="C3", lw=1.6, label="error"),
            Line2D([], [], color="k", lw=1.1, ls="--", label="mean error"),
            Line2D([], [], color="C2", lw=1.3, ls=":",
                   label=f"{nk}-burst noise floor"),
            Line2D([], [], color=MUTED, lw=1.1, ls="-.",
                   label="anti-alias edge"),
        ],
        loc="upper center", ncol=6, fontsize=10.0, frameon=False,
        bbox_to_anchor=(0.5, 0.966), handlelength=1.5,
        handletextpad=0.35, columnspacing=1.0,
    )
    plt.tight_layout(rect=[0.015, 0.012, 0.985, 0.945], h_pad=0.9, w_pad=1.0)

    OUT.parent.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=150, facecolor="white",
                    bbox_inches="tight")
    print(f"wrote {OUT}.png / .pdf")
    print(f"   EVM real {evm_y:.2f}%   recon {evm_c:.2f}%")


if __name__ == "__main__":
    main()
