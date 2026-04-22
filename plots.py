"""Produce four publication-quality figures from ./output/results.json."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
FIG  = ROOT / "figs"
FIG.mkdir(exist_ok=True)
RES  = json.loads((ROOT / "output" / "results.json").read_text())

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})
BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"
PURPLE, ORANGE = "#9467bd", "#ff7f0e"

I = RES["instance"]["suppliers"]
J = RES["instance"]["customers"]
P = RES["instance"]["products"]
T = RES["instance"]["periods"]
NI, NJ, NP, NT = len(I), len(J), len(P), len(T)

# Pretty labels
LAB_I = {"FAB_KR1": "Fab-KR1", "FAB_KR2": "Fab-KR2", "FAB_TW": "Fab-TW"}
LAB_J = {"CUS_AI": "AI-Hyper", "CUS_SRV": "Srv-OEM",
         "CUS_PH": "Phone-OEM", "CUS_AUTO": "Auto-T1", "CUS_IND": "Industrial"}
I_lbl = [LAB_I[i] for i in I]
J_lbl = [LAB_J[j] for j in J]

# ----------------------------------------------- aggregate shipments i-j-p
X = np.zeros((NI, NJ, NP))
for key, v in RES["MILP"]["x"].items():
    i, j, p, t = key.split("|")
    X[I.index(i), J.index(j), P.index(p)] += v


def fig_allocation():
    fig, axes = plt.subplots(1, NP, figsize=(9.4, 2.7), sharey=True)
    for pi, ax in enumerate(axes):
        data = X[:, :, pi]
        vmax = max(1e-6, data.max())
        im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(P[pi])
        ax.set_xticks(range(NJ)); ax.set_xticklabels(J_lbl, rotation=30, ha="right")
        ax.set_yticks(range(NI)); ax.set_yticklabels(I_lbl)
        ax.grid(False)
        for ii in range(NI):
            for jj in range(NJ):
                val = data[ii, jj]
                if val > 0.5:
                    color = "white" if val > vmax * 0.55 else "black"
                    ax.text(jj, ii, f"{val:.0f}", ha="center", va="center",
                            fontsize=7.5, color=color)
    fig.suptitle("MILP allocation summed across the 4-week horizon  "
                 "(thousand wafer-equivalents)", y=1.04)
    fig.tight_layout()
    fig.savefig(FIG / "allocation.pdf")
    plt.close(fig)


# ----------------------------------------------- shadow-price heatmap
def fig_shadow():
    # duals are negative because equality constraint sign convention;
    # plot magnitudes.
    sh = np.zeros((NI, NP, NT))
    for key, val in RES["LP"]["dual_supply"].items():
        i, p, t = key.split("|")
        sh[I.index(i), P.index(p), T.index(t)] = abs(val)

    fig, axes = plt.subplots(1, NP, figsize=(9.4, 2.4), sharey=True)
    vmax_global = sh.max()
    for pi, ax in enumerate(axes):
        im = ax.imshow(sh[:, pi, :], cmap="OrRd", aspect="auto",
                       vmin=0, vmax=vmax_global)
        ax.set_title(P[pi])
        ax.set_xticks(range(NT)); ax.set_xticklabels(T)
        ax.set_yticks(range(NI)); ax.set_yticklabels(I_lbl)
        ax.grid(False)
        for ii in range(NI):
            for ti in range(NT):
                val = sh[ii, pi, ti]
                if val > 0.03:
                    color = "white" if val > vmax_global * 0.55 else "black"
                    ax.text(ti, ii, f"{val:.1f}", ha="center", va="center",
                            fontsize=7.5, color=color)
    # colorbar to the right of the rightmost axis
    fig.subplots_adjust(right=0.92)
    cax = fig.add_axes([0.935, 0.15, 0.012, 0.72])
    fig.colorbar(im, cax=cax, label=r"$|\mu^\star|$  (\$/kWE)")
    fig.suptitle("LP shadow prices on per-period supply capacity", y=1.02)
    fig.savefig(FIG / "shadow.pdf")
    plt.close(fig)


# ----------------------------------------------- LP vs MILP cost breakdown
def fig_lp_vs_milp():
    labels = ["Transport", "Shortage", "Holding", "Lane-fixed"]
    lp_vals = [RES["LP"][k] for k in
               ("cost_transport", "cost_shortage", "cost_holding", "cost_lane_fixed")]
    mi_vals = [RES["MILP"][k] for k in
               ("cost_transport", "cost_shortage", "cost_holding", "cost_lane_fixed")]
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    b1 = ax.bar(x - w / 2, lp_vals, w, color=BLUE,   label="LP relaxation")
    b2 = ax.bar(x + w / 2, mi_vals, w, color=RED,    label="MILP")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"Cost component  (\$k)")
    zlp = RES["LP"]["objective"]; zmi = RES["MILP"]["objective"]
    ax.set_title(f"LP vs. MILP cost decomposition\n"
                 f"$z^\\star_{{LP}} = {zlp:.0f}$k;   "
                 f"$z^\\star_{{MILP}} = {zmi:.0f}$k  "
                 f"(gap {RES['gap_pct']:.2f}%, {RES['MILP']['open_lanes']} / 15 lanes)",
                 fontsize=9)
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if h > 1:
                ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.0f}",
                        ha="center", va="bottom", fontsize=7.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "lp_vs_milp.pdf")
    plt.close(fig)


# ----------------------------------------------- sensitivity curve
def fig_sensitivity():
    scan = RES["sensitivity"]
    mults = np.array([s["mult"] for s in scan])
    cost  = np.array([s["z"]    for s in scan])
    fill  = np.array([100 * s["fill_ai_hbm"] for s in scan])

    fig, ax1 = plt.subplots(figsize=(5.2, 3.2))
    ax1.plot(mults, cost, color=BLUE, marker="o", lw=1.6,
             label=r"Optimal cost $z^\star$")
    ax1.set_xlabel("HBM3e capacity at Fab-KR1  (× nominal)")
    ax1.set_ylabel(r"$z^\star$  (\$k)"); ax1.yaxis.label.set_color(BLUE)
    ax1.tick_params(axis="y", colors=BLUE)

    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(mults, fill, color=RED, marker="s", ls="--", lw=1.6,
             label="AI-Hyper HBM3e fill (%)")
    ax2.set_ylabel("AI-Hyper HBM3e fill (%)"); ax2.yaxis.label.set_color(RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.spines.right.set_color(RED); ax2.spines.right.set_visible(True)
    ax2.spines.top.set_visible(False)

    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity.pdf")
    plt.close(fig)


def main():
    fig_allocation()
    fig_shadow()
    fig_lp_vs_milp()
    fig_sensitivity()
    print("wrote", [p.name for p in sorted(FIG.iterdir())])


if __name__ == "__main__":
    main()
