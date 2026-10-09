"""Static Matplotlib dashboard (for the report / slides) + decision-tree figure."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import plot_tree
from config import CAUSE_COLORS
from attribute import FEATURES, FEATURE_LABELS

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def _grid(z, col, R, C):
    g = np.full((R, C), np.nan, dtype=object if z[col].dtype == object else float)
    for r in z.itertuples():
        g[r.row, r.col] = getattr(r, col)
    return g


def build_dashboard(z, typ, clf, metrics, area_name, out_path, landmarks=(), subtitle=""):
    R, C = z.row.max() + 1, z.col.max() + 1
    ext = [z.lon.min(), z.lon.max(), z.lat.min(), z.lat.max()]
    fig = plt.figure(figsize=(17, 10), facecolor="white")
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.42)
    fig.suptitle(f"Urban Heat Anomaly Diagnosis & Action — {area_name}", fontsize=16, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.935, "DETECT → EXPLAIN → PRESCRIBE  ·  typical day  ·  " + subtitle,
             color="#c0392b", fontsize=10)

    def labels(ax):
        if True:
            for name, la, lo, *_ in list(landmarks)[:14]:
                ax.plot(lo, la, "o", ms=3, color="black")
                ax.annotate(name.split("(")[0].strip(), (lo, la), fontsize=6.5, xytext=(3, 3),
                            textcoords="offset points", color="black")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    # (a) temperature at 3 PM
    ax = fig.add_subplot(gs[0, 0])
    t15 = typ[typ.hour == 15].set_index("zone_id").temp
    g = _grid(z.assign(t15=z.zone_id.map(t15)), "t15", R, C).astype(float)
    im = ax.imshow(g, extent=ext, cmap="RdYlBu_r", origin="upper", aspect="auto")
    fig.colorbar(im, ax=ax, label="°C", shrink=0.85)
    ax.set_title("DETECT · Heat map at 3 PM", loc="left", fontweight="bold"); labels(ax)

    # (b) severity
    ax = fig.add_subplot(gs[0, 1])
    sev = {"Normal": 0, "Watch": 1, "Hot": 2}
    g = _grid(z.assign(s=z.severity.map(sev)), "s", R, C).astype(float)
    ax.imshow(g, extent=ext, cmap=ListedColormap(["#deebf7", "#fd8d3c", "#e31a1c"]), vmin=0, vmax=2,
              origin="upper", aspect="auto")
    for k, c in [("Normal", "#deebf7"), ("Watch", "#fd8d3c"), ("Hot", "#e31a1c")]:
        ax.scatter([], [], marker="s", color=c, label=f"{k} ({(z.severity == k).sum()})")
    ax.legend(loc="lower right", fontsize=7.5, frameon=True)
    ax.set_title("DETECT · Severity (peak ΔT vs city median)", loc="left", fontweight="bold"); labels(ax)

    # (c) cause map
    ax = fig.add_subplot(gs[0, 2])
    causes = list(CAUSE_COLORS)
    g = _grid(z.assign(ci=z.cause.map(causes.index)), "ci", R, C).astype(float)
    ax.imshow(g, extent=ext, cmap=ListedColormap([CAUSE_COLORS[c] for c in causes]),
              vmin=-0.5, vmax=len(causes) - 0.5, origin="upper", aspect="auto", alpha=0.9)
    pc = z[z.plant_candidate]
    ax.scatter(pc.lon, pc.lat, s=14, marker="^", color="#2e7d32", edgecolor="white", lw=0.5, label="Plant site")
    for c in causes[:-1]:
        n = (z.cause == c).sum()
        if n: ax.scatter([], [], marker="s", color=CAUSE_COLORS[c], label=f"{c} ({n})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
    ax.set_title("EXPLAIN · Dominant cause per zone", loc="left", fontweight="bold"); labels(ax)

    # (d) diurnal dT by cause
    ax = fig.add_subplot(gs[1, 0])
    tt = typ.merge(z[["zone_id", "cause"]], on="zone_id")
    for c, grp in tt.groupby("cause"):
        prof = grp.groupby("hour").dT.mean()
        ax.plot(prof.index, prof.values, lw=2.2 if c != "None" else 1.2,
                color=CAUSE_COLORS[c], label=c if c != "None" else "Normal zones")
    ax.axhline(0, color="#999", lw=0.8); ax.axvspan(14, 18, color="#fdd49e", alpha=0.35, lw=0)
    ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("Hour of day"); ax.set_ylabel("Mean ΔT (°C)")
    ax.legend(fontsize=7, ncol=2); ax.set_title("EXPLAIN · When each cause heats the zone", loc="left", fontweight="bold")

    # (e) feature importance
    ax = fig.add_subplot(gs[1, 1])
    imp = sorted(zip(clf.feature_importances_, FEATURES))
    ax.barh([FEATURE_LABELS[f] for _, f in imp], [v for v, _ in imp], color="#c0392b")
    ax.set_xlabel("Gini importance")
    ax.set_title(f"EXPLAIN · Decision Tree drivers (test acc. {metrics['accuracy']:.0%})", loc="left", fontweight="bold")

    # (f) priority action table
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.set_title("PRESCRIBE · Top priority actions", loc="left", fontweight="bold")
    top = z.sort_values("priority_rank").head(9)
    short = {"Surface retention": "Surface", "Low ventilation": "Ventilation", "Activity spike": "Activity",
             "Low green cover": "Low green", "Mixed": "Mixed"}
    cell = [[f"{r.priority_rank}", r.near.split("(")[0].strip()[:20], f"+{r.peak_dT:.1f}",
             short.get(r.cause, r.cause), r.headline] for r in top.itertuples()]
    tb = ax.table(cellText=cell, colLabels=["#", "Area", "ΔT", "Cause", "Fix"],
                  bbox=[0, 0.02, 1.12, 0.95], colWidths=[0.05, 0.31, 0.08, 0.21, 0.35], cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(7.8)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#ddd")
        if i == 0: c.set_facecolor("#2c3e50"); c.get_text().set_color("white"); c.get_text().set_weight("bold")
        elif j == 3: c.get_text().set_color(CAUSE_COLORS.get(top.cause.iloc[i - 1], "black")); c.get_text().set_weight("bold")

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_tree_figure(clf, out_path, depth=3):
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(clf, max_depth=depth, feature_names=[FEATURE_LABELS[f] for f in FEATURES],
              class_names=list(clf.classes_), filled=True, rounded=True, impurity=False,
              proportion=True, fontsize=8, ax=ax)
    ax.set_title(f"Cause classifier — first {depth} levels (full rules in model_rules.txt)", fontsize=13)
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out_path
