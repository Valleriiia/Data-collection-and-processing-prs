import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#  1. Завантаження даних
files = {
    "Природне світло (вікно)": "Natural.csv",
    "Штучне освітлення 1 (лампа)": "Lamp1.csv",
    "Штучне освітлення 2 (настільна лампа)": "Lamp2.csv",
    "Темрява (контроль)": "Dark.csv",
}

COLORS = {
    "Природне світло (вікно)": "#F5A623",
    "Штучне освітлення 1 (лампа)": "#5B9BD5",
    "Штучне освітлення 2 (настільна лампа)": "#70AD47",
    "Темрява (контроль)": "#404040",
}

datasets = {}
for label, fname in files.items():
    df = pd.read_csv(fname)
    df.columns = ["time_s", "illuminance_lx"]
    datasets[label] = df
    print(f"✓ {fname}: {len(df)} вимір(ів)")

#  2. Статистика
def classify_illuminance(lx):
    if lx < 1:        return "Темрява"
    elif lx < 50:     return "Дуже слабке"
    elif lx < 200:    return "Слабке (коридор)"
    elif lx < 500:    return "Нормальне (офіс)"
    elif lx < 2000:   return "Яскраве (читання)"
    elif lx < 10000:  return "Дуже яскраве"
    else:             return "Пряме сонячне"

print("\n  Статистика освітленості (лк)")

stats = {}
for label, df in datasets.items():
    lx = df["illuminance_lx"]
    s = {
        "mean":   lx.mean(),
        "median": lx.median(),
        "std":    lx.std() if len(lx) > 1 else 0.0,
        "min":    lx.min(),
        "max":    lx.max(),
        "n":      len(lx),
        "class":  classify_illuminance(lx.mean()),
    }
    stats[label] = s
    print(f"\n  {label}")
    print(f"    Кількість вимірів : {s['n']}")
    print(f"    Середнє           : {s['mean']:>10.2f} лк")
    print(f"    Медіана           : {s['median']:>10.2f} лк")
    print(f"    Стд. відхилення   : {s['std']:>10.2f} лк")
    print(f"    Мін / Макс        : {s['min']:.2f} / {s['max']:.2f} лк")
    print(f"    Класифікація      : {s['class']}")

labels   = list(datasets.keys())
means    = [stats[l]["mean"] for l in labels]
stds     = [stats[l]["std"]  for l in labels]
colors   = [COLORS[l]        for l in labels]

nat_mean = stats["Природне світло (вікно)"]["mean"]
print("\n  Відсоток від природнього світла")
for label, s in stats.items():
    if label == "Природне світло (вікно)": continue
    pct = s["mean"] / nat_mean * 100 if nat_mean > 0 else 0
    print(f"  {label:<38}: {pct:.2f}%")

#  Візуалізація
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "#FAFAFA",
    "axes.facecolor":    "#FAFAFA",
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.45,
})

fig1, axes = plt.subplots(2, 2, figsize=(12, 9))
fig1.suptitle(
    "Часові ряди освітленості по умовах вимірювання",
    fontsize=14, fontweight="bold", color="#1A1A2E", y=1.01
)
fig1.patch.set_facecolor("#FAFAFA")

for ax, (label, df) in zip(axes.flat, datasets.items()):
    t  = df["time_s"]
    lx = df["illuminance_lx"]
    c  = COLORS[label]
    mean_val = lx.mean()

    ax.fill_between(t, lx, alpha=0.18, color=c)
    ax.plot(t, lx, "o-", color=c, linewidth=2.2,
            markersize=8, markerfacecolor="white",
            markeredgecolor=c, markeredgewidth=2.2, zorder=4)
    ax.axhline(mean_val, color=c, linestyle="--",
               linewidth=1.3, alpha=0.75,
               label=f"Середнє: {mean_val:.1f} лк")

    for _, row in df.iterrows():
        ax.annotate(f"{row.illuminance_lx:.0f} лк",
                    (row.time_s, row.illuminance_lx),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="#444",
                    fontweight="bold")

    ax.set_title(label, fontsize=10, fontweight="bold", color="#1A1A2E", pad=6)
    ax.set_xlabel("Час (с)", fontsize=9)
    ax.set_ylabel("Освітленість (лк)", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.5, loc="lower right")
    ax.set_axisbelow(True)

    y_max = lx.max() * 1.35 if lx.max() > 0 else 10
    ax.set_ylim(bottom=0, top=y_max)
    for thresh, lbl in [(300, "офіс 300"), (500, "норма 500")]:
        if thresh < y_max * 0.9:
            ax.axhline(thresh, color="#CC0000", linewidth=0.7,
                       linestyle=":", alpha=0.5)
            ax.text(t.max() * 0.98 if t.max() > 0 else 1, thresh,
                    lbl, va="bottom", ha="right",
                    fontsize=7, color="#CC0000", alpha=0.7)

fig1.tight_layout()
fig1.savefig("fig1_timeseries.png",
             dpi=150, bbox_inches="tight", facecolor="#FAFAFA")

fig2, ax2 = plt.subplots(figsize=(13, 6))
fig2.patch.set_facecolor("#FAFAFA")

x = np.arange(len(labels))
bars = ax2.bar(x, means, color=colors, width=0.52,
               yerr=stds, capsize=7,
               error_kw={"elinewidth": 1.8, "ecolor": "#555"},
               edgecolor="white", linewidth=1.3, zorder=3)

for bar, val, std in zip(bars, means, stds):
    offset = std + max(means) * 0.012
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + offset,
             f"{val:,.1f} лк",
             ha="center", va="bottom",
             fontsize=9.5, fontweight="bold", color="#222")

for thresh, lbl, clr in [(300, "EN мін. офіс (300 лк)", "#E67E22"),
                          (500, "EN норма офіс (500 лк)", "#C0392B")]:
    ax2.axhline(thresh, color=clr, linewidth=1.1, linestyle="--", alpha=0.75, zorder=2)
    ax2.text(len(labels) - 0.45, thresh * 1.15, lbl,
             color=clr, fontsize=8, alpha=0.85)

ax2.set_yscale("symlog", linthresh=10)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10)
ax2.set_ylabel("Освітленість (лк)  [логарифмічна шкала]", fontsize=11)
ax2.set_title(
    "Порівняння середньої освітленості по умовах\n"
    "(± стандартне відхилення; логарифмічна шкала)",
    fontsize=12, fontweight="bold", color="#1A1A2E", pad=10
)
ax2.set_axisbelow(True)
ax2.text(0.99, 0.01, "* symlog: лінійна біля 0, логарифмічна далі",
         transform=ax2.transAxes, ha="right", va="bottom",
         fontsize=7.5, color="#999", style="italic")

fig2.tight_layout()
fig2.savefig("fig2_comparison_log.png",
             dpi=150, bbox_inches="tight", facecolor="#FAFAFA")

plt.show()