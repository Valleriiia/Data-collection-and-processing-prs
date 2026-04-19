import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geonamescache
from shapely.geometry import Point
from matplotlib.colors import Normalize
import pygadm  # Використовуємо pygadm для отримання офіційних кордонів

warnings.filterwarnings("ignore")
np.random.seed(42)

# 1. Завантаження меж України та сусідів
print("\nЗавантаження офіційних кордонів через pygadm...")

ukraine = pygadm.get_items(name='Ukraine', content_level=0)

neighbors_list = ['Poland', 'Slovakia', 'Hungary', 'Romania', 'Moldova', 'Belarus', 'Russia']
neighbors = pygadm.get_items(name=neighbors_list, content_level=0)

print(f"Кордони завантажено")

# 2. Дані міст (geonamescache + синтетична аналітика)
gc = geonamescache.GeonamesCache()
ua_raw = [
    v for v in gc.get_cities().values()
    if v["countrycode"] == "UA" and v["population"] > 30_000
]

df = pd.DataFrame(ua_raw)[["name", "latitude", "longitude", "population"]]
df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
df["population"] = df["population"].astype(int)
df["energy_GWh"] = (df["population"] / 1000 * 0.75
                    + np.random.normal(0, 30, len(df))).clip(10).astype(int)
df["renewable_pct"] = np.random.uniform(5, 48, len(df)).round(1)
df["size_class"] = pd.cut(
    df["population"],
    bins=[0, 100_000, 300_000, 800_000, 5_000_000],
    labels=["Мале\n(<100k)", "Середнє\n(100–300k)",
            "Велике\n(300k–800k)", "Мегаполіс\n(>800k)"],
)

geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

print(f"\nЗавантажено {len(gdf)} міст України (населення > 30 000)")
print(f"\nПерші 5 рядків:")
print(gdf[["name", "population", "energy_GWh", "renewable_pct"]].head().to_string(index=False))
print(f"\nCRS: {gdf.crs}")

# 3. Геопросторові операції
print("\n  Геопросторові операції")

# Перепроєцювання в UTM 36N
gdf_utm = gdf.to_crs("EPSG:32636")
print(f"\nПерепроєцьовано в EPSG:32636 (UTM 36N, метри)")

# Відстані від Києва
kyiv_geom = gdf_utm[gdf_utm["name"] == "Kyiv"].geometry.values[0]
gdf_utm["dist_kyiv_km"] = (gdf_utm.geometry.distance(kyiv_geom) / 1000).round(1)

print("\nНайближчі до Києва міста:")
print(gdf_utm.nsmallest(6, "dist_kyiv_km")[["name", "dist_kyiv_km"]].to_string(index=False))

print("\nНайвіддаленіші від Києва міста:")
print(gdf_utm.nlargest(5, "dist_kyiv_km")[["name", "dist_kyiv_km"]].to_string(index=False))

# Буферні зони для міст > 300k
large = gdf_utm[gdf_utm["population"] > 300_000].copy()
large["buffer_geom"] = large.geometry.buffer(40_000)
print(f"\nБуферні зони 40 км: {len(large)} великих міст")

# Просторовий запит: міста в буфері Києва
kyiv_buf = large[large["name"] == "Kyiv"]["buffer_geom"].values[0]
in_buf = gdf_utm[gdf_utm.geometry.within(kyiv_buf) & (gdf_utm["name"] != "Kyiv")]
print(f"\nМіст у радіусі 40 км від Києва: {len(in_buf)}")
if len(in_buf):
    print("   " + ", ".join(in_buf["name"].tolist()))

corr = gdf["population"].corr(gdf["energy_GWh"])
print(f"\nКореляція населення та енергоспоживання: {corr:.3f}")
print("\nТоп-5 міст за споживанням:")
print(gdf.nlargest(5, "energy_GWh")[["name", "energy_GWh", "population"]].to_string(index=False))

# 4. Візуалізація
EXTENT = [21.5, 40.5, 44.0, 52.8]

def base_map(ax, title):
    ax.set_facecolor("#c8dff0")
    neighbors.plot(ax=ax, color="#e4ddd0", edgecolor="#aaa", linewidth=0.6, zorder=1)
    ukraine.plot(ax=ax, color="#f2efe8", edgecolor="#555", linewidth=1.5, zorder=2)
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Довгота", fontsize=9)
    ax.set_ylabel("Широта", fontsize=9)
    ax.grid(True, alpha=0.18, linestyle="--", color="#777")
    ax.tick_params(labelsize=8)


def label_large(ax, threshold=250_000):
    for _, row in gdf[gdf["population"] > threshold].iterrows():
        ax.annotate(
            row["name"], xy=(row["lon"], row["lat"]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=7.5, fontweight="bold", color="#111", zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.55, ec="none"),
        )


# Карта 1: Енергоспоживання
fig1, ax1 = plt.subplots(figsize=(13, 8))
base_map(ax1, "Міста України: споживання електроенергії та населення")

sizes = (gdf["population"] / gdf["population"].max() * 500 + 20).clip(20, 520)
sc1 = ax1.scatter(
    gdf["lon"], gdf["lat"],
    c=gdf["energy_GWh"], s=sizes,
    cmap="YlOrRd",
    norm=Normalize(gdf["energy_GWh"].min(), gdf["energy_GWh"].max()),
    edgecolors="#444", linewidths=0.5, alpha=0.88, zorder=4,
)
cbar1 = plt.colorbar(sc1, ax=ax1, shrink=0.55, pad=0.02)
cbar1.set_label("Споживання (ГВт·год)", fontsize=9)
label_large(ax1)

for pop, lbl in [(50_000, "50k"), (200_000, "200k"), (1_000_000, "1M")]:
    s = pop / gdf["population"].max() * 500 + 20
    ax1.scatter([], [], s=s, c="#bbb", edgecolors="#555",
                linewidths=0.5, alpha=0.85, label=lbl)
ax1.legend(title="Населення", loc="lower left",
           fontsize=8, title_fontsize=8.5, framealpha=0.85)

plt.tight_layout()
plt.savefig("map_energy.png", dpi=160, bbox_inches="tight")
plt.close()

# Карта 2: Буфери та відстані
fig2, ax2 = plt.subplots(figsize=(13, 8))
base_map(ax2, "Класи міст, буферні зони 40 км та відстані від Києва")

buf_wgs = large.set_geometry("buffer_geom").to_crs("EPSG:4326")
buf_wgs.plot(ax=ax2, color="#4dac26", alpha=0.13,
             edgecolor="#4dac26", linewidth=0.9, zorder=3)

kyiv_row = gdf[gdf["name"] == "Kyiv"].iloc[0]
for _, row in gdf[gdf["population"] > 300_000].iterrows():
    if row["name"] == "Kyiv":
        continue
    ax2.plot([kyiv_row["lon"], row["lon"]],
             [kyiv_row["lat"], row["lat"]],
             color="#888", linewidth=0.7, linestyle="--", alpha=0.4, zorder=3)
    dist_val = gdf_utm.loc[gdf_utm["name"] == row["name"], "dist_kyiv_km"]
    if len(dist_val):
        ax2.text(
            (kyiv_row["lon"] + row["lon"]) / 2,
            (kyiv_row["lat"] + row["lat"]) / 2,
            f"{dist_val.values[0]:.0f}km",
            fontsize=6, color="#555", ha="center", zorder=5,
        )

class_colors = {
    "Мале\n(<100k)": "#2c7bb6",
    "Середнє\n(100–300k)": "#74c476",
    "Велике\n(300k–800k)": "#fd8d3c",
    "Мегаполіс\n(>800k)": "#d62728",
}
for cls, color in class_colors.items():
    sub = gdf[gdf["size_class"] == cls]
    ax2.scatter(sub["lon"], sub["lat"], c=color, s=35,
                edgecolors="#333", linewidths=0.4, zorder=4,
                label=cls.replace("\n", " "))

label_large(ax2)

buf_patch = mpatches.Patch(facecolor="#4dac26", alpha=0.3,
                           edgecolor="#4dac26", label="Буфер 40 км")
handles, labels_leg = ax2.get_legend_handles_labels()
ax2.legend(handles + [buf_patch], labels_leg + ["Буфер 40 км"],
           title="Клас міста", loc="lower left",
           fontsize=8, title_fontsize=8.5, framealpha=0.85)

plt.tight_layout()
plt.savefig("map_buffers.png", dpi=160, bbox_inches="tight")
plt.close()

# Карта 3: ВДЕ
fig3, ax3 = plt.subplots(figsize=(13, 8))
base_map(ax3, "Частка відновлювальних джерел енергії (ВДЕ) по містах")

sc3 = ax3.scatter(
    gdf["lon"], gdf["lat"],
    c=gdf["renewable_pct"], s=60,
    cmap="RdYlGn", norm=Normalize(0, 50),
    edgecolors="#333", linewidths=0.4, alpha=0.9, zorder=4,
)
cbar3 = plt.colorbar(sc3, ax=ax3, shrink=0.55, pad=0.02)
cbar3.set_label("Частка ВДЕ (%)", fontsize=9)

for _, row in gdf[gdf["population"] > 150_000].iterrows():
    ax3.annotate(
        f"{row['name']}\n{row['renewable_pct']:.0f}%",
        xy=(row["lon"], row["lat"]),
        xytext=(5, 3), textcoords="offset points",
        fontsize=6.5, color="#111", zorder=5,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.5, ec="none"),
    )

plt.tight_layout()
plt.savefig("map_renewable.png", dpi=160, bbox_inches="tight")
plt.close()

# Графік 4: Аналітика
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))
fig4.suptitle("Аналіз атрибутів міст України", fontsize=13, fontweight="bold")

for cls, color in class_colors.items():
    sub = gdf[gdf["size_class"] == cls]
    axes4[0].scatter(sub["population"] / 1e3, sub["energy_GWh"],
                     c=color, s=35, alpha=0.8,
                     edgecolors="#333", linewidths=0.3,
                     label=cls.replace("\n", " "))
xr = np.linspace(gdf["population"].min(), gdf["population"].max(), 100)
axes4[0].plot(xr / 1e3,
              np.poly1d(np.polyfit(gdf["population"], gdf["energy_GWh"], 1))(xr),
              "k--", linewidth=1.1, alpha=0.5)
axes4[0].set_xlabel("Населення (тис.)")
axes4[0].set_ylabel("Споживання (ГВт·год)")
axes4[0].set_title("Населення vs Споживання", fontweight="bold")
axes4[0].text(0.05, 0.92, f"r = {corr:.3f}", transform=axes4[0].transAxes,
              fontsize=9, fontweight="bold")
axes4[0].legend(fontsize=7, title="Клас", title_fontsize=7)
axes4[0].grid(True, alpha=0.25)

top10 = gdf.nlargest(10, "renewable_pct")
axes4[1].barh(top10["name"], top10["renewable_pct"],
              color=plt.cm.RdYlGn(top10["renewable_pct"].values / 50),
              edgecolor="white", linewidth=0.5)
axes4[1].set_xlabel("Частка ВДЕ (%)")
axes4[1].set_title("Топ-10 міст за часткою ВДЕ", fontweight="bold")
axes4[1].invert_yaxis()
axes4[1].grid(True, alpha=0.25, axis="x")

sc_a = axes4[2].scatter(
    gdf_utm["dist_kyiv_km"], gdf["renewable_pct"],
    c=gdf["population"] / 1e3, cmap="Blues",
    s=40, edgecolors="#333", linewidths=0.3, alpha=0.85,
)
plt.colorbar(sc_a, ax=axes4[2], label="Населення (тис.)")
axes4[2].set_xlabel("Відстань від Києва (км)")
axes4[2].set_ylabel("Частка ВДЕ (%)")
axes4[2].set_title("Відстань від Києва vs ВДЕ", fontweight="bold")
axes4[2].grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("analytics.png", dpi=160, bbox_inches="tight")
plt.close()

# 5. Підсумок
print("\n  Підсумок")
print(f"  Міст у датасеті:            {len(gdf)}")
print(f"  Буферних зон 40 км:         {len(large)}")
print(f"  Міст у буфері Києва:        {len(in_buf)}")
print(f"  Найдальше від Києва:        "
      f"{gdf_utm.nlargest(1, 'dist_kyiv_km')['name'].values[0]} "
      f"({gdf_utm['dist_kyiv_km'].max():.0f} км)")
second_closest = gdf_utm.nsmallest(2, 'dist_kyiv_km').iloc[1]
print(f"  Найближче до Києва:         "
      f"{second_closest['name']} "
      f"({second_closest['dist_kyiv_km']:.0f} км)")
print(f"  Найбільше споживання:       "
      f"{gdf.nlargest(1, 'energy_GWh')['name'].values[0]} "
      f"({gdf['energy_GWh'].max()} ГВт·год)")
print(f"  Кореляція нас./споживання:  {corr:.3f}")