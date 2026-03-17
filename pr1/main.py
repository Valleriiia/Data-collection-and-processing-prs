import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# 1. Генерація тестових даних
np.random.seed(42)
n = 500

price    = np.random.normal(loc=500, scale=150, size=n).clip(50, 1200)
discount = 0.3 * (price - price.mean()) / price.std() * (-1) + np.random.normal(0, 0.08, n)
discount = (discount - discount.min()) / (discount.max() - discount.min()) * 0.5  # [0, 0.5]

quantity = -0.4 * (price / price.max()) + 0.25 * np.random.rand(n) + 0.6
quantity = (quantity * 200).clip(1, 300).astype(int).astype(float)

rating = 0.3 * (quantity / quantity.max()) - 0.2 * (price / price.max()) + np.random.normal(3.5, 0.4, n)
rating = rating.clip(1.0, 5.0)

df = pd.DataFrame({
    "price":    price,
    "quantity": quantity,
    "discount": discount,
    "rating":   rating
})

print("\nПерші 5 рядків датасету:")
print(df.head().to_string(index=False))

print("\nОписова статистика:")
print(df.describe().round(4).to_string())

# 2. Кореляційна матриця — numpy.corrcoef

print("\nМЕТОД 1: numpy.corrcoef")

data_matrix = df[["price", "quantity", "discount", "rating"]].values.T
numpy_corr = np.corrcoef(data_matrix)

columns = ["price", "quantity", "discount", "rating"]
numpy_corr_df = pd.DataFrame(numpy_corr, index=columns, columns=columns)

print("Кореляційна матриця (numpy.corrcoef):")
print(numpy_corr_df.round(6).to_string())

# 3. Кореляційна матриця — pandas.corr

print("\nМЕТОД 2: pandas.corr  (метод Пірсона)")

pandas_corr = df[["price", "quantity", "discount", "rating"]].corr(method="pearson")

print("Кореляційна матриця (pandas.corr):")
print(pandas_corr.round(6).to_string())

# 4. Порівняння результатів

print("\nПорівняння: різниця між матрицями")

diff = numpy_corr_df - pandas_corr
print("Матриця абсолютних різниць (numpy − pandas):")
print(diff.round(10).to_string())

max_diff = diff.abs().max().max()
print(f"\nМаксимальна абсолютна різниця: {max_diff:.2e}")

if max_diff < 1e-10:
    print("Висновок: результати ідентичні (різниця < 1e-10 — лише похибка float64)")
else:
    print("Увага: є значуща різниця між методами!")

# 5. Інтерпретація кореляцій

print("\nІнтерпретація кореляційних зв'язків")

pairs = [
    ("price",    "quantity"),
    ("price",    "discount"),
    ("price",    "rating"),
    ("quantity", "discount"),
    ("quantity", "rating"),
    ("discount", "rating"),
]

def interpret(r):
    a = abs(r)
    direction = "позитивна" if r > 0 else "негативна"
    if a >= 0.7:   strength = "сильна"
    elif a >= 0.4: strength = "помірна"
    elif a >= 0.2: strength = "слабка"
    else:          strength = "дуже слабка / відсутня"
    return f"{direction}, {strength} (r = {r:.4f})"

print(f"\n{'Пара змінних':<28} {'Кореляція (pandas)'}")
print("-" * 60)
for a, b in pairs:
    r = pandas_corr.loc[a, b]
    print(f"  {a:12} ↔ {b:12}  {interpret(r)}")

# 6. Візуалізація

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Кореляційний аналіз: price, quantity, discount, rating",
             fontsize=15, fontweight="bold", y=1.02)

cmap = sns.diverging_palette(220, 10, as_cmap=True)
common_kw = dict(annot=True, fmt=".4f", cmap=cmap,
                 vmin=-1, vmax=1, square=True,
                 linewidths=0.5, cbar_kws={"shrink": 0.8})

# Heatmap numpy
sns.heatmap(numpy_corr_df, ax=axes[0], **common_kw)
axes[0].set_title("numpy.corrcoef", fontsize=13, fontweight="bold")

# Heatmap pandas
sns.heatmap(pandas_corr, ax=axes[1], **common_kw)
axes[1].set_title("pandas.corr (Pearson)", fontsize=13, fontweight="bold")

# Heatmap різниці
diff_plot = (numpy_corr_df - pandas_corr).round(12)
sns.heatmap(diff_plot, ax=axes[2],
            annot=True, fmt=".2e",
            cmap="RdBu", center=0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[2].set_title("Різниця (numpy − pandas)", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("correlation_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nТеплові карти збережено: correlation_heatmaps.png")

# Scatter plot матриця
fig2, axes2 = plt.subplots(4, 4, figsize=(14, 14))
fig2.suptitle("Scatter Plot Matrix (попарні розсіювання)", fontsize=14, fontweight="bold")

cols = ["price", "quantity", "discount", "rating"]
colors = plt.cm.RdYlGn

for i, col_y in enumerate(cols):
    for j, col_x in enumerate(cols):
        ax = axes2[i][j]
        if i == j:
            ax.hist(df[col_x], bins=25, color="#4C72B0", edgecolor="white", alpha=0.85)
            ax.set_facecolor("#f8f8f8")
        else:
            r = pandas_corr.loc[col_y, col_x]
            c = colors((r + 1) / 2)
            ax.scatter(df[col_x], df[col_y], alpha=0.25, s=6, color=c)
            ax.text(0.05, 0.92, f"r={r:.3f}", transform=ax.transAxes,
                    fontsize=9, fontweight="bold",
                    color="red" if abs(r) > 0.4 else "black")
        if i == 3:
            ax.set_xlabel(col_x, fontsize=10, fontweight="bold")
        if j == 0:
            ax.set_ylabel(col_y, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig("scatter_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Scatter матрицю збережено: scatter_matrix.png")