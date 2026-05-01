import pandas as pd
import numpy as np
import time
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from functools import lru_cache

# 1. Генерація LAION-like датасету (2 млн записів)
np.random.seed(42)
unique_captions = [
    "A dog playing in the park",
    "A cat sitting on the sofa",
    "Sunset over the mountains",
    "A group of people at a concert",
    "Delicious homemade pizza on a table",
    "Modern city skyline at night",
    "A child riding a bicycle",
    "Beautiful beach with palm trees",
    "Laptop and coffee on desk",
    "Snow covered forest in winter"
]

N = 2_000_000
df = pd.DataFrame({
    "image_url": np.random.randint(1000000, 9999999, size=N).astype(str),
    "caption": np.random.choice(unique_captions, size=N),
    "similarity_score": np.random.random(size=N)
})
print("Dataset shape:", df.shape)

# 2. Дорога текстова трансформація (імітація NLP-обробки)
def expensive_text_processing(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    unique_words = sorted(set(words))
    return "_".join(unique_words)

# 3. Без кешування
start = time.time()
df["processed_no_cache"] = df["caption"].apply(expensive_text_processing)
time_no_cache = round(time.time() - start, 3)
print("Час без кешування:", time_no_cache, "сек")

# 4. З кешуванням (lru_cache)
@lru_cache(maxsize=None)
def cached_text_processing(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    unique_words = sorted(set(words))
    return "_".join(unique_words)

start = time.time()
df["processed_cached"] = df["caption"].apply(cached_text_processing)
time_cached = round(time.time() - start, 3)
print("Час з кешуванням:", time_cached, "сек")

cache_info = cached_text_processing.cache_info()
unique_count = df["caption"].nunique()
print("Унікальних caption:", unique_count)
print("Розмір кешу:", cache_info)

speedup = round(time_no_cache / time_cached, 1)

# Графіки
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(12, 12))
fig.suptitle("Аналіз ефективності кешування LRU (LAION-like датасет, 2 млн рядків)",
             fontsize=15, fontweight="bold", y=0.98)

colors = {"no_cache": "#E74C3C", "cached": "#2ECC71", "neutral": "#3498DB"}

# Графік 1: Порівняння часу виконання (бар-чарт)
ax1 = fig.add_subplot(2, 2, 1)
bars = ax1.bar(
    ["Без кешування", "З LRU-кешем"],
    [time_no_cache, time_cached],
    color=[colors["no_cache"], colors["cached"]],
    width=0.5, edgecolor="white", linewidth=1.5
)
for bar, val in zip(bars, [time_no_cache, time_cached]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             f"{val} с", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax1.set_title("Час виконання (секунди)", fontweight="bold")
ax1.set_ylabel("Секунди")
ax1.set_ylim(0, time_no_cache * 1.25)
ax1.annotate(f"Прискорення: ×{speedup}", xy=(0.5, 0.92),
             xycoords="axes fraction", ha="center",
             fontsize=11, color="#8E44AD", fontweight="bold")

# Графік 2: Горизонтальний бар — hits vs misses
ax2 = fig.add_subplot(2, 2, 2)
hits   = cache_info.hits
misses = cache_info.misses
ax2.barh(["Cache Hits", "Cache Misses"], [hits, misses],
         color=[colors["cached"], colors["no_cache"]],
         edgecolor="white", linewidth=1.2)
for i, val in enumerate([hits, misses]):
    ax2.text(val + N * 0.01, i, f"{val:,}", va="center", fontweight="bold")
ax2.set_title("Cache Hits vs Misses", fontweight="bold")
ax2.set_xlabel("Кількість викликів")
ax2.set_xlim(0, N * 1.15)

# Графік 3: Розподіл similarity_score (гістограма)
ax3 = fig.add_subplot(2, 2, 3)
ax3.hist(df["similarity_score"], bins=60, color=colors["neutral"],
         edgecolor="white", linewidth=0.4, alpha=0.85)
ax3.axvline(df["similarity_score"].mean(), color="#E74C3C",
            linestyle="--", linewidth=1.8, label=f'Mean: {df["similarity_score"].mean():.3f}')
ax3.axvline(df["similarity_score"].median(), color="#F39C12",
            linestyle=":", linewidth=1.8, label=f'Median: {df["similarity_score"].median():.3f}')
ax3.set_title("Розподіл Similarity Score", fontweight="bold")
ax3.set_xlabel("Similarity Score")
ax3.set_ylabel("Кількість записів")
ax3.legend(fontsize=9)

# Графік 4: Частота кожного caption (горизонтальний бар)
ax4 = fig.add_subplot(2, 2, 4)
caption_counts = df["caption"].value_counts().sort_values()
short_labels = [c.split()[0] + "…" + c.split()[-1] for c in caption_counts.index]
bar_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(caption_counts)))
ax4.barh(short_labels, caption_counts.values, color=bar_colors, edgecolor="white")
ax4.set_title("Частота унікальних Caption", fontweight="bold")
ax4.set_xlabel("Кількість записів")
expected = N / unique_count
ax4.axvline(expected, color="red", linestyle="--", linewidth=1.5,
            label=f"Очікувана: {expected:,.0f}")
ax4.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("lru_cache_analysis.png", dpi=150, bbox_inches="tight")
print("Графік збережено: lru_cache_analysis.png")