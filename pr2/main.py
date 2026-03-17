import pandas as pd
import numpy as np
import time
import re
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
end = time.time()

print("Час без кешування:", round(end - start, 3), "сек")

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
end = time.time()

print("Час з кешуванням:", round(end - start, 3), "сек")

# 5. Перевірка кількості реальних викликів функції

print("Унікальних caption:", df["caption"].nunique())
print("Розмір кешу:", cached_text_processing.cache_info())