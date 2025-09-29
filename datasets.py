"""
Модуль для завантаження даних енергосистеми з CSV.
Повертає:
- X: матриця ознак (numpy array)
- y: цільовий вектор (numpy array, навантаження)
- cols: список назв ознак
"""

import pandas as pd                     # pandas — для зручного читання CSV у DataFrame
import numpy as np                      # numpy — для роботи з масивами
from static.mappings import DATASET_PATHS  # Імпортуємо словник з шляхами до датасетів

def load_dataset(key):
    """
    Завантажує датасет за ключем (s1, s2, s3, s4).
    Очікується, що остання колонка у CSV — це навантаження (y),
    решта колонок — ознаки (X).
    """
    path = DATASET_PATHS.get(key)       # Отримуємо шлях до файлу за ключем
    if not path:                        # Якщо ключ некоректний
        raise ValueError(f"❌ Невідомий датасет: {key}")

    # Читаємо CSV у DataFrame
    df = pd.read_csv(path)

    if df.shape[1] < 2:                 # Якщо у файлі менше ніж 2 колонки
        raise ValueError(f"❌ У файлі {path} замало колонок для X і y")

    # Остання колонка = ціль (навантаження), решта — ознаки
    cols = list(df.columns[:-1])        # Список назв ознак (усі колонки крім останньої)
    X = df[cols].values.astype(np.float32)          # Матриця ознак у форматі numpy.float32
    y = df[df.columns[-1]].values.astype(np.float32) # Вектор цілі (остання колонка)

    return X, y, cols                   # Повертаємо ознаки, ціль і список назв ознак
