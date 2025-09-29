"""
Модуль для завантаження даних енергосистеми з CSV.
Повертає:
- X: матриця ознак (numpy array)
- y: цільовий вектор (numpy array, навантаження)
- cols: список назв ознак
"""

import pandas as pd
import numpy as np
from static.mappings import DATASET_PATHS

def load_dataset(key):
    """
    Завантажує датасет за ключем (s1, s2, s3, s4).
    Очікується, що остання колонка у CSV — це навантаження (y),
    решта колонок — ознаки (X).
    """
    path = DATASET_PATHS.get(key)
    if not path:
        raise ValueError(f"❌ Невідомий датасет: {key}")

    # Читаємо CSV
    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError(f"❌ У файлі {path} замало колонок для X і y")

    # Остання колонка = ціль (навантаження)
    cols = list(df.columns[:-1])
    X = df[cols].values.astype(np.float32)
    y = df[df.columns[-1]].values.astype(np.float32)

    return X, y, cols