# Відображення ключів датасетів у файли
DATASET_PATHS = {
    "s1": "data/s1_hourly.csv",
    "s2": "data/s2_daily.csv",
    "s3": "data/s3_hourly_vre.csv",
    "s4": "data/s4_shift.csv"
}

# Людяні назви датасетів для GUI
DATASET_NAMES = {
    "s1": "Hourly Load (S1)",
    "s2": "Daily Load (S2)",
    "s3": "Hourly VRE (S3)",
    "s4": "Shifted Load (S4)"
}

# Відображення режимів у зрозумілі назви
MODE_NAMES = {
    "features": "Відбір ознак",
    "params": "Параметричний синтез",
    "structure": "Структурний синтез",
    "opt": "Оптимізація структури (Парето)"
}