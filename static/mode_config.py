"""
Конфігурація режимів аналізу:
мапить назву режиму на відповідну функцію GA та опис.
"""

# Імпортуємо реалізації генетичних алгоритмів для різних етапів
from ga_load_features import ga_load_feature_selection          # Відбір ознак
from ga_model_params import ga_model_param_synthesis            # Параметричний синтез
from ga_network_structure import ga_network_structure_synthesis # Структурний синтез
from ga_multiobjective_opt import ga_multiobjective_optimization # Багатокритеріальна оптимізація

# Словник конфігурації режимів
MODE_CONFIG = {
    "features": {  # Режим відбору ознак
        "func": ga_load_feature_selection,                     # Функція, яка реалізує GA для відбору ознак
        "desc": "Відбір інформативних ознак навантаження"      # Людяний опис для GUI
    },
    "params": {   # Режим параметричного синтезу
        "func": ga_model_param_synthesis,                      # Функція GA для оптимізації параметрів (hidden, lr, alpha)
        "desc": "Параметричний синтез моделі прогнозу"
    },
    "structure": { # Режим структурного синтезу
        "func": ga_network_structure_synthesis,                # Функція GA для оптимізації архітектури (шари × нейрони)
        "desc": "Структурний синтез архітектури нейромережі"
    },
    "opt": {      # Режим багатокритеріальної оптимізації
        "func": ga_multiobjective_optimization,                # Функція GA для Парето-оптимізації (MAE, RMSE)
        "desc": "Багатокритеріальна оптимізація (Парето‑фронт)"
    }
}
