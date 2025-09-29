"""
Конфігурація режимів аналізу:
мапить назву режиму на відповідну функцію GA та опис.
"""

from ga_load_features import ga_load_feature_selection
from ga_model_params import ga_model_param_synthesis
from ga_network_structure import ga_network_structure_synthesis
from ga_multiobjective_opt import ga_multiobjective_optimization

MODE_CONFIG = {
    "features": {
        "func": ga_load_feature_selection,
        "desc": "Відбір інформативних ознак навантаження"
    },
    "params": {
        "func": ga_model_param_synthesis,
        "desc": "Параметричний синтез моделі прогнозу"
    },
    "structure": {
        "func": ga_network_structure_synthesis,
        "desc": "Структурний синтез архітектури нейромережі"
    },
    "opt": {
        "func": ga_multiobjective_optimization,
        "desc": "Багатокритеріальна оптимізація (Парето‑фронт)"
    }
}