"""
Багатокритеріальна оптимізація структури нейромережі
для прогнозування навантаження енергосистеми.
"""

import numpy as np, random                           # numpy для числових обчислень, random для випадкових виборів
from sklearn.neural_network import MLPRegressor      # MLPRegressor — багатошаровий персептрон для регресії
from sklearn.model_selection import TimeSeriesSplit  # TimeSeriesSplit — крос-валідація для часових рядів
from sklearn.metrics import mean_absolute_error      # mean_absolute_error — метрика MAE

# Можливі варіанти кількості шарів і кількості нейронів у шарі
LAYER_CHOICES = [1, 2, 3]                            # Кількість прихованих шарів
NEURON_CHOICES = [16, 32, 64, 128]                   # Кількість нейронів у кожному шарі

def evaluate_architecture(X, y, layers, neurons, n_splits=3, max_iter=200):
    """Оцінка архітектури за MAE і RMSE."""
    tscv = TimeSeriesSplit(n_splits=n_splits)        # Розбиваємо дані на n_splits для крос-валідації
    maes, rmses = [], []                             # Списки для збереження похибок
    hidden = tuple([neurons] * layers)               # Формуємо архітектуру: повторюємо neurons layers разів
    for tr, val in tscv.split(X):                    # Для кожного розбиття train/val
        model = MLPRegressor(hidden_layer_sizes=hidden,  # Створюємо модель MLP
                             max_iter=max_iter,
                             random_state=0)
        model.fit(X[tr], y[tr])                      # Навчаємо на train
        pred = model.predict(X[val])                 # Прогнозуємо на val
        maes.append(mean_absolute_error(y[val], pred))   # Обчислюємо MAE
        rmses.append(np.sqrt(np.mean((y[val] - pred) ** 2))) # Обчислюємо RMSE
    return np.mean(maes), np.mean(rmses)             # Повертаємо середні значення MAE і RMSE

def dominates(a, b):
    """Перевірка, чи рішення a домінує над b (Парето)."""
    # a домінує над b, якщо воно не гірше за обома критеріями і краще хоча б за одним
    return (a["mae"] <= b["mae"] and a["rmse"] <= b["rmse"]) and (a["mae"] < b["mae"] or a["rmse"] < b["rmse"])

def ga_multiobjective_optimization(X, y,
                                   pop_size=10, n_gen=5, mutation_rate=0.2,
                                   max_iter=200, cv_splits=3,
                                   progress_cb=None):
    """
    Генетичний алгоритм для багатокритеріальної оптимізації архітектури.
    Повертає Парето‑фронт (список рішень).
    """
    rng = np.random.RandomState(21)                  # Генератор випадкових чисел для відтворюваності
    # Початкова популяція: випадкові архітектури (layers, neurons)
    pop = [(random.choice(LAYER_CHOICES), random.choice(NEURON_CHOICES)) for _ in range(pop_size)]
    pareto_front = []                                # Початковий Парето-фронт порожній

    for gen in range(n_gen):                         # Для кожного покоління
        evals = []                                   # Список оцінених рішень
        for layers, neurons in pop:                  # Для кожної архітектури у популяції
            mae, rmse = evaluate_architecture(X, y, layers, neurons,
                                              n_splits=cv_splits, max_iter=max_iter)
            evals.append({"mae": mae, "rmse": rmse,  # Зберігаємо результат
                          "layers": layers, "neurons": neurons})

        # Оновлюємо Парето‑фронт
        new_front = []
        for cand in evals:                           # Для кожного кандидата
            if not any(dominates(other, cand) for other in evals):  # Якщо його ніхто не домінує
                new_front.append(cand)               # Додаємо у новий фронт
        pareto_front = new_front                     # Оновлюємо фронт

        if progress_cb:                              # Якщо передано callback для прогресу
            for cand in pareto_front:
                progress_cb(gen, cand["mae"], cand["rmse"], f"{cand['layers']}×{cand['neurons']}")

        # Нова популяція
        new_pop = [(c["layers"], c["neurons"]) for c in pareto_front]  # Починаємо з Парето-рішень
        while len(new_pop) < pop_size:               # Поки не заповнили популяцію
            p1, p2 = random.choice(evals), random.choice(evals)  # Вибираємо двох батьків
            # Кросовер: випадково беремо параметри від p1 або p2
            child_layers = p1["layers"] if rng.rand() > 0.5 else p2["layers"]
            child_neurons = p1["neurons"] if rng.rand() > 0.5 else p2["neurons"]
            # Мутація: з певною ймовірністю змінюємо параметри
            if rng.rand() < mutation_rate:
                child_layers = random.choice(LAYER_CHOICES)
            if rng.rand() < mutation_rate:
                child_neurons = random.choice(NEURON_CHOICES)
            new_pop.append((child_layers, child_neurons))  # Додаємо дитину у нову популяцію
        pop = new_pop                                    # Оновлюємо популяцію

    return pareto_front                                  # Повертаємо фінальний Парето-фронт
