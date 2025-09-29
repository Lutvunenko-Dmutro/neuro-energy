"""
Багатокритеріальна оптимізація структури нейромережі
для прогнозування навантаження енергосистеми.
"""

import numpy as np, random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

LAYER_CHOICES = [1, 2, 3]
NEURON_CHOICES = [16, 32, 64, 128]

def evaluate_architecture(X, y, layers, neurons, n_splits=3, max_iter=200):
    """Оцінка архітектури за MAE і RMSE."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []
    hidden = tuple([neurons] * layers)
    for tr, val in tscv.split(X):
        model = MLPRegressor(hidden_layer_sizes=hidden,
                             max_iter=max_iter,
                             random_state=0)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[val])
        maes.append(mean_absolute_error(y[val], pred))
        rmses.append(np.sqrt(np.mean((y[val] - pred) ** 2)))
    return np.mean(maes), np.mean(rmses)

def dominates(a, b):
    """Перевірка, чи рішення a домінує над b (Парето)."""
    return (a["mae"] <= b["mae"] and a["rmse"] <= b["rmse"]) and (a["mae"] < b["mae"] or a["rmse"] < b["rmse"])

def ga_multiobjective_optimization(X, y,
                                   pop_size=10, n_gen=5, mutation_rate=0.2,
                                   max_iter=200, cv_splits=3,
                                   progress_cb=None):
    """
    Генетичний алгоритм для багатокритеріальної оптимізації архітектури.
    Повертає Парето‑фронт (список рішень).
    """
    rng = np.random.RandomState(21)
    pop = [(random.choice(LAYER_CHOICES), random.choice(NEURON_CHOICES)) for _ in range(pop_size)]
    pareto_front = []

    for gen in range(n_gen):
        evals = []
        for layers, neurons in pop:
            mae, rmse = evaluate_architecture(X, y, layers, neurons,
                                              n_splits=cv_splits, max_iter=max_iter)
            evals.append({"mae": mae, "rmse": rmse,
                          "layers": layers, "neurons": neurons})

        # Оновлюємо Парето‑фронт
        new_front = []
        for cand in evals:
            if not any(dominates(other, cand) for other in evals):
                new_front.append(cand)
        pareto_front = new_front

        if progress_cb:
            for cand in pareto_front:
                progress_cb(gen, cand["mae"], cand["rmse"], f"{cand['layers']}×{cand['neurons']}")

        # Нова популяція
        new_pop = [(c["layers"], c["neurons"]) for c in pareto_front]
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(evals), random.choice(evals)
            child_layers = p1["layers"] if rng.rand() > 0.5 else p2["layers"]
            child_neurons = p1["neurons"] if rng.rand() > 0.5 else p2["neurons"]
            if rng.rand() < mutation_rate:
                child_layers = random.choice(LAYER_CHOICES)
            if rng.rand() < mutation_rate:
                child_neurons = random.choice(NEURON_CHOICES)
            new_pop.append((child_layers, child_neurons))
        pop = new_pop

    return pareto_front