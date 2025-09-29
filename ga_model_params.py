"""
Генетичний алгоритм для параметричного синтезу моделі прогнозу
завантаженості енергосистеми.
"""

import numpy as np, random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Діапазони параметрів для оптимізації
HIDDEN_CHOICES = [16, 32, 64, 128]
LR_CHOICES = [0.001, 0.01, 0.05]
ALPHA_CHOICES = [0.0001, 0.001, 0.01]

def evaluate_params(X, y, hidden, lr, alpha, n_splits=3, max_iter=200):
    """
    Оцінка конкретного набору параметрів (hidden, lr, alpha).
    Використовує TimeSeriesSplit для крос-валідації.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []
    for tr, val in tscv.split(X):
        model = MLPRegressor(hidden_layer_sizes=(hidden,),
                             learning_rate_init=lr,
                             alpha=alpha,
                             max_iter=max_iter,
                             random_state=0)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[val])
        maes.append(mean_absolute_error(y[val], pred))
        rmses.append(np.sqrt(np.mean((y[val] - pred) ** 2)))
    return np.mean(maes), np.mean(rmses)

def ga_model_param_synthesis(X, y,
                             pop_size=8, n_gen=5, mutation_rate=0.2,
                             max_iter=200, cv_splits=3,
                             progress_cb=None):
    """
    Генетичний алгоритм для пошуку оптимальних параметрів моделі прогнозу.
    Оптимізує:
    - hidden (кількість нейронів у прихованому шарі)
    - lr (швидкість навчання)
    - alpha (коефіцієнт регуляризації)
    """
    rng = np.random.RandomState(42)

    # Початкова популяція
    pop = [(random.choice(HIDDEN_CHOICES),
            random.choice(LR_CHOICES),
            random.choice(ALPHA_CHOICES)) for _ in range(pop_size)]

    best = None

    for gen in range(n_gen):
        evals = []
        for hidden, lr, alpha in pop:
            mae, rmse = evaluate_params(X, y, hidden, lr, alpha,
                                        n_splits=cv_splits, max_iter=max_iter)
            evals.append((mae, rmse, hidden, lr, alpha))

        # Сортуємо за MAE (мінімізуємо)
        evals.sort(key=lambda x: x[0])
        mae, rmse, hidden, lr, alpha = evals[0]
        best = {"mae": mae, "rmse": rmse,
                "hidden": hidden, "lr": lr, "alpha": alpha}

        if progress_cb:
            progress_cb(gen, mae, rmse, f"h={hidden}, lr={lr}, a={alpha}")

        # Нова популяція (елітний відбір + кросовер + мутації)
        new_pop = [(hidden, lr, alpha)]
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(evals), random.choice(evals)
            child = [p1[2], p1[3], p1[4]]
            if rng.rand() < mutation_rate:
                child[0] = random.choice(HIDDEN_CHOICES)
            if rng.rand() < mutation_rate:
                child[1] = random.choice(LR_CHOICES)
            if rng.rand() < mutation_rate:
                child[2] = random.choice(ALPHA_CHOICES)
            new_pop.append(tuple(child))
        pop = new_pop

    return best