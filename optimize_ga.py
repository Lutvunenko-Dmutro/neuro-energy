import numpy as np, random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed

def evaluate_arch(X, y, arch, n_splits=3, max_iter=200, stop_flag=lambda: False):
    """Оцінка архітектури для Парето-фронту"""
    if stop_flag():
        return 1e9, 1e9, 1e9
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []
    for tr, val in tscv.split(X):
        if stop_flag():
            return 1e9, 1e9, 1e9
        model = MLPRegressor(hidden_layer_sizes=arch,
                             max_iter=max_iter,
                             random_state=0)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[val])
        maes.append(mean_absolute_error(y[val], pred))
        rmses.append(np.sqrt(np.mean((y[val]-pred)**2)))
    return np.mean(maes), np.mean(rmses), sum(arch)

def random_arch(rng):
    """Генерує випадкову архітектуру"""
    n_layers = rng.randint(1, 4)  # 1–3 шари
    return tuple(rng.randint(8, 64) for _ in range(n_layers))

def pareto_front(solutions):
    """Фільтрує список словників і повертає лише недоміновані рішення"""
    front = []
    for s in solutions:
        dominated = False
        for t in solutions:
            if (t["mae"] <= s["mae"] and t["rmse"] <= s["rmse"]) and (t != s):
                dominated = True
                break
        if not dominated:
            front.append(s)
    return front

def ga_optimize_structure(X, y,
                          pop_size, n_gen, max_iter, cv_splits,
                          progress_cb=None, stop_flag=lambda: False):
    rng = np.random.RandomState(123)
    pop = [random_arch(rng) for _ in range(pop_size)]
    pareto = []

    for gen in range(n_gen):
        if stop_flag():
            break
        if progress_cb:
            progress_cb(gen, None, None, None, None)

        scored = Parallel(n_jobs=-1, backend="threading")(
            delayed(lambda arch: (evaluate_arch(X, y, arch,
                                                n_splits=cv_splits,
                                                max_iter=max_iter,
                                                stop_flag=stop_flag), arch))(arch)
            for arch in pop
        )

        if stop_flag():
            break

        scored = [(mae, rmse, n_params, arch) for (mae, rmse, n_params), arch in scored]

        # додаємо всі рішення у список
        for mae, rmse, n_params, arch in scored:
            pareto.append({"mae": mae, "rmse": rmse, "n_params": n_params, "arch": arch})

        # сортуємо за MAE
        scored.sort(key=lambda x: x[0])

        if progress_cb:
            best = scored[0]
            if progress_cb(gen, best[0], best[1], best[2], best[3]) is False:
                break

        # нова популяція
        new_pop = [scored[0][3]]  # еліта
        while len(new_pop) < pop_size:
            if stop_flag():
                break
            p1, p2 = random.choice(scored[:5])[3], random.choice(scored[:5])[3]
            min_len = min(len(p1), len(p2))
            if min_len > 1:
                cut = rng.randint(1, min_len)
                child = p1[:cut] + p2[cut:]
            else:
                # якщо обидва з 1 шаром — просто копіюємо одного з батьків
                child = random.choice([p1, p2])

            # мутація з підвищеною ймовірністю
            if rng.rand() < 0.6:
                idx = rng.randint(len(child))
                child = list(child)
                child[idx] = rng.randint(8, 64)
                child = tuple(child)

            # гарантуємо, що хоч щось зміниться
            if child == p1 or child == p2:
                idx = rng.randint(len(child))
                child = list(child)
                child[idx] = rng.randint(8, 64)
                child = tuple(child)

            new_pop.append(child)
        pop = new_pop

    # повертаємо тільки недоміновані рішення
    return pareto_front(pareto)