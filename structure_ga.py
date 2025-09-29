import numpy as np, random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed

def evaluate_structure(X, y, arch, n_splits=3, max_iter=200, stop_flag=lambda: False):
    """Оцінка архітектури мережі"""
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

def ga_structure_search(X, y,
                        pop_size, n_gen, max_iter, cv_splits,
                        progress_cb=None, stop_flag=lambda: False):
    rng = np.random.RandomState(123)
    pop = [random_arch(rng) for _ in range(pop_size)]
    best = None

    for gen in range(n_gen):
        if stop_flag():
            break
        if progress_cb:
            progress_cb(gen, None, None, None, None)

        scored = Parallel(n_jobs=-1, backend="threading")(
            delayed(lambda arch: (evaluate_structure(X, y, arch,
                                                     n_splits=cv_splits,
                                                     max_iter=max_iter,
                                                     stop_flag=stop_flag), arch))(arch)
            for arch in pop
        )

        if stop_flag():
            break

        scored = [(mae, rmse, n_params, arch) for (mae, rmse, n_params), arch in scored]
        scored.sort(key=lambda x: x[0])  # мінімізуємо MAE

        if best is None or scored[0][0] < best[0]:
            best = scored[0]

        if progress_cb:
            if progress_cb(gen, scored[0][0], scored[0][1], scored[0][2], scored[0][3]) is False:
                break

        # нова популяція
        new_pop = [scored[0][3]]
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

            # мутація: випадково змінюємо кількість нейронів
            if rng.rand() < 0.3:
                idx = rng.randint(len(child))
                child = list(child)
                child[idx] = rng.randint(8, 64)
                child = tuple(child)

            new_pop.append(child)
        pop = new_pop

    return {"mae": best[0], "rmse": best[1], "n_params": best[2], "arch": best[3]} if best else None