import numpy as np, random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed

def evaluate_params(X, y, hidden, lr, alpha, n_splits=3, max_iter=200, stop_flag=lambda: False):
    """Оцінка параметрів моделі"""
    if stop_flag():
        return 1e9, 1e9
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []
    for tr, val in tscv.split(X):
        if stop_flag():
            return 1e9, 1e9
        model = MLPRegressor(hidden_layer_sizes=(hidden,),
                             learning_rate_init=lr,
                             alpha=alpha,
                             max_iter=max_iter,
                             random_state=0)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[val])
        maes.append(mean_absolute_error(y[val], pred))
        rmses.append(np.sqrt(np.mean((y[val]-pred)**2)))
    return np.mean(maes), np.mean(rmses)

def ga_param_search(X, y,
                    pop_size, n_gen, max_iter, cv_splits,
                    progress_cb=None, stop_flag=lambda: False):
    rng = np.random.RandomState(123)

    def random_ind():
        return [rng.randint(16, 64), 10**rng.uniform(-4, -2), 10**rng.uniform(-6, -3)]

    pop = [random_ind() for _ in range(pop_size)]
    best = None

    for gen in range(n_gen):
        if stop_flag():
            break
        if progress_cb:
            progress_cb(gen, None, None, None)

        scored = Parallel(n_jobs=-1, backend="threading")(
            delayed(lambda ind: (evaluate_params(X, y, *ind,
                                                 n_splits=cv_splits,
                                                 max_iter=max_iter,
                                                 stop_flag=stop_flag), ind))(ind)
            for ind in pop
        )

        if stop_flag():
            break

        scored = [(mae, rmse, ind) for (mae, rmse), ind in scored]
        scored.sort(key=lambda x: x[0])  # мінімізуємо MAE

        if best is None or scored[0][0] < best[0]:
            best = scored[0]

        if progress_cb:
            if progress_cb(gen, scored[0][0], scored[0][1], scored[0][2]) is False:
                break

        new_pop = [scored[0][2]]
        while len(new_pop) < pop_size:
            if stop_flag():
                break
            p1, p2 = random.choice(scored[:5])[2], random.choice(scored[:5])[2]
            child = [random.choice([p1[0], p2[0]]),
                     random.choice([p1[1], p2[1]]),
                     random.choice([p1[2], p2[2]])]
            if rng.rand() < 0.3:
                child[0] = rng.randint(16, 64)
            if rng.rand() < 0.3:
                child[1] = 10**rng.uniform(-4, -2)
            if rng.rand() < 0.3:
                child[2] = 10**rng.uniform(-6, -3)
            new_pop.append(child)
        pop = new_pop

    return {"mae": best[0], "rmse": best[1], "params": best[2]} if best else None