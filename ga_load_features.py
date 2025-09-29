import numpy as np, random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

def evaluate_load_features(X, y, mask, n_splits=3, max_iter=200):
    """Оцінка підмножини ознак навантаження енергосистеми."""
    cols_idx = np.where(mask == 1)[0]
    if len(cols_idx) == 0:
        return 1e9, 1e9, 1e9

    Xs = X[:, cols_idx]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []

    for tr, val in tscv.split(Xs):
        model = MLPRegressor(hidden_layer_sizes=(32,),
                             max_iter=max_iter,
                             random_state=0)
        model.fit(Xs[tr], y[tr])
        pred = model.predict(Xs[val])
        maes.append(mean_absolute_error(y[val], pred))
        rmses.append(np.sqrt(np.mean((y[val] - pred) ** 2)))

    return np.mean(maes), np.mean(rmses), np.std(maes)

def ga_load_feature_selection(X, y, cols,
                              pop_size=8, n_gen=5, mutation_rate=0.2,
                              max_iter=200, cv_splits=3,
                              progress_cb=None):
    """Генетичний алгоритм для відбору інформативних ознак навантаження."""
    rng = np.random.RandomState(123)
    n_features = X.shape[1]
    pop = rng.randint(0, 2, size=(pop_size, n_features))
    best = None

    for gen in range(n_gen):
        evals = []
        for ind in pop:
            mae, rmse, std_mae = evaluate_load_features(X, y, ind,
                                                        n_splits=cv_splits,
                                                        max_iter=max_iter)
            evals.append((mae, rmse, std_mae, ind.sum(), ind))

        evals.sort(key=lambda x: x[0])  # мінімізуємо MAE
        mae, rmse, std_mae, nf, mask = evals[0]
        best = {
            "mae": mae,
            "rmse": rmse,
            "std_mae": std_mae,
            "n_features": nf,
            "features": [cols[i] for i in np.where(mask == 1)[0]],
            "mask": mask
        }

        if progress_cb:
            progress_cb(gen, mae, rmse, nf)

        # Створюємо нову популяцію
        new_pop = [mask]
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(evals)[4], random.choice(evals)[4]
            cx = rng.randint(1, n_features - 1)
            child = np.concatenate([p1[:cx], p2[cx:]])
            mut = rng.rand(n_features) < mutation_rate
            child[mut] = 1 - child[mut]
            new_pop.append(child)
        pop = np.array(new_pop)

    return best