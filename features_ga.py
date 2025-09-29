import numpy as np, random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

def evaluate_subset(X, y, mask, n_splits=3, max_iter=200, stop_flag=lambda: False):
    """Оцінка підмножини ознак за MAE, RMSE і stdMAE з підтримкою зупинки."""
    if stop_flag():
        return 1e9, 1e9, 1e9
    cols_idx = np.where(mask == 1)[0]
    if len(cols_idx) == 0:
        return 1e9, 1e9, 1e9

    Xs = X[:, cols_idx]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []

    for tr, val in tscv.split(Xs):
        if stop_flag():
            return 1e9, 1e9, 1e9
        model = MLPRegressor(hidden_layer_sizes=(32,),
                             max_iter=max_iter,
                             random_state=0)
        model.fit(Xs[tr], y[tr])
        pred = model.predict(Xs[val])
        maes.append(mean_absolute_error(y[val], pred))
        rmses.append(np.sqrt(np.mean((y[val] - pred) ** 2)))

    return np.mean(maes), np.mean(rmses), np.std(maes)


def ga_feature_selection(X, y, cols,
                         pop_size, n_gen, mutation_rate, max_iter, cv_splits,
                         progress_cb=None, stop_flag=lambda: False):
    """
    Генетичний алгоритм для відбору ознак (послідовна оцінка для швидкої зупинки).
    """
    rng = np.random.RandomState(123)
    n_features = X.shape[1]
    pop = rng.randint(0, 2, size=(pop_size, n_features))
    best = None

    for gen in range(n_gen):
        if stop_flag():
            break

        if progress_cb and progress_cb(gen, None, None, None) is False:
            break

        evals = []
        for ind in pop:
            if stop_flag():
                break
            mae, rmse, std_mae = evaluate_subset(X, y, ind,
                                                 n_splits=cv_splits,
                                                 max_iter=max_iter,
                                                 stop_flag=stop_flag)
            evals.append((mae, rmse, std_mae, ind.sum(), ind))
        if stop_flag() or not evals:
            break

        evals.sort(key=lambda x: x[0])
        mae, rmse, std_mae, nf, mask = evals[0]
        best = {
            "mae": mae,
            "rmse": rmse,
            "std_mae": std_mae,
            "n_features": nf,
            "features": [cols[i] for i in np.where(mask == 1)[0]],
            "mask": mask
        }

        if progress_cb and progress_cb(gen, mae, rmse, nf) is False:
            break

        # Створюємо нову популяцію з перевіркою stop_flag
        new_pop = [mask]
        while len(new_pop) < pop_size:
            if stop_flag():
                break
            p1, p2 = random.choice(evals)[4], random.choice(evals)[4]
            cx = rng.randint(1, n_features - 1)
            child = np.concatenate([p1[:cx], p2[cx:]])
            mut = rng.rand(n_features) < mutation_rate
            child[mut] = 1 - child[mut]
            new_pop.append(child)
        pop = np.array(new_pop)

    return best