import numpy as np, random                           # numpy для роботи з масивами, random для випадкових виборів
from sklearn.neural_network import MLPRegressor      # MLPRegressor — багатошаровий персептрон для регресії
from sklearn.model_selection import TimeSeriesSplit  # TimeSeriesSplit — крос-валідація для часових рядів
from sklearn.metrics import mean_absolute_error      # mean_absolute_error — метрика MAE

def evaluate_load_features(X, y, mask, n_splits=3, max_iter=200):
    """Оцінка підмножини ознак навантаження енергосистеми."""
    cols_idx = np.where(mask == 1)[0]                # Індекси ознак, які вибрані (mask == 1)
    if len(cols_idx) == 0:                           # Якщо жодної ознаки не вибрано
        return 1e9, 1e9, 1e9                         # Повертаємо дуже великі похибки (щоб відсіяти)

    Xs = X[:, cols_idx]                              # Вибираємо лише потрібні ознаки
    tscv = TimeSeriesSplit(n_splits=n_splits)        # Крос-валідація для часових рядів
    maes, rmses = [], []                             # Списки для збереження похибок

    for tr, val in tscv.split(Xs):                   # Для кожного розбиття train/val
        model = MLPRegressor(hidden_layer_sizes=(32,), # Мережа з одним прихованим шаром на 32 нейрони
                             max_iter=max_iter,
                             random_state=0)
        model.fit(Xs[tr], y[tr])                     # Навчаємо модель на train
        pred = model.predict(Xs[val])                # Прогнозуємо на val
        maes.append(mean_absolute_error(y[val], pred))   # Обчислюємо MAE
        rmses.append(np.sqrt(np.mean((y[val] - pred) ** 2))) # Обчислюємо RMSE

    return np.mean(maes), np.mean(rmses), np.std(maes) # Повертаємо середні MAE, RMSE і стандартне відхилення MAE

def ga_load_feature_selection(X, y, cols,
                              pop_size=8, n_gen=5, mutation_rate=0.2,
                              max_iter=200, cv_splits=3,
                              progress_cb=None):
    """Генетичний алгоритм для відбору інформативних ознак навантаження."""
    rng = np.random.RandomState(123)                 # Генератор випадкових чисел для відтворюваності
    n_features = X.shape[1]                          # Кількість ознак у датасеті
    pop = rng.randint(0, 2, size=(pop_size, n_features)) # Початкова популяція: випадкові бінарні маски
    best = None                                      # Найкраще рішення (буде оновлюватись)

    for gen in range(n_gen):                         # Для кожного покоління
        evals = []                                   # Список оцінених рішень
        for ind in pop:                              # Для кожної особини (маски ознак)
            mae, rmse, std_mae = evaluate_load_features(X, y, ind,
                                                        n_splits=cv_splits,
                                                        max_iter=max_iter)
            evals.append((mae, rmse, std_mae, ind.sum(), ind)) # Зберігаємо результат: MAE, RMSE, std, кількість ознак, маска

        evals.sort(key=lambda x: x[0])               # Сортуємо за MAE (мінімізуємо)
        mae, rmse, std_mae, nf, mask = evals[0]      # Беремо найкраще рішення
        best = {
            "mae": mae,
            "rmse": rmse,
            "std_mae": std_mae,
            "n_features": nf,
            "features": [cols[i] for i in np.where(mask == 1)[0]], # Список назв вибраних ознак
            "mask": mask
        }

        if progress_cb:                              # Якщо передано callback для прогресу
            progress_cb(gen, mae, rmse, nf)

        # Створюємо нову популяцію
        new_pop = [mask]                             # Починаємо з найкращої маски (елітний відбір)
        while len(new_pop) < pop_size:               # Поки не заповнили популяцію
            p1, p2 = random.choice(evals)[4], random.choice(evals)[4] # Вибираємо двох батьків (маски)
            cx = rng.randint(1, n_features - 1)      # Точка кросоверу
            child = np.concatenate([p1[:cx], p2[cx:]]) # Дитина: частина від p1, частина від p2
            mut = rng.rand(n_features) < mutation_rate # Випадкові мутації
            child[mut] = 1 - child[mut]              # Інвертуємо біти у місцях мутації
            new_pop.append(child)                    # Додаємо дитину у нову популяцію
        pop = np.array(new_pop)                      # Оновлюємо популяцію

    return best                                      # Повертаємо найкраще знайдене рішення
