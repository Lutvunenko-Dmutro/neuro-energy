"""
Генетичний алгоритм для параметричного синтезу моделі прогнозу
завантаженості енергосистеми.
"""

import numpy as np, random                           # numpy для числових обчислень, random для випадкових виборів
from sklearn.neural_network import MLPRegressor      # MLPRegressor — багатошаровий персептрон для регресії
from sklearn.model_selection import TimeSeriesSplit  # TimeSeriesSplit — крос-валідація для часових рядів
from sklearn.metrics import mean_absolute_error      # mean_absolute_error — метрика MAE

# Діапазони параметрів для оптимізації
HIDDEN_CHOICES = [16, 32, 64, 128]                   # Можливі кількості нейронів у прихованому шарі
LR_CHOICES = [0.001, 0.01, 0.05]                     # Можливі швидкості навчання
ALPHA_CHOICES = [0.0001, 0.001, 0.01]                # Можливі коефіцієнти регуляризації

def evaluate_params(X, y, hidden, lr, alpha, n_splits=3, max_iter=200):
    """
    Оцінка конкретного набору параметрів (hidden, lr, alpha).
    Використовує TimeSeriesSplit для крос-валідації.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)        # Розбиваємо дані на n_splits для крос-валідації
    maes, rmses = [], []                             # Списки для збереження похибок
    for tr, val in tscv.split(X):                    # Для кожного розбиття train/val
        model = MLPRegressor(hidden_layer_sizes=(hidden,), # Створюємо модель MLP з одним прихованим шаром
                             learning_rate_init=lr,        # Встановлюємо швидкість навчання
                             alpha=alpha,                  # Встановлюємо коефіцієнт регуляризації
                             max_iter=max_iter,            # Максимальна кількість ітерацій
                             random_state=0)               # Фіксуємо seed для відтворюваності
        model.fit(X[tr], y[tr])                      # Навчаємо модель на train
        pred = model.predict(X[val])                 # Прогнозуємо на val
        maes.append(mean_absolute_error(y[val], pred))   # Обчислюємо MAE
        rmses.append(np.sqrt(np.mean((y[val] - pred) ** 2))) # Обчислюємо RMSE
    return np.mean(maes), np.mean(rmses)             # Повертаємо середні значення MAE і RMSE

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
    rng = np.random.RandomState(42)                  # Генератор випадкових чисел для відтворюваності

    # Початкова популяція: випадкові комбінації параметрів
    pop = [(random.choice(HIDDEN_CHOICES),
            random.choice(LR_CHOICES),
            random.choice(ALPHA_CHOICES)) for _ in range(pop_size)]

    best = None                                      # Найкраще рішення (буде оновлюватись)

    for gen in range(n_gen):                         # Для кожного покоління
        evals = []                                   # Список оцінених рішень
        for hidden, lr, alpha in pop:                # Для кожного набору параметрів
            mae, rmse = evaluate_params(X, y, hidden, lr, alpha,
                                        n_splits=cv_splits, max_iter=max_iter)
            evals.append((mae, rmse, hidden, lr, alpha)) # Зберігаємо результат

        # Сортуємо за MAE (мінімізуємо)
        evals.sort(key=lambda x: x[0])
        mae, rmse, hidden, lr, alpha = evals[0]      # Беремо найкраще рішення
        best = {"mae": mae, "rmse": rmse,
                "hidden": hidden, "lr": lr, "alpha": alpha}

        if progress_cb:                              # Якщо передано callback для прогресу
            progress_cb(gen, mae, rmse, f"h={hidden}, lr={lr}, a={alpha}")

        # Нова популяція (елітний відбір + кросовер + мутації)
        new_pop = [(hidden, lr, alpha)]              # Починаємо з найкращого (елітний відбір)
        while len(new_pop) < pop_size:               # Поки не заповнили популяцію
            p1, p2 = random.choice(evals), random.choice(evals) # Вибираємо двох батьків
            child = [p1[2], p1[3], p1[4]]            # Дитина успадковує параметри від p1
            # Мутації: з певною ймовірністю змінюємо параметри
            if rng.rand() < mutation_rate:
                child[0] = random.choice(HIDDEN_CHOICES)
            if rng.rand() < mutation_rate:
                child[1] = random.choice(LR_CHOICES)
            if rng.rand() < mutation_rate:
                child[2] = random.choice(ALPHA_CHOICES)
            new_pop.append(tuple(child))             # Додаємо дитину у нову популяцію
        pop = new_pop                                # Оновлюємо популяцію

    return best                                      # Повертаємо найкраще знайдене рішення
